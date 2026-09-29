"""
IDR Unified Multi-Task Motion Model.

Architecture: 1D-TCN + GRU backbone with 5 output heads:
  1. Velocity head      — 3D velocity (ENU) + 3D uncertainty
  2. Displacement head   — 3D relative displacement + 3D uncertainty
  3. Heading head        — heading change (scalar) + uncertainty
  4. Speed head          — scalar speed + uncertainty
  5. Motion class head   — 14-class softmax

Target: ~240K parameters, <5ms inference at FP16.
Reference: Blueprint v1.0 §6, TLIO architecture.

Input: (batch, window_size, channels) where channels = 9 (accel_xyz + gyro_xyz + gravity_xyz)
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModelOutput(NamedTuple):
    """All outputs from the unified model."""

    velocity: torch.Tensor        # (B, 3)  velocity ENU, m/s
    velocity_log_var: torch.Tensor  # (B, 3)  log-variance of velocity
    displacement: torch.Tensor    # (B, 3)  displacement ENU, m
    displacement_log_var: torch.Tensor  # (B, 3)
    heading_change: torch.Tensor  # (B, 1)  heading change, rad
    heading_log_var: torch.Tensor  # (B, 1)
    speed: torch.Tensor           # (B, 1)  scalar speed, m/s
    speed_log_var: torch.Tensor   # (B, 1)
    motion_class: torch.Tensor    # (B, 14) logits


class TemporalConvBlock(nn.Module):
    """Causal 1D temporal convolution block with residual connection."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # Causal padding

        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(0.1)

        self.residual = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.kernel_size = kernel_size
        self.dilation = dilation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. x: (B, C, T)"""
        # Causal: remove future samples after convolution
        trim = (self.kernel_size - 1) * self.dilation

        out = self.conv1(x)
        if trim > 0:
            out = out[:, :, :-trim]
        out = self.bn1(out)
        out = F.gelu(out)

        out = self.conv2(out)
        if trim > 0:
            out = out[:, :, :-trim]
        out = self.bn2(out)
        out = self.dropout(out)
        out = F.gelu(out)

        res = self.residual(x)
        return out + res


class UnifiedModel(nn.Module):
    """Multi-task motion model for IDR.

    Architecture:
        Input → TCN stack (4 blocks, dilations 1,2,4,8) → GRU → 5 heads

    Args:
        input_channels: Number of input channels (default 9: accel+gyro+gravity)
        tcn_channels: Hidden channels in TCN blocks
        gru_hidden: GRU hidden size
        num_classes: Number of motion classes
        dropout: Dropout probability (for MC Dropout uncertainty)
    """

    def __init__(
        self,
        input_channels: int = 9,
        tcn_channels: int = 64,
        gru_hidden: int = 64,
        num_classes: int = 14,
        dropout: float = 0.1,
    ):
        super().__init__()

        # ── TCN backbone ──
        self.tcn = nn.Sequential(
            TemporalConvBlock(input_channels, tcn_channels, kernel_size=3, dilation=1),
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=2),
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=4),
            TemporalConvBlock(tcn_channels, tcn_channels, kernel_size=3, dilation=8),
        )

        # ── GRU temporal aggregation ──
        self.gru = nn.GRU(
            input_size=tcn_channels,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # ── Output heads ──
        # Each head: Linear → ReLU → Linear
        feat_dim = gru_hidden

        # Velocity: 3 mean + 3 log-variance
        self.velocity_head = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 6),  # [vel_x, vel_y, vel_z, logvar_x, logvar_y, logvar_z]
        )

        # Displacement: 3 mean + 3 log-variance
        self.displacement_head = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 6),
        )

        # Heading change: 1 mean + 1 log-variance
        self.heading_head = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2),
        )

        # Speed: 1 mean + 1 log-variance
        self.speed_head = nn.Sequential(
            nn.Linear(feat_dim, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2),
        )

        # Motion classification: 14 logits
        self.class_head = nn.Sequential(
            nn.Linear(feat_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/He initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, C) where
               B = batch size, T = window size, C = channels (9)

        Returns:
            ModelOutput with all 5 head outputs.
        """
        # TCN expects (B, C, T)
        out = x.permute(0, 2, 1)
        out = self.tcn(out)

        # GRU expects (B, T, C)
        out = out.permute(0, 2, 1)
        gru_out, _ = self.gru(out)

        # Use last hidden state as feature
        features = self.dropout(gru_out[:, -1, :])  # (B, gru_hidden)

        # ── Heads ──
        vel_out = self.velocity_head(features)  # (B, 6)
        disp_out = self.displacement_head(features)  # (B, 6)
        hdg_out = self.heading_head(features)  # (B, 2)
        spd_out = self.speed_head(features)  # (B, 2)
        cls_out = self.class_head(features)  # (B, 14)

        return ModelOutput(
            velocity=vel_out[:, :3],
            velocity_log_var=vel_out[:, 3:],
            displacement=disp_out[:, :3],
            displacement_log_var=disp_out[:, 3:],
            heading_change=hdg_out[:, :1],
            heading_log_var=hdg_out[:, 1:],
            speed=spd_out[:, :1],
            speed_log_var=spd_out[:, 1:],
            motion_class=cls_out,
        )

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def estimate_onnx_size_mb(self) -> float:
        """Estimate ONNX model size in MB (FP32)."""
        total_params = sum(p.numel() for p in self.parameters())
        return total_params * 4 / (1024 * 1024)  # 4 bytes per float32


def gaussian_nll_loss(
    pred_mean: torch.Tensor,
    pred_log_var: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Gaussian negative log-likelihood loss.

    L = 0.5 * (log_var + (target - mean)² / exp(log_var))

    This encourages the model to output calibrated uncertainty:
    - When confident and correct: low log_var, low loss
    - When confident and wrong: low log_var, high loss (penalized!)
    - When uncertain: high log_var, moderate loss
    """
    # Clamp log_var for numerical stability
    log_var = torch.clamp(pred_log_var, min=-10.0, max=10.0)
    precision = torch.exp(-log_var)
    return 0.5 * torch.mean(log_var + precision * (target - pred_mean) ** 2)


class MultiTaskLoss(nn.Module):
    """Multi-task loss with learnable task weights.

    Uses homoscedastic uncertainty weighting (Kendall et al., 2018):
    L_total = Σ (1 / (2σ²_i)) * L_i + log(σ_i)
    """

    def __init__(self, num_tasks: int = 5):
        super().__init__()
        # Learnable log-variances for task weighting
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(
        self,
        output: ModelOutput,
        target_velocity: torch.Tensor,
        target_displacement: torch.Tensor,
        target_heading: torch.Tensor,
        target_speed: torch.Tensor,
        target_motion_class: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute weighted multi-task loss.

        Returns:
            total_loss: scalar
            losses: dict of individual losses for logging
        """
        # Individual losses
        l_vel = gaussian_nll_loss(output.velocity, output.velocity_log_var, target_velocity)
        l_disp = gaussian_nll_loss(
            output.displacement, output.displacement_log_var, target_displacement
        )
        l_hdg = gaussian_nll_loss(output.heading_change, output.heading_log_var, target_heading)
        l_spd = gaussian_nll_loss(output.speed, output.speed_log_var, target_speed)

        losses = [l_vel, l_disp, l_hdg, l_spd]

        # Motion class loss (cross-entropy)
        if target_motion_class is not None:
            l_cls = F.cross_entropy(output.motion_class, target_motion_class)
            losses.append(l_cls)
        else:
            l_cls = torch.tensor(0.0, device=output.velocity.device)

        # Weighted combination
        total = torch.tensor(0.0, device=output.velocity.device)
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total = total + precision * loss + self.log_vars[i]

        loss_dict = {
            "velocity": l_vel.item(),
            "displacement": l_disp.item(),
            "heading": l_hdg.item(),
            "speed": l_spd.item(),
            "motion_class": l_cls.item(),
            "total": total.item(),
        }

        return total, loss_dict
