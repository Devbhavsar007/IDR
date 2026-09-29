"""
IDR Lightweight Edge Motion Model (LLIO-Net Style).

Architecture: Depthwise-Separable 1D Inverted Residual Blocks (MBConv-1D) + Squeeze-and-Excitation
with lightweight GRU and 5 multi-task heads.

Target: ~35K-45K parameters (<1MB ONNX), <1.2ms inference at FP16/INT8 on mobile CPUs.
Ideal for low-tier Indian Android devices and embedded edge microcontrollers.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.unified_model import ModelOutput


class SqueezeExcitation1D(nn.Module):
    """Squeeze-and-Excitation block for 1D temporal channel attention."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_ch = max(4, channels // reduction)
        self.fc1 = nn.Conv1d(channels, reduced_ch, 1)
        self.fc2 = nn.Conv1d(reduced_ch, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> pool over T -> (B, C, 1)
        w = F.adaptive_avg_pool1d(x, 1)
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w


class DepthwiseSeparableConvBlock(nn.Module):
    """Depthwise-Separable 1D Inverted Residual Block with dilation."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilation: int = 1,
        expand_ratio: int = 2,
    ):
        super().__init__()
        self.use_residual = in_ch == out_ch
        hidden_ch = in_ch * expand_ratio
        padding = (kernel_size - 1) * dilation // 2

        layers = []
        # Expansion pointwise conv
        if expand_ratio != 1:
            layers.extend([
                nn.Conv1d(in_ch, hidden_ch, 1, bias=False),
                nn.BatchNorm1d(hidden_ch),
                nn.ReLU6(inplace=True),
            ])

        # Depthwise conv
        layers.extend([
            nn.Conv1d(
                hidden_ch,
                hidden_ch,
                kernel_size,
                padding=padding,
                dilation=dilation,
                groups=hidden_ch,
                bias=False,
            ),
            nn.BatchNorm1d(hidden_ch),
            nn.ReLU6(inplace=True),
        ])

        # SE Attention
        layers.append(SqueezeExcitation1D(hidden_ch))

        # Projection pointwise conv
        layers.extend([
            nn.Conv1d(hidden_ch, out_ch, 1, bias=False),
            nn.BatchNorm1d(out_ch),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_residual:
            return x + self.block(x)
        return self.block(x)


class LightweightMotionModel(nn.Module):
    """
    Ultra-lightweight depthwise-separable multi-task motion model for edge deployment.
    """

    def __init__(
        self,
        in_channels: int = 9,
        base_channels: int = 24,
        gru_hidden: int = 32,
        num_classes: int = 14,
    ):
        super().__init__()
        self.in_channels = in_channels

        # Stem: project input to base channels
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU6(inplace=True),
        )

        # Stage 1: Dilation 1
        self.stage1 = DepthwiseSeparableConvBlock(base_channels, base_channels, kernel_size=3, dilation=1)
        # Stage 2: Dilation 2
        self.stage2 = DepthwiseSeparableConvBlock(base_channels, base_channels * 2, kernel_size=3, dilation=2)
        # Stage 3: Dilation 4
        self.stage3 = DepthwiseSeparableConvBlock(base_channels * 2, base_channels * 2, kernel_size=3, dilation=4)

        tcn_out_ch = base_channels * 2

        # Temporal GRU bridge
        self.gru = nn.GRU(
            input_size=tcn_out_ch,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )

        head_in = gru_hidden

        # Multi-task prediction heads
        self.velocity_head = nn.Linear(head_in, 3)
        self.velocity_var_head = nn.Linear(head_in, 3)

        self.displacement_head = nn.Linear(head_in, 3)
        self.displacement_var_head = nn.Linear(head_in, 3)

        self.heading_head = nn.Linear(head_in, 1)
        self.heading_var_head = nn.Linear(head_in, 1)

        self.speed_head = nn.Linear(head_in, 1)
        self.speed_var_head = nn.Linear(head_in, 1)

        self.class_head = nn.Linear(head_in, num_classes)

    def forward(self, x: torch.Tensor) -> ModelOutput:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, T, C) where C = 9 (accel, gyro, gravity)
        Returns:
            ModelOutput with all 5 task predictions
        """
        # Convert (B, T, C) -> (B, C, T) for Conv1D
        x_t = x.transpose(1, 2)

        feat = self.stem(x_t)
        feat = self.stage1(feat)
        feat = self.stage2(feat)
        feat = self.stage3(feat)

        # Convert back (B, C, T) -> (B, T, C) for GRU
        feat_gru_in = feat.transpose(1, 2)
        gru_out, _ = self.gru(feat_gru_in)

        # Take last time step embedding
        last_emb = gru_out[:, -1, :]

        return ModelOutput(
            velocity=self.velocity_head(last_emb),
            velocity_log_var=self.velocity_var_head(last_emb),
            displacement=self.displacement_head(last_emb),
            displacement_log_var=self.displacement_var_head(last_emb),
            heading_change=self.heading_head(last_emb),
            heading_log_var=self.heading_var_head(last_emb),
            speed=F.relu(self.speed_head(last_emb)),
            speed_log_var=self.speed_var_head(last_emb),
            motion_class=self.class_head(last_emb),
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
