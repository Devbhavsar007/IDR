"""
IDR Training Pipeline.

Handles:
  - DataLoader creation from IO-VNBD windows
  - Training loop with multi-task loss
  - Validation and metric logging
  - Checkpoint saving
  - Learning rate scheduling
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ml.models.unified_model import ModelOutput, MultiTaskLoss, UnifiedModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    # Architecture
    input_channels: int = 9
    window_size: int = 200
    tcn_channels: int = 64
    gru_hidden: int = 64

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    warmup_epochs: int = 5

    # Scheduler
    lr_min: float = 1e-6
    lr_schedule: str = "cosine"  # "cosine" | "step" | "plateau"

    # Data
    train_split: float = 0.8
    val_split: float = 0.1
    # test_split = 1 - train - val

    # Checkpoint
    checkpoint_dir: str = "checkpoints"
    save_every_epochs: int = 10
    early_stopping_patience: int = 15

    # Device
    device: str = "auto"  # "auto" | "cuda" | "cpu"


class IdrDataset(Dataset):
    """PyTorch Dataset wrapping IDR training windows."""

    def __init__(self, windows: list[dict]):
        self.windows = windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict:
        w = self.windows[idx]
        # Concatenate IMU (6ch) + gravity (3ch) = 9 channels
        imu = w["imu"]  # (W, 6)
        gravity = w["gravity"]  # (W, 3)
        x = np.concatenate([imu, gravity], axis=1)  # (W, 9)

        return {
            "input": torch.from_numpy(x),
            "velocity": torch.from_numpy(w["label_velocity"]),
            "displacement": torch.from_numpy(w["label_displacement"]),
            "heading_change": torch.tensor([w["label_heading_change"]]),
            "speed": torch.tensor([w["label_speed"]]),
            "gps_coverage": torch.tensor([w["gps_coverage"]]),
        }


@dataclass
class TrainingMetrics:
    """Metrics for a single epoch."""

    epoch: int = 0
    train_loss: float = 0.0
    val_loss: float = 0.0
    val_velocity_mae: float = 0.0
    val_displacement_mae: float = 0.0
    val_speed_mae: float = 0.0
    val_heading_mae_deg: float = 0.0
    learning_rate: float = 0.0
    epoch_time_sec: float = 0.0
    best_val_loss: float = float("inf")


class Trainer:
    """IDR model trainer."""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Device
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        logger.info(f"Training on device: {self.device}")

        # Model
        self.model = UnifiedModel(
            input_channels=config.input_channels,
            tcn_channels=config.tcn_channels,
            gru_hidden=config.gru_hidden,
        ).to(self.device)
        logger.info(f"Model parameters: {self.model.count_parameters():,}")

        # Loss
        self.criterion = MultiTaskLoss(num_tasks=5).to(self.device)

        # Optimizer (includes loss task weights)
        self.optimizer = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Scheduler
        self.scheduler = self._create_scheduler()

        # State
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history: list[TrainingMetrics] = []

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LRScheduler:
        cfg = self.config
        if cfg.lr_schedule == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=cfg.epochs - cfg.warmup_epochs,
                eta_min=cfg.lr_min,
            )
        elif cfg.lr_schedule == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=30, gamma=0.5
            )
        else:
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=10, factor=0.5
            )

    def prepare_data(self, windows: list[dict]) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Split windows and create DataLoaders."""
        np.random.seed(42)
        indices = np.random.permutation(len(windows))

        n_train = int(len(windows) * self.config.train_split)
        n_val = int(len(windows) * self.config.val_split)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        train_windows = [windows[i] for i in train_idx]
        val_windows = [windows[i] for i in val_idx]
        test_windows = [windows[i] for i in test_idx]

        logger.info(
            f"Data split: train={len(train_windows)}, "
            f"val={len(val_windows)}, test={len(test_windows)}"
        )

        train_loader = DataLoader(
            IdrDataset(train_windows),
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            IdrDataset(val_windows),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        test_loader = DataLoader(
            IdrDataset(test_windows),
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["input"].to(self.device)
            target_vel = batch["velocity"].to(self.device)
            target_disp = batch["displacement"].to(self.device)
            target_hdg = batch["heading_change"].to(self.device)
            target_spd = batch["speed"].to(self.device)

            self.optimizer.zero_grad()
            output = self.model(x)

            loss, loss_dict = self.criterion(
                output,
                target_velocity=target_vel,
                target_displacement=target_disp,
                target_heading=target_hdg,
                target_speed=target_spd,
            )

            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> TrainingMetrics:
        """Validate and compute metrics."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        vel_errors = []
        disp_errors = []
        speed_errors = []
        heading_errors = []

        for batch in val_loader:
            x = batch["input"].to(self.device)
            target_vel = batch["velocity"].to(self.device)
            target_disp = batch["displacement"].to(self.device)
            target_hdg = batch["heading_change"].to(self.device)
            target_spd = batch["speed"].to(self.device)

            output = self.model(x)
            loss, _ = self.criterion(
                output, target_vel, target_disp, target_hdg, target_spd
            )

            total_loss += loss.item()
            n_batches += 1

            # MAE per task
            vel_errors.append((output.velocity - target_vel).abs().mean().item())
            disp_errors.append((output.displacement - target_disp).abs().mean().item())
            speed_errors.append((output.speed - target_spd).abs().mean().item())
            heading_errors.append(
                (output.heading_change - target_hdg).abs().mean().item()
            )

        metrics = TrainingMetrics()
        metrics.val_loss = total_loss / max(n_batches, 1)
        metrics.val_velocity_mae = float(np.mean(vel_errors)) if vel_errors else 0.0
        metrics.val_displacement_mae = float(np.mean(disp_errors)) if disp_errors else 0.0
        metrics.val_speed_mae = float(np.mean(speed_errors)) if speed_errors else 0.0
        metrics.val_heading_mae_deg = (
            float(np.mean(heading_errors)) * 180.0 / np.pi if heading_errors else 0.0
        )

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> list[TrainingMetrics]:
        """Full training loop."""
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            t0 = time.time()

            # Warmup LR
            if epoch <= self.config.warmup_epochs:
                warmup_lr = self.config.learning_rate * epoch / self.config.warmup_epochs
                for pg in self.optimizer.param_groups:
                    pg["lr"] = warmup_lr

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            metrics = self.validate(val_loader)
            metrics.epoch = epoch
            metrics.train_loss = train_loss
            metrics.learning_rate = self.optimizer.param_groups[0]["lr"]
            metrics.epoch_time_sec = time.time() - t0
            metrics.best_val_loss = self.best_val_loss

            # LR schedule
            if epoch > self.config.warmup_epochs:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(metrics.val_loss)
                else:
                    self.scheduler.step()

            # Checkpoint
            if metrics.val_loss < self.best_val_loss:
                self.best_val_loss = metrics.val_loss
                self.patience_counter = 0
                self._save_checkpoint(checkpoint_dir / "best.pt", epoch, metrics)
            else:
                self.patience_counter += 1

            if epoch % self.config.save_every_epochs == 0:
                self._save_checkpoint(
                    checkpoint_dir / f"epoch_{epoch:04d}.pt", epoch, metrics
                )

            self.history.append(metrics)

            # Log
            logger.info(
                f"Epoch {epoch:3d}/{self.config.epochs} | "
                f"train={train_loss:.4f} val={metrics.val_loss:.4f} | "
                f"vel_mae={metrics.val_velocity_mae:.3f} "
                f"spd_mae={metrics.val_speed_mae:.3f} "
                f"hdg_mae={metrics.val_heading_mae_deg:.1f}° | "
                f"lr={metrics.learning_rate:.2e} | "
                f"{metrics.epoch_time_sec:.1f}s"
            )

            # Early stopping
            if self.patience_counter >= self.config.early_stopping_patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={self.config.early_stopping_patience})"
                )
                break

        return self.history

    def _save_checkpoint(
        self, path: Path, epoch: int, metrics: TrainingMetrics
    ) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "criterion_state_dict": self.criterion.state_dict(),
                "best_val_loss": self.best_val_loss,
                "config": self.config,
                "metrics": {
                    "train_loss": metrics.train_loss,
                    "val_loss": metrics.val_loss,
                    "val_velocity_mae": metrics.val_velocity_mae,
                    "val_speed_mae": metrics.val_speed_mae,
                    "val_heading_mae_deg": metrics.val_heading_mae_deg,
                },
            },
            path,
        )
        logger.info(f"Checkpoint saved: {path} (val_loss={metrics.val_loss:.4f})")

    def load_checkpoint(self, path: str | Path) -> int:
        """Load a checkpoint. Returns the epoch number."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "criterion_state_dict" in checkpoint:
            self.criterion.load_state_dict(checkpoint["criterion_state_dict"])
        self.best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        epoch = checkpoint.get("epoch", 0)
        logger.info(f"Loaded checkpoint from epoch {epoch}")
        return epoch
