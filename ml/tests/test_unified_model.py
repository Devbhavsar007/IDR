"""Tests for the unified multi-task motion model."""

import pytest
import torch

from ml.models.unified_model import (
    ModelOutput,
    MultiTaskLoss,
    UnifiedModel,
    gaussian_nll_loss,
)

BATCH_SIZE = 4
WINDOW_SIZE = 200
INPUT_CHANNELS = 9


@pytest.fixture
def model():
    return UnifiedModel(input_channels=INPUT_CHANNELS)


@pytest.fixture
def sample_input():
    return torch.randn(BATCH_SIZE, WINDOW_SIZE, INPUT_CHANNELS)


class TestUnifiedModel:
    def test_output_shapes(self, model, sample_input):
        output = model(sample_input)
        assert output.velocity.shape == (BATCH_SIZE, 3)
        assert output.velocity_log_var.shape == (BATCH_SIZE, 3)
        assert output.displacement.shape == (BATCH_SIZE, 3)
        assert output.displacement_log_var.shape == (BATCH_SIZE, 3)
        assert output.heading_change.shape == (BATCH_SIZE, 1)
        assert output.heading_log_var.shape == (BATCH_SIZE, 1)
        assert output.speed.shape == (BATCH_SIZE, 1)
        assert output.speed_log_var.shape == (BATCH_SIZE, 1)
        assert output.motion_class.shape == (BATCH_SIZE, 14)

    def test_parameter_count(self, model):
        n_params = model.count_parameters()
        # Should be in the ~100K-500K range
        assert 50_000 < n_params < 1_000_000, f"Parameter count out of range: {n_params}"
        print(f"Model parameters: {n_params:,}")

    def test_onnx_size_estimate(self, model):
        size_mb = model.estimate_onnx_size_mb()
        assert size_mb < 10.0, f"Model too large: {size_mb:.2f} MB"
        print(f"Estimated ONNX size (FP32): {size_mb:.2f} MB")

    def test_gradient_flow(self, model, sample_input):
        output = model(sample_input)
        loss = (output.velocity.sum() + output.velocity_log_var.sum() +
                output.displacement.sum() + output.displacement_log_var.sum() +
                output.heading_change.sum() + output.heading_log_var.sum() +
                output.speed.sum() + output.speed_log_var.sum() +
                output.motion_class.sum())
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert torch.isfinite(param.grad).all(), f"Non-finite gradient for {name}"

    def test_deterministic_eval(self, model, sample_input):
        model.eval()
        with torch.no_grad():
            out1 = model(sample_input)
            out2 = model(sample_input)
        assert torch.allclose(out1.velocity, out2.velocity, atol=1e-6)

    def test_different_window_sizes(self, model):
        """Model should handle variable-length input (within reason)."""
        for ws in [50, 100, 200, 400]:
            x = torch.randn(2, ws, INPUT_CHANNELS)
            output = model(x)
            assert output.velocity.shape == (2, 3)


class TestGaussianNllLoss:
    def test_zero_loss_at_prediction(self):
        pred_mean = torch.tensor([1.0, 2.0, 3.0])
        pred_log_var = torch.zeros(3)
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = gaussian_nll_loss(pred_mean, pred_log_var, target)
        # Loss should be minimal (just the log_var term = 0)
        assert loss.item() < 0.1

    def test_high_loss_at_wrong_prediction(self):
        pred_mean = torch.tensor([10.0, 20.0, 30.0])
        pred_log_var = torch.zeros(3)  # Confident
        target = torch.tensor([1.0, 2.0, 3.0])
        loss = gaussian_nll_loss(pred_mean, pred_log_var, target)
        assert loss.item() > 10.0

    def test_uncertainty_reduces_loss(self):
        pred_mean = torch.tensor([10.0])
        target = torch.tensor([1.0])

        # Confident and wrong → high loss
        loss_confident = gaussian_nll_loss(pred_mean, torch.tensor([0.0]), target)

        # Uncertain and wrong → lower loss
        loss_uncertain = gaussian_nll_loss(pred_mean, torch.tensor([5.0]), target)

        assert loss_uncertain < loss_confident


class TestMultiTaskLoss:
    def test_loss_computes(self, model, sample_input):
        criterion = MultiTaskLoss(num_tasks=5)
        output = model(sample_input)

        total, losses = criterion(
            output,
            target_velocity=torch.randn(BATCH_SIZE, 3),
            target_displacement=torch.randn(BATCH_SIZE, 3),
            target_heading=torch.randn(BATCH_SIZE, 1),
            target_speed=torch.rand(BATCH_SIZE, 1),
            target_motion_class=torch.randint(0, 14, (BATCH_SIZE,)),
        )

        assert torch.isfinite(total)
        assert "total" in losses
        assert "velocity" in losses

    def test_task_weights_are_learnable(self):
        criterion = MultiTaskLoss()
        assert criterion.log_vars.requires_grad
