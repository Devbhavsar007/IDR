import tempfile
from pathlib import Path
import torch

from ml.models.lightweight_model import LightweightMotionModel
from ml.export.quantize_onnx import export_lightweight_onnx


def test_lightweight_model_output_shapes():
    model = LightweightMotionModel()
    model.eval()

    batch_size = 4
    window_size = 100
    channels = 9
    x = torch.randn(batch_size, window_size, channels)

    with torch.no_grad():
        out = model(x)

    assert out.velocity.shape == (batch_size, 3)
    assert out.velocity_log_var.shape == (batch_size, 3)
    assert out.displacement.shape == (batch_size, 3)
    assert out.displacement_log_var.shape == (batch_size, 3)
    assert out.heading_change.shape == (batch_size, 1)
    assert out.heading_log_var.shape == (batch_size, 1)
    assert out.speed.shape == (batch_size, 1)
    assert out.speed_log_var.shape == (batch_size, 1)
    assert out.motion_class.shape == (batch_size, 14)


def test_lightweight_model_parameter_count():
    model = LightweightMotionModel()
    num_params = model.count_parameters()

    # Must be under 50K parameters for edge/mobile constraints
    assert 20000 < num_params < 50000, f"Expected < 50K parameters, got {num_params}"


def test_lightweight_model_gradient_flow():
    model = LightweightMotionModel()
    model.train()

    x = torch.randn(2, 100, 9)
    out = model(x)

    # Combined multi-task loss
    loss = (
        out.velocity.sum()
        + out.velocity_log_var.sum()
        + out.displacement.sum()
        + out.displacement_log_var.sum()
        + out.heading_change.sum()
        + out.speed.sum()
        + out.motion_class.sum()
    )
    loss.backward()

    # Stem and depthwise convs must receive non-zero gradients
    assert model.stem[0].weight.grad is not None
    assert model.stem[0].weight.grad.abs().sum() > 0.0


def test_export_lightweight_onnx_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_file = Path(tmpdir) / "test_lightweight.onnx"
        summary = export_lightweight_onnx(onnx_file)

        assert onnx_file.is_file()
        assert summary["edge_ready"] is True
        assert summary["parameter_count"] < 50000
        assert summary["fp32_size_kb"] < 300.0  # < 300 KB for FP32 weights
