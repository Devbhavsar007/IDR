"""
ONNX Export for IDR Unified Model.

Exports the trained PyTorch model to ONNX format with:
  - Dynamic batch size
  - Fixed window size (configurable)
  - FP16 / INT8 quantization options
  - SHA-256 checksum for model signing
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


def export_onnx(
    model: torch.nn.Module,
    output_path: str | Path,
    window_size: int = 200,
    input_channels: int = 9,
    opset_version: int = 17,
    fp16: bool = False,
) -> Path:
    """Export model to ONNX format.

    Args:
        model: Trained PyTorch model
        output_path: Output .onnx file path
        window_size: Fixed window size for export
        input_channels: Number of input channels
        opset_version: ONNX opset version
        fp16: If True, convert to FP16

    Returns:
        Path to the exported ONNX file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Dummy input
    dummy = torch.randn(1, window_size, input_channels)

    # Dynamic axes for batch dimension
    dynamic_axes = {"input": {0: "batch_size"}}
    output_names = [
        "velocity",
        "velocity_log_var",
        "displacement",
        "displacement_log_var",
        "heading_change",
        "heading_log_var",
        "speed",
        "speed_log_var",
        "motion_class",
    ]
    for name in output_names:
        dynamic_axes[name] = {0: "batch_size"}

    # Export
    torch.onnx.export(
        model,
        dummy,
        str(output_path),
        input_names=["input"],
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset_version,
        do_constant_folding=True,
    )

    logger.info(f"Exported ONNX model to {output_path}")

    # Verify
    import onnx

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verification passed")

    # Optional FP16 conversion
    if fp16:
        try:
            from onnxconverter_common import float16

            fp16_model = float16.convert_float_to_float16(onnx_model)
            fp16_path = output_path.with_suffix(".fp16.onnx")
            onnx.save(fp16_model, str(fp16_path))
            logger.info(f"FP16 model saved to {fp16_path}")
        except ImportError:
            logger.warning("onnxconverter-common not installed; skipping FP16 conversion")

    # Compute SHA-256 checksum
    sha256 = hashlib.sha256()
    with open(output_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    checksum = sha256.hexdigest()
    logger.info(f"SHA-256: {checksum}")

    # Save checksum
    checksum_path = output_path.with_suffix(".sha256")
    checksum_path.write_text(f"{checksum}  {output_path.name}\n")

    # Report size
    size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Model size: {size_mb:.2f} MB")

    return output_path


def verify_onnx_inference(
    onnx_path: str | Path,
    model: torch.nn.Module,
    window_size: int = 200,
    input_channels: int = 9,
    tolerance: float = 1e-4,
) -> bool:
    """Verify ONNX output matches PyTorch output.

    Returns True if outputs match within tolerance.
    """
    import onnxruntime as ort

    model.eval()
    dummy = torch.randn(1, window_size, input_channels)

    # PyTorch inference
    with torch.no_grad():
        pt_output = model(dummy)

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path))
    ort_inputs = {"input": dummy.numpy()}
    ort_outputs = session.run(None, ort_inputs)

    # Compare
    pt_values = [
        pt_output.velocity.numpy(),
        pt_output.velocity_log_var.numpy(),
        pt_output.displacement.numpy(),
        pt_output.displacement_log_var.numpy(),
        pt_output.heading_change.numpy(),
        pt_output.heading_log_var.numpy(),
        pt_output.speed.numpy(),
        pt_output.speed_log_var.numpy(),
        pt_output.motion_class.numpy(),
    ]

    all_close = True
    for i, (pt_val, ort_val) in enumerate(zip(pt_values, ort_outputs)):
        if not np.allclose(pt_val, ort_val, atol=tolerance):
            max_diff = np.max(np.abs(pt_val - ort_val))
            logger.warning(f"Output {i} mismatch: max_diff={max_diff:.6f}")
            all_close = False

    if all_close:
        logger.info("ONNX verification PASSED: all outputs match PyTorch")
    else:
        logger.warning("ONNX verification FAILED: outputs differ")

    return all_close


def main():
    parser = argparse.ArgumentParser(description="Export IDR model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="PyTorch checkpoint path")
    parser.add_argument("--output", default="models/idr_unified.onnx", help="Output path")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--fp16", action="store_true", help="Also export FP16 variant")
    parser.add_argument("--verify", action="store_true", help="Verify ONNX vs PyTorch")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from ml.models.unified_model import UnifiedModel

    model = UnifiedModel()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded checkpoint: {args.checkpoint}")

    onnx_path = export_onnx(
        model,
        args.output,
        window_size=args.window_size,
        fp16=args.fp16,
    )

    if args.verify:
        verify_onnx_inference(onnx_path, model, window_size=args.window_size)


if __name__ == "__main__":
    main()
