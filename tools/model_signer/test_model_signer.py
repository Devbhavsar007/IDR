"""Unit tests for IDR Model Signer."""

import tempfile
from pathlib import Path
from tools.model_signer.model_signer import sign_model, verify_model


def test_sign_and_verify_roundtrip():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "motion_model.onnx"
        model_file.write_bytes(b"dummy_onnx_model_content_bytes_12345")

        key = b"secret_test_key_xyz"
        manifest = sign_model(model_file, key)

        assert manifest["model_file"] == "motion_model.onnx"
        assert len(manifest["sha256"]) == 64
        assert "signature" in manifest

        # Verification with correct key
        assert verify_model(model_file, key) is True


def test_tampered_model_fails_verification():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "motion_model.onnx"
        model_file.write_bytes(b"original_model_bytes")

        key = b"secret_test_key"
        sign_model(model_file, key)

        # Tamper with the model file
        model_file.write_bytes(b"maliciously_modified_bytes")

        # Verification should fail
        assert verify_model(model_file, key) is False


def test_wrong_key_fails_verification():
    with tempfile.TemporaryDirectory() as tmpdir:
        model_file = Path(tmpdir) / "motion_model.onnx"
        model_file.write_bytes(b"model_bytes")

        sign_model(model_file, b"correct_key")

        # Verification with wrong key
        assert verify_model(model_file, b"wrong_key") is False
