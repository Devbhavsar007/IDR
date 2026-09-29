"""
IDR Model Signer & Integrity Verifier.

Cryptographically signs ONNX neural motion models and verifies integrity
before mobile / edge deployment to prevent tampering or adversarial corruption
as specified in IDR Blueprint Revision v1.1 §8–9.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict


def compute_sha256(filepath: str | Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def sign_model(
    model_path: str | Path,
    secret_key: bytes,
    author: str = "IDR Project",
    version: str = "1.0.0",
    description: str = "Unified TCN+GRU Motion Model",
) -> Dict[str, Any]:
    """Generate signed manifest for an ONNX model."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    sha256_hash = compute_sha256(path)
    file_size_bytes = path.stat().st_size

    manifest: Dict[str, Any] = {
        "model_file": path.name,
        "sha256": sha256_hash,
        "size_bytes": file_size_bytes,
        "author": author,
        "version": version,
        "description": description,
        "signed_at_utc": int(time.time()),
        "security_standard": "DPDP-Act-2023-Aligned",
    }

    # Compute HMAC-SHA256 signature over manifest fields (excluding signature itself)
    payload = f"{sha256_hash}:{file_size_bytes}:{version}:{author}".encode("utf-8")
    sig = hmac.new(secret_key, payload, hashlib.sha256).hexdigest()
    manifest["signature"] = sig

    manifest_path = path.with_suffix(".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def verify_model(
    model_path: str | Path,
    secret_key: bytes,
    manifest_path: str | Path | None = None,
) -> bool:
    """Verify an ONNX model against its signed manifest."""
    m_path = Path(model_path)
    if not m_path.is_file():
        print(f"Error: model file does not exist: {model_path}", file=sys.stderr)
        return False

    man_path = Path(manifest_path) if manifest_path else m_path.with_suffix(".manifest.json")
    if not man_path.is_file():
        print(f"Error: manifest file does not exist: {man_path}", file=sys.stderr)
        return False

    with open(man_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # 1. Verify file hash
    actual_hash = compute_sha256(m_path)
    expected_hash = manifest.get("sha256", "")
    if actual_hash != expected_hash:
        print(f"FAILED: Hash mismatch! Expected {expected_hash}, got {actual_hash}", file=sys.stderr)
        return False

    # 2. Verify HMAC signature
    author = manifest.get("author", "")
    version = manifest.get("version", "")
    file_size = manifest.get("size_bytes", 0)
    expected_sig = manifest.get("signature", "")

    payload = f"{actual_hash}:{file_size}:{version}:{author}".encode("utf-8")
    computed_sig = hmac.new(secret_key, payload, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(computed_sig, expected_sig):
        print("FAILED: Cryptographic signature verification failed!", file=sys.stderr)
        return False

    print(f"VERIFIED: Model '{m_path.name}' is authentic and unmodified (v{version}).")
    return True


def main():
    parser = argparse.ArgumentParser(description="Sign or verify IDR ONNX motion models.")
    parser.add_argument("action", choices=["sign", "verify"], help="Action to perform")
    parser.add_argument("model", help="Path to ONNX model file")
    parser.add_argument("--key", default="idr_default_developer_secret_key_2026", help="HMAC secret key")
    parser.add_argument("--manifest", default=None, help="Explicit path to manifest file (for verify)")

    args = parser.parse_args()
    key = args.key.encode("utf-8")

    if args.action == "sign":
        manifest = sign_model(args.model, key)
        print(f"Signed successfully. Manifest generated: {Path(args.model).with_suffix('.manifest.json')}")
        print(json.dumps(manifest, indent=2))
    elif args.action == "verify":
        ok = verify_model(args.model, key, args.manifest)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
