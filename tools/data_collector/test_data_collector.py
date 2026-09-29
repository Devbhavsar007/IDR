"""Unit tests for IDR Data Collector and Validator."""

import tempfile
from pathlib import Path
from tools.data_collector.data_collector import SyntheticDriveGenerator, DataLogValidator


def test_synthetic_drive_generation_and_validation():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "test_drive.csv"

        gen = SyntheticDriveGenerator(duration_sec=5.0, imu_rate_hz=50.0)
        count = gen.generate(csv_path, outage_windows=[(2.0, 4.0)])

        assert count > 200
        assert csv_path.is_file()

        # Validate generated file
        valid, issues = DataLogValidator.validate_csv(csv_path)
        assert valid is True
        assert len(issues) == 0


def test_data_validator_detects_corrupt_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "corrupt_drive.csv"
        # Write file with non-monotonic timestamps
        csv_path.write_text(
            "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,lat,lon,alt,speed,bearing,accuracy\n"
            "1.0,0.0,0.0,9.81,0.0,0.0,0.0,28.6,77.2,200.0,0.0,0.0,5.0\n"
            "0.5,0.0,0.0,9.81,0.0,0.0,0.0,28.6,77.2,200.0,0.0,0.0,5.0\n"
        )

        valid, issues = DataLogValidator.validate_csv(csv_path)
        assert valid is False
        assert any("Timestamp anomaly" in s for s in issues)
