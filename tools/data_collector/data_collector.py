"""
IDR Field Data Collector & Stream Ingestion Tool.

Supports:
  1. Synthetic drive dataset generation with GNSS outage intervals
  2. Stream ingestion & CSV serialization
  3. Field log validation (sanity check for IMU jitter, dropped frames, coordinate validity)
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class DatasetMetadata:
    vehicle_type: str = "car"
    device_model: str = "Field Logger v1"
    route_name: str = "Urban Corridor"
    sample_rate_hz: float = 100.0
    notes: str = ""
    created_at_utc: float = field(default_factory=time.time)


class SyntheticDriveGenerator:
    """Generates synthetic high-rate IMU and GNSS datasets for benchmarking."""

    def __init__(
        self,
        duration_sec: float = 30.0,
        imu_rate_hz: float = 100.0,
        origin_lat: float = 28.613939,
        origin_lon: float = 77.209023,
        origin_alt: float = 216.0,
    ):
        self.duration_sec = duration_sec
        self.dt = 1.0 / imu_rate_hz
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.origin_alt = origin_alt

    def generate(
        self,
        output_csv_path: str | Path,
        outage_windows: Optional[List[Tuple[float, float]]] = None,
    ) -> int:
        """
        Generate synthetic drive CSV.
        Simulates:
          - Stop at start (0-3s): stationary ZUPT
          - Acceleration to 15 m/s (~54 km/h) (3-8s)
          - Constant speed with gentle turn (8-20s)
          - Deceleration to stop (20-25s)
          - Stationary (25-30s)
        """
        outage_windows = outage_windows or []
        path = Path(output_csv_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        t = 0.0
        cur_lat = self.origin_lat
        cur_lon = self.origin_lon
        cur_speed = 0.0
        cur_heading = 0.0  # North

        while t <= self.duration_sec:
            # Kinematics profile
            acc_forward = 0.0
            yaw_rate = 0.0

            if 3.0 <= t < 8.0:
                acc_forward = 3.0  # 3 m/s² accel
                cur_speed = min(15.0, cur_speed + acc_forward * self.dt)
            elif 8.0 <= t < 15.0:
                acc_forward = 0.0
            elif 15.0 <= t < 20.0:
                # Turn right 90 deg over 5s (18 deg/s = ~0.314 rad/s)
                yaw_rate = math.radians(18.0)
                cur_heading += math.degrees(yaw_rate * self.dt)
            elif 20.0 <= t < 24.0:
                acc_forward = -3.75  # brake
                cur_speed = max(0.0, cur_speed + acc_forward * self.dt)
            else:
                cur_speed = 0.0

            # Displace position
            if cur_speed > 0:
                dist = cur_speed * self.dt
                d_lat = (dist * math.cos(math.radians(cur_heading))) / 111111.0
                d_lon = (dist * math.sin(math.radians(cur_heading))) / (
                    111111.0 * math.cos(math.radians(cur_lat))
                )
                cur_lat += d_lat
                cur_lon += d_lon

            # IMU measurements in body frame (x: right, y: forward, z: up)
            acc_x = cur_speed * yaw_rate  # Centripetal accel
            acc_y = acc_forward
            acc_z = 9.81
            gyro_x = 0.0
            gyro_y = 0.0
            gyro_z = -yaw_rate  # CW heading -> negative Z in standard NED/ENU

            # GNSS: 1 Hz
            is_gnss_epoch = (abs(t % 1.0) < self.dt) or (t == 0.0)
            in_outage = any(start <= t <= end for start, end in outage_windows)

            if is_gnss_epoch and not in_outage:
                rows.append([
                    f"{t:.3f}",
                    f"{acc_x:.4f}", f"{acc_y:.4f}", f"{acc_z:.4f}",
                    f"{gyro_x:.4f}", f"{gyro_y:.4f}", f"{gyro_z:.4f}",
                    f"{cur_lat:.7f}", f"{cur_lon:.7f}", f"{self.origin_alt:.1f}",
                    f"{cur_speed:.2f}", f"{cur_heading % 360:.1f}", "2.5"
                ])
            else:
                rows.append([
                    f"{t:.3f}",
                    f"{acc_x:.4f}", f"{acc_y:.4f}", f"{acc_z:.4f}",
                    f"{gyro_x:.4f}", f"{gyro_y:.4f}", f"{gyro_z:.4f}",
                    "", "", "", "", "", ""
                ])

            t += self.dt

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z",
                "lat", "lon", "alt", "speed", "bearing", "accuracy"
            ])
            writer.writerows(rows)

        return len(rows)


class DataLogValidator:
    """Sanity checks field recording CSVs for anomalies, drops, and sensor corruption."""

    @staticmethod
    def validate_csv(filepath: str | Path) -> Tuple[bool, List[str]]:
        path = Path(filepath)
        if not path.is_file():
            return False, [f"File not found: {filepath}"]

        issues = []
        timestamps = []
        imu_count = 0
        gnss_count = 0

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            expected = {"timestamp", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"}
            if not expected.issubset(set(reader.fieldnames or [])):
                return False, [f"Missing required columns in CSV header. Found: {reader.fieldnames}"]

            for i, row in enumerate(reader, start=2):
                try:
                    t = float(row["timestamp"])
                    timestamps.append(t)
                    imu_count += 1

                    ax = float(row["acc_x"])
                    ay = float(row["acc_y"])
                    az = float(row["acc_z"])
                    norm = math.sqrt(ax * ax + ay * ay + az * az)
                    if norm > 80.0:  # > 8g
                        issues.append(f"Line {i}: extreme accel norm {norm:.1f} m/s²")

                    if row.get("lat") and row.get("lon"):
                        lat = float(row["lat"])
                        lon = float(row["lon"])
                        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
                            issues.append(f"Line {i}: invalid coordinates ({lat}, {lon})")
                        gnss_count += 1
                except ValueError as e:
                    issues.append(f"Line {i}: parsing error: {e}")

        # Check timestamp monotonically increasing
        for i in range(1, len(timestamps)):
            dt = timestamps[i] - timestamps[i - 1]
            if dt <= 0.0:
                issues.append(f"Timestamp anomaly at sample {i}: dt={dt:.4f}s")
            elif dt > 0.5:
                issues.append(f"Large sensor gap at sample {i}: dt={dt:.4f}s")

        passed = len(issues) == 0
        return passed, issues


def main():
    parser = argparse.ArgumentParser(description="IDR Data Collection & Validation Utility")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen_parser = subparsers.add_parser("generate", help="Generate synthetic drive dataset")
    gen_parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    gen_parser.add_argument("-d", "--duration", type=float, default=30.0, help="Duration in seconds")
    gen_parser.add_argument("--outage-start", type=float, default=-1.0, help="Simulated outage start")
    gen_parser.add_argument("--outage-end", type=float, default=-1.0, help="Simulated outage end")

    val_parser = subparsers.add_parser("validate", help="Validate a recorded dataset CSV")
    val_parser.add_argument("file", help="CSV file path")

    args = parser.parse_args()

    if args.command == "generate":
        outages = [(args.outage_start, args.outage_end)] if (args.outage_start >= 0 and args.outage_end >= 0) else []
        gen = SyntheticDriveGenerator(duration_sec=args.duration)
        n = gen.generate(args.output, outages)
        print(f"Generated {n} records to {args.output}")
    elif args.command == "validate":
        passed, issues = DataLogValidator.validate_csv(args.file)
        if passed:
            print(f"VALID: {args.file} passed all sanity checks.")
        else:
            print(f"INVALID: {len(issues)} issues found in {args.file}:")
            for issue in issues[:10]:
                print(f"  - {issue}")
            sys.exit(1)


if __name__ == "__main__":
    main()
