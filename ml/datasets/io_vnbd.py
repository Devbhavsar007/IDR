"""
IO-VNBD Dataset Loader.

Loads and parses the IO-VNBD (Inertial Odometry - Vehicle Navigation Benchmark Dataset).
Reference: Onyekpe et al. — https://github.com/onyekpeu/IO-VNBD

The dataset contains ~58 hours (4,400 km) of smartphone sensor data at 10 Hz:
  - Accelerometer (m/s²)
  - Gyroscope (rad/s)
  - Magnetometer (µT)
  - GPS (lat, lon, alt, speed, bearing, accuracy)
  - Gravity vector
  - Game rotation vector (quaternion)

This loader:
  1. Parses CSV files from the IO-VNBD directory structure
  2. Validates data ranges and flags anomalies
  3. Computes derived quantities (ENU position, ground-truth velocity)
  4. Generates training windows with labels
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── Expected column mapping ───
# IO-VNBD CSVs have these columns (names may vary slightly):
COLUMN_MAP = {
    "timestamp": "timestamp_s",
    "acc_x": "accel_x",
    "acc_y": "accel_y",
    "acc_z": "accel_z",
    "gyro_x": "gyro_x",
    "gyro_y": "gyro_y",
    "gyro_z": "gyro_z",
    "mag_x": "mag_x",
    "mag_y": "mag_y",
    "mag_z": "mag_z",
    "grav_x": "gravity_x",
    "grav_y": "gravity_y",
    "grav_z": "gravity_z",
    "lat": "latitude",
    "lon": "longitude",
    "alt": "altitude",
    "speed": "speed_mps",
    "bearing": "bearing_deg",
    "accuracy": "horizontal_accuracy_m",
}


@dataclass
class DriveMetadata:
    """Metadata for a single drive recording."""

    drive_id: str
    path: Path
    duration_sec: float = 0.0
    distance_km: float = 0.0
    num_samples: int = 0
    sample_rate_hz: float = 0.0
    gps_coverage_pct: float = 0.0
    vehicle: str = "unknown"
    route: str = "unknown"


@dataclass
class DriveData:
    """Parsed and validated data from a single drive."""

    metadata: DriveMetadata
    df: pd.DataFrame  # Full synchronized DataFrame

    # Derived arrays (numpy, for fast access during training)
    timestamps: np.ndarray = field(default_factory=lambda: np.array([]))
    accel: np.ndarray = field(default_factory=lambda: np.array([]))  # (N, 3)
    gyro: np.ndarray = field(default_factory=lambda: np.array([]))   # (N, 3)
    gravity: np.ndarray = field(default_factory=lambda: np.array([]))  # (N, 3)

    # Ground truth (from GPS)
    gt_position_enu: np.ndarray = field(default_factory=lambda: np.array([]))  # (N, 3)
    gt_velocity_enu: np.ndarray = field(default_factory=lambda: np.array([]))  # (N, 3)
    gt_speed: np.ndarray = field(default_factory=lambda: np.array([]))         # (N,)
    gt_heading_rad: np.ndarray = field(default_factory=lambda: np.array([]))   # (N,)

    # GPS validity mask (True where GPS data is trustworthy)
    gps_valid: np.ndarray = field(default_factory=lambda: np.array([]))  # (N,) bool


class IoVnbdLoader:
    """Loader for the IO-VNBD dataset.

    Usage:
        loader = IoVnbdLoader("/path/to/IO-VNBD")
        drives = loader.discover_drives()
        data = loader.load_drive(drives[0])
    """

    def __init__(self, dataset_root: str | Path):
        self.root = Path(dataset_root)
        if not self.root.exists():
            raise FileNotFoundError(f"IO-VNBD root not found: {self.root}")

    def discover_drives(self) -> list[DriveMetadata]:
        """Discover all drive recordings in the dataset."""
        drives = []
        # IO-VNBD structure: root/vehicle_type/route/drive_xxx.csv
        # or root/data/drive_xxx.csv — we handle both patterns
        csv_files = sorted(self.root.rglob("*.csv"))

        for csv_path in csv_files:
            drive_id = csv_path.stem
            meta = DriveMetadata(
                drive_id=drive_id,
                path=csv_path,
                vehicle=csv_path.parent.parent.name
                if csv_path.parent.name != "data"
                else "unknown",
                route=csv_path.parent.name,
            )
            drives.append(meta)

        logger.info(f"Discovered {len(drives)} drives in {self.root}")
        return drives

    def load_drive(self, meta: DriveMetadata) -> DriveData:
        """Load and validate a single drive recording."""
        logger.info(f"Loading drive: {meta.drive_id} from {meta.path}")

        # ── Read CSV ──
        df = pd.read_csv(meta.path)
        logger.info(f"  Raw shape: {df.shape}, columns: {list(df.columns)}")

        # ── Normalize column names ──
        df = self._normalize_columns(df)

        # ── Basic validation ──
        self._validate_data(df, meta)

        # ── Sort by timestamp ──
        if "timestamp_s" in df.columns:
            df = df.sort_values("timestamp_s").reset_index(drop=True)

        # ── Compute derived quantities ──
        data = self._compute_derived(df, meta)

        return data

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to our standard naming."""
        # Try direct match first
        rename_map = {}
        cols_lower = {c.lower().strip(): c for c in df.columns}

        for src, dst in COLUMN_MAP.items():
            if src in cols_lower:
                rename_map[cols_lower[src]] = dst
            elif src.replace("_", "") in cols_lower:
                rename_map[cols_lower[src.replace("_", "")]] = dst

        if rename_map:
            df = df.rename(columns=rename_map)

        return df

    def _validate_data(self, df: pd.DataFrame, meta: DriveMetadata) -> None:
        """Validate data ranges and compute metadata."""
        n = len(df)
        meta.num_samples = n

        if "timestamp_s" in df.columns:
            ts = df["timestamp_s"].values
            meta.duration_sec = float(ts[-1] - ts[0]) if n > 1 else 0.0
            meta.sample_rate_hz = (n - 1) / meta.duration_sec if meta.duration_sec > 0 else 0.0

        # Accelerometer range check (phone-grade: expect < 20 m/s²)
        for axis in ["accel_x", "accel_y", "accel_z"]:
            if axis in df.columns:
                vals = df[axis].dropna()
                if vals.abs().max() > 200:
                    logger.warning(f"  {axis} has extreme values (max={vals.abs().max():.1f})")

        # Gyroscope range check (< 35 rad/s ≈ 2000 °/s)
        for axis in ["gyro_x", "gyro_y", "gyro_z"]:
            if axis in df.columns:
                vals = df[axis].dropna()
                if vals.abs().max() > 35:
                    logger.warning(f"  {axis} has extreme values (max={vals.abs().max():.1f})")

        # GPS coverage
        if "latitude" in df.columns:
            valid_gps = df["latitude"].notna() & (df["latitude"].abs() > 0.1)
            meta.gps_coverage_pct = float(valid_gps.sum()) / n * 100.0

        logger.info(
            f"  Validated: {n} samples, {meta.duration_sec:.1f}s, "
            f"{meta.sample_rate_hz:.1f}Hz, GPS coverage: {meta.gps_coverage_pct:.1f}%"
        )

    def _compute_derived(self, df: pd.DataFrame, meta: DriveMetadata) -> DriveData:
        """Compute ENU positions, velocities, and training-ready arrays."""
        n = len(df)

        # Extract numpy arrays
        timestamps = df["timestamp_s"].values if "timestamp_s" in df.columns else np.arange(n)

        accel = np.stack(
            [
                df.get("accel_x", pd.Series(np.zeros(n))).fillna(0).values,
                df.get("accel_y", pd.Series(np.zeros(n))).fillna(0).values,
                df.get("accel_z", pd.Series(np.zeros(n))).fillna(0).values,
            ],
            axis=1,
        )

        gyro = np.stack(
            [
                df.get("gyro_x", pd.Series(np.zeros(n))).fillna(0).values,
                df.get("gyro_y", pd.Series(np.zeros(n))).fillna(0).values,
                df.get("gyro_z", pd.Series(np.zeros(n))).fillna(0).values,
            ],
            axis=1,
        )

        gravity = np.stack(
            [
                df.get("gravity_x", pd.Series(np.zeros(n))).fillna(0).values,
                df.get("gravity_y", pd.Series(np.zeros(n))).fillna(0).values,
                df.get("gravity_z", pd.Series(np.zeros(n))).fillna(0).values,
            ],
            axis=1,
        )

        # GPS validity
        gps_valid = np.zeros(n, dtype=bool)
        if "latitude" in df.columns and "longitude" in df.columns:
            lat = df["latitude"].values
            lon = df["longitude"].values
            gps_valid = np.isfinite(lat) & np.isfinite(lon) & (np.abs(lat) > 0.1)

        # Ground truth ENU from GPS
        gt_position_enu = np.zeros((n, 3))
        gt_velocity_enu = np.zeros((n, 3))
        gt_speed = np.zeros(n)
        gt_heading_rad = np.zeros(n)

        if gps_valid.any():
            # Use first valid GPS as origin
            first_valid = np.argmax(gps_valid)
            lat = df["latitude"].values
            lon = df["longitude"].values
            alt = df.get("altitude", pd.Series(np.zeros(n))).fillna(0).values

            ref_lat = lat[first_valid]
            ref_lon = lon[first_valid]
            ref_alt = alt[first_valid]

            # Convert to ENU (vectorized approximation)
            dlat = np.radians(lat - ref_lat)
            dlon = np.radians(lon - ref_lon)
            cos_ref = np.cos(np.radians(ref_lat))

            R_EARTH = 6378137.0
            gt_position_enu[:, 0] = dlon * R_EARTH * cos_ref  # East
            gt_position_enu[:, 1] = dlat * R_EARTH             # North
            gt_position_enu[:, 2] = alt - ref_alt               # Up

            # Zero out invalid positions
            gt_position_enu[~gps_valid] = 0.0

            # Velocity from position differentiation (for valid GPS segments)
            dt = np.diff(timestamps, prepend=timestamps[0])
            dt = np.where(dt > 0, dt, 0.1)  # avoid division by zero
            gt_velocity_enu[1:, :] = np.diff(gt_position_enu, axis=0) / dt[1:, None]
            gt_velocity_enu[~gps_valid] = 0.0

            # Speed from GPS if available
            if "speed_mps" in df.columns:
                gt_speed = df["speed_mps"].fillna(0).values.astype(float)
            else:
                gt_speed = np.linalg.norm(gt_velocity_enu, axis=1)

            # Heading from GPS bearing
            if "bearing_deg" in df.columns:
                bearing = df["bearing_deg"].fillna(0).values.astype(float)
                gt_heading_rad = np.radians(bearing)
            else:
                gt_heading_rad = np.arctan2(
                    gt_velocity_enu[:, 0], gt_velocity_enu[:, 1]
                )  # atan2(E, N)

            # Compute distance
            if gps_valid.sum() > 1:
                dists = np.sqrt(np.sum(np.diff(gt_position_enu[gps_valid], axis=0) ** 2, axis=1))
                meta.distance_km = float(np.sum(dists)) / 1000.0

        return DriveData(
            metadata=meta,
            df=df,
            timestamps=timestamps,
            accel=accel,
            gyro=gyro,
            gravity=gravity,
            gt_position_enu=gt_position_enu,
            gt_velocity_enu=gt_velocity_enu,
            gt_speed=gt_speed,
            gt_heading_rad=gt_heading_rad,
            gps_valid=gps_valid,
        )


def generate_windows(
    drive: DriveData,
    window_size: int = 200,
    stride: int = 100,
    require_gps_label: bool = True,
) -> list[dict]:
    """Generate sliding windows from a drive for training.

    Each window contains:
      - imu: (window_size, 6) — accel_xyz + gyro_xyz
      - gravity: (window_size, 3)
      - label_velocity: (3,) — mean ground-truth velocity ENU over window
      - label_displacement: (3,) — total displacement ENU over window
      - label_heading_change: float — heading change over window (rad)
      - label_speed: float — mean speed over window
      - gps_coverage: float — fraction of window with valid GPS
    """
    n = len(drive.timestamps)
    windows = []

    for start in range(0, n - window_size + 1, stride):
        end = start + window_size

        # GPS coverage in this window
        gps_mask = drive.gps_valid[start:end]
        gps_coverage = float(gps_mask.sum()) / window_size

        if require_gps_label and gps_coverage < 0.5:
            continue  # Skip windows without enough GPS ground truth

        # IMU features
        imu = np.concatenate(
            [drive.accel[start:end], drive.gyro[start:end]], axis=1
        )  # (W, 6)

        gravity = drive.gravity[start:end]  # (W, 3)

        # Labels
        label_velocity = np.mean(drive.gt_velocity_enu[start:end], axis=0)
        label_displacement = drive.gt_position_enu[end - 1] - drive.gt_position_enu[start]
        label_speed = float(np.mean(drive.gt_speed[start:end]))

        # Heading change
        h_start = drive.gt_heading_rad[start]
        h_end = drive.gt_heading_rad[end - 1]
        label_heading_change = float(np.arctan2(np.sin(h_end - h_start), np.cos(h_end - h_start)))

        windows.append(
            {
                "imu": imu.astype(np.float32),
                "gravity": gravity.astype(np.float32),
                "label_velocity": label_velocity.astype(np.float32),
                "label_displacement": label_displacement.astype(np.float32),
                "label_heading_change": np.float32(label_heading_change),
                "label_speed": np.float32(label_speed),
                "gps_coverage": np.float32(gps_coverage),
                "drive_id": drive.metadata.drive_id,
                "window_start": start,
            }
        )

    logger.info(
        f"Generated {len(windows)} windows from {drive.metadata.drive_id} "
        f"(window={window_size}, stride={stride})"
    )
    return windows
