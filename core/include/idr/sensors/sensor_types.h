// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Sensor data structures — platform-agnostic.
// These structures are the contract between sensor adapters (Android, serial,
// UDP, file replay) and the IDR core engine.

#pragma once

#include "idr/types/common.h"
#include <cmath>

namespace idr {

// ────────────────────────────────────────────────────────
// IMU sample
// ────────────────────────────────────────────────────────

/// A single IMU measurement from accelerometer and gyroscope.
/// All values are in the PHONE frame unless explicitly noted.
///
/// Fields marked "optional" should be set to NaN if unavailable.
struct ImuSample {
    /// Monotonic timestamp (nanoseconds)
    Timestamp timestamp_ns = kInvalidTimestamp;

    /// Accelerometer reading (m/s²), phone frame.
    /// Includes gravity component (as reported by Android TYPE_ACCELEROMETER).
    Vec3d accel = Vec3d::Zero();

    /// Gyroscope reading (rad/s), phone frame.
    Vec3d gyro = Vec3d::Zero();

    // ── Optional sensors (NaN if unavailable) ──

    /// Gravity vector (m/s²), phone frame.
    /// From Android TYPE_GRAVITY (sensor-fusion-derived).
    /// If available, linear_accel = accel - gravity.
    Vec3d gravity = Vec3d::Constant(std::numeric_limits<double>::quiet_NaN());

    /// Magnetometer reading (µT), phone frame.
    Vec3d mag = Vec3d::Constant(std::numeric_limits<double>::quiet_NaN());

    /// Barometric pressure (hPa). NaN if unavailable.
    double pressure_hpa = std::numeric_limits<double>::quiet_NaN();

    /// Sensor quality indicators
    SensorHealth accel_health = SensorHealth::VALID;
    SensorHealth gyro_health  = SensorHealth::VALID;

    bool hasGravity() const { return std::isfinite(gravity.x()); }
    bool hasMag() const { return std::isfinite(mag.x()); }
    bool hasPressure() const { return std::isfinite(pressure_hpa); }
    bool isValid() const { return timestamp_ns != kInvalidTimestamp; }
};

// ────────────────────────────────────────────────────────
// GNSS measurement
// ────────────────────────────────────────────────────────

/// A single GNSS position/velocity fix.
struct GnssMeasurement {
    /// Monotonic timestamp (nanoseconds)
    Timestamp timestamp_ns = kInvalidTimestamp;

    /// WGS84 position
    double latitude_deg  = 0.0;
    double longitude_deg = 0.0;
    double altitude_m    = 0.0;

    /// Reported accuracy (1σ, meters)
    float horizontal_accuracy_m = std::numeric_limits<float>::max();
    float vertical_accuracy_m   = std::numeric_limits<float>::max();

    /// Speed over ground (m/s), always ≥ 0
    float speed_mps          = 0.0f;
    float speed_accuracy_mps = std::numeric_limits<float>::max();

    /// Bearing (degrees clockwise from North). NaN if unavailable.
    float bearing_deg          = std::numeric_limits<float>::quiet_NaN();
    float bearing_accuracy_deg = std::numeric_limits<float>::quiet_NaN();

    /// Satellite information
    int satellite_count = 0;
    GnssFixType fix_type = GnssFixType::NO_FIX;

    bool hasBearing() const { return std::isfinite(bearing_deg); }
    bool hasFix() const { return fix_type != GnssFixType::NO_FIX; }
    bool isValid() const {
        return timestamp_ns != kInvalidTimestamp &&
               hasFix() &&
               std::abs(latitude_deg) <= 90.0 &&
               std::abs(longitude_deg) <= 180.0;
    }

    /// Convert to GeoCoord
    GeoCoord toGeoCoord() const {
        return {latitude_deg, longitude_deg, altitude_m};
    }
};

// ────────────────────────────────────────────────────────
// Magnetometer sample (standalone)
// ────────────────────────────────────────────────────────

/// Standalone magnetometer sample, for when mag arrives separately from IMU.
struct MagSample {
    Timestamp timestamp_ns = kInvalidTimestamp;
    Vec3d field_ut = Vec3d::Zero();  ///< Magnetic field (µT), phone frame
    SensorHealth health = SensorHealth::VALID;

    bool isValid() const { return timestamp_ns != kInvalidTimestamp; }
};

// ────────────────────────────────────────────────────────
// Barometer sample (standalone)
// ────────────────────────────────────────────────────────

/// Standalone barometric pressure sample.
struct BaroSample {
    Timestamp timestamp_ns = kInvalidTimestamp;
    double pressure_hpa = std::numeric_limits<double>::quiet_NaN();
    SensorHealth health = SensorHealth::VALID;

    bool isValid() const {
        return timestamp_ns != kInvalidTimestamp &&
               std::isfinite(pressure_hpa) &&
               pressure_hpa > 300.0 && pressure_hpa < 1100.0;  // sane range
    }
};

// ────────────────────────────────────────────────────────
// Wheel Speed / Vehicle Odometry sample
// ────────────────────────────────────────────────────────

/// Wheel speed or OBD-II vehicle speed measurement from vehicle CAN bus.
struct WheelSpeedSample {
    Timestamp timestamp_ns = kInvalidTimestamp;
    double speed_mps = 0.0;               ///< Forward ground speed in m/s
    double accuracy_mps = 0.2;            ///< Measurement 1σ uncertainty in m/s
    SensorHealth health = SensorHealth::VALID;

    bool isValid() const {
        return timestamp_ns != kInvalidTimestamp &&
               std::isfinite(speed_mps) &&
               speed_mps >= 0.0 && speed_mps < 120.0;  // realistic vehicle speed (0 to ~430 km/h)
    }
};

}  // namespace idr

