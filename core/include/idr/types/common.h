// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Core common types: Vec3d, Quaterniond, GeoCoord.
// All mathematical types wrap Eigen for performance and SIMD.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cmath>
#include <cstdint>
#include <limits>

namespace idr {

// ────────────────────────────────────────────────────────
// Fundamental numeric types
// ────────────────────────────────────────────────────────

using Vec3d = Eigen::Vector3d;
using Vec3f = Eigen::Vector3f;
using Vec4d = Eigen::Vector4d;
using Mat3d = Eigen::Matrix3d;
using Mat4d = Eigen::Matrix4d;
using MatXd = Eigen::MatrixXd;
using VecXd = Eigen::VectorXd;
using Quaterniond = Eigen::Quaterniond;

// ────────────────────────────────────────────────────────
// Timestamp type (nanoseconds, monotonic)
// ────────────────────────────────────────────────────────

/// Nanosecond monotonic timestamp. All internal timing uses this.
using Timestamp = int64_t;

/// Sentinel value indicating "no timestamp available"
constexpr Timestamp kInvalidTimestamp = std::numeric_limits<int64_t>::min();

/// Convert nanoseconds to seconds
inline constexpr double nsToSec(Timestamp ns) {
    return static_cast<double>(ns) * 1e-9;
}

/// Convert seconds to nanoseconds
inline constexpr Timestamp secToNs(double sec) {
    return static_cast<Timestamp>(sec * 1e9);
}

// ────────────────────────────────────────────────────────
// Geographic coordinate (WGS84)
// ────────────────────────────────────────────────────────

/// WGS84 geodetic coordinate.
/// latitude/longitude in DEGREES, altitude in METERS.
struct GeoCoord {
    double latitude_deg  = 0.0;
    double longitude_deg = 0.0;
    double altitude_m    = 0.0;

    bool isValid() const {
        return std::abs(latitude_deg) <= 90.0 &&
               std::abs(longitude_deg) <= 180.0 &&
               std::isfinite(altitude_m);
    }
};

// ────────────────────────────────────────────────────────
// Navigation mode
// ────────────────────────────────────────────────────────

enum class NavigationMode : uint8_t {
    INIT           = 0,  ///< System initializing
    ALIGNING       = 1,  ///< Phone-to-vehicle alignment in progress
    GNSS_ONLY      = 2,  ///< GNSS available, no INS yet
    GNSS_INS       = 3,  ///< Full GNSS+INS fusion
    DEGRADED       = 4,  ///< GNSS intermittent/noisy
    DEAD_RECKONING = 5,  ///< GNSS denied, pure DR
    RECOVERY       = 6,  ///< GNSS returning, reintegrating
    LOW_CONFIDENCE = 7,  ///< Uncertainty exceeds usable threshold
};

/// Human-readable name for NavigationMode
inline const char* toString(NavigationMode mode) {
    switch (mode) {
        case NavigationMode::INIT:           return "INIT";
        case NavigationMode::ALIGNING:       return "ALIGNING";
        case NavigationMode::GNSS_ONLY:      return "GNSS_ONLY";
        case NavigationMode::GNSS_INS:       return "GNSS_INS";
        case NavigationMode::DEGRADED:       return "DEGRADED";
        case NavigationMode::DEAD_RECKONING: return "DEAD_RECKONING";
        case NavigationMode::RECOVERY:       return "RECOVERY";
        case NavigationMode::LOW_CONFIDENCE: return "LOW_CONFIDENCE";
    }
    return "UNKNOWN";
}

// ────────────────────────────────────────────────────────
// Integrity status
// ────────────────────────────────────────────────────────

enum class IntegrityStatus : uint8_t {
    NOMINAL  = 0,  ///< All sources consistent, high confidence
    CAUTION  = 1,  ///< Minor inconsistency detected
    WARNING  = 2,  ///< Significant inconsistency; accuracy degraded
    ALERT    = 3,  ///< Critical failure; accuracy unreliable
};

// ────────────────────────────────────────────────────────
// Sensor health
// ────────────────────────────────────────────────────────

enum class SensorHealth : uint8_t {
    VALID    = 0,  ///< Sensor data is trustworthy
    DEGRADED = 1,  ///< Sensor data is noisy or partially invalid
    INVALID  = 2,  ///< Sensor data should not be used
};

// ────────────────────────────────────────────────────────
// Vehicle type
// ────────────────────────────────────────────────────────

enum class VehicleType : uint8_t {
    CAR         = 0,
    TRUCK_BUS   = 1,
    TWO_WHEELER = 2,
    UNKNOWN     = 3,
};

// ────────────────────────────────────────────────────────
// GNSS fix type
// ────────────────────────────────────────────────────────

enum class GnssFixType : uint8_t {
    NO_FIX    = 0,
    FIX_2D    = 1,
    FIX_3D    = 2,
    RTK_FLOAT = 3,
    RTK_FIXED = 4,
};

// ────────────────────────────────────────────────────────
// GNSS integrity state
// ────────────────────────────────────────────────────────

enum class GnssIntegrity : uint8_t {
    HEALTHY  = 0,
    DEGRADED = 1,
    SUSPECT  = 2,
    DENIED   = 3,
};

// ────────────────────────────────────────────────────────
// Constants
// ────────────────────────────────────────────────────────

namespace constants {

/// WGS84 semi-major axis (meters)
constexpr double kWgs84A = 6378137.0;

/// WGS84 flattening
constexpr double kWgs84F = 1.0 / 298.257223563;

/// Standard gravity (m/s²)
constexpr double kGravity = 9.80665;

/// Pi (to full double precision)
constexpr double kPi = 3.14159265358979323846;

/// Degrees to radians
constexpr double kDegToRad = kPi / 180.0;

/// Radians to degrees
constexpr double kRadToDeg = 180.0 / kPi;

}  // namespace constants

}  // namespace idr
