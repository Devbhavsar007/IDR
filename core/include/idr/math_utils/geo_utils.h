// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Coordinate conversion utilities.
// WGS84 ↔ ECEF ↔ ENU conversions for the entire project.

#pragma once

#include "idr/types/common.h"
#include <cmath>

namespace idr {
namespace geo {

/// Convert WGS84 geodetic (lat, lon, alt) to ECEF (x, y, z) in meters.
inline Vec3d llaToEcef(double lat_deg, double lon_deg, double alt_m) {
    using namespace constants;
    const double lat = lat_deg * kDegToRad;
    const double lon = lon_deg * kDegToRad;

    const double e2 = 2.0 * kWgs84F - kWgs84F * kWgs84F;
    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);
    const double sin_lon = std::sin(lon);
    const double cos_lon = std::cos(lon);

    const double N = kWgs84A / std::sqrt(1.0 - e2 * sin_lat * sin_lat);

    return {
        (N + alt_m) * cos_lat * cos_lon,
        (N + alt_m) * cos_lat * sin_lon,
        (N * (1.0 - e2) + alt_m) * sin_lat
    };
}

/// Convert ECEF (x, y, z) to WGS84 geodetic (lat_deg, lon_deg, alt_m).
/// Uses Bowring's iterative method.
inline GeoCoord ecefToLla(const Vec3d& ecef) {
    using namespace constants;
    const double e2 = 2.0 * kWgs84F - kWgs84F * kWgs84F;
    const double x = ecef.x(), y = ecef.y(), z = ecef.z();
    const double p = std::sqrt(x * x + y * y);
    const double lon = std::atan2(y, x);

    // Iterative latitude computation (Bowring's method, ~3 iterations)
    double lat = std::atan2(z, p * (1.0 - e2));
    for (int i = 0; i < 5; ++i) {
        double sin_lat = std::sin(lat);
        double N = kWgs84A / std::sqrt(1.0 - e2 * sin_lat * sin_lat);
        lat = std::atan2(z + e2 * N * sin_lat, p);
    }

    double sin_lat = std::sin(lat);
    double N = kWgs84A / std::sqrt(1.0 - e2 * sin_lat * sin_lat);
    double alt = p / std::cos(lat) - N;

    return {lat * kRadToDeg, lon * kRadToDeg, alt};
}

/// Compute the rotation matrix from ECEF to local ENU frame at a given
/// geodetic reference point.
inline Mat3d ecefToEnuRotation(double ref_lat_deg, double ref_lon_deg) {
    using namespace constants;
    const double lat = ref_lat_deg * kDegToRad;
    const double lon = ref_lon_deg * kDegToRad;

    const double sin_lat = std::sin(lat);
    const double cos_lat = std::cos(lat);
    const double sin_lon = std::sin(lon);
    const double cos_lon = std::cos(lon);

    Mat3d R;
    // Row 0: East
    R(0, 0) = -sin_lon;
    R(0, 1) =  cos_lon;
    R(0, 2) =  0.0;
    // Row 1: North
    R(1, 0) = -sin_lat * cos_lon;
    R(1, 1) = -sin_lat * sin_lon;
    R(1, 2) =  cos_lat;
    // Row 2: Up
    R(2, 0) =  cos_lat * cos_lon;
    R(2, 1) =  cos_lat * sin_lon;
    R(2, 2) =  sin_lat;

    return R;
}

/// Convert WGS84 geodetic to local ENU (meters) relative to a reference point.
inline Vec3d llaToEnu(double lat_deg, double lon_deg, double alt_m,
                      double ref_lat_deg, double ref_lon_deg, double ref_alt_m) {
    Vec3d ecef_point = llaToEcef(lat_deg, lon_deg, alt_m);
    Vec3d ecef_ref   = llaToEcef(ref_lat_deg, ref_lon_deg, ref_alt_m);
    Mat3d R_enu      = ecefToEnuRotation(ref_lat_deg, ref_lon_deg);

    return R_enu * (ecef_point - ecef_ref);
}

/// Convert local ENU (meters) back to WGS84 geodetic.
inline GeoCoord enuToLla(const Vec3d& enu,
                         double ref_lat_deg, double ref_lon_deg, double ref_alt_m) {
    Vec3d ecef_ref = llaToEcef(ref_lat_deg, ref_lon_deg, ref_alt_m);
    Mat3d R_enu = ecefToEnuRotation(ref_lat_deg, ref_lon_deg);

    Vec3d ecef_point = ecef_ref + R_enu.transpose() * enu;
    return ecefToLla(ecef_point);
}

/// Approximate distance in meters between two geodetic points (Haversine).
/// Only suitable for distances < 500 km. For longer distances, use ECEF.
inline double haversineDistance(double lat1_deg, double lon1_deg,
                                double lat2_deg, double lon2_deg) {
    using namespace constants;
    const double lat1 = lat1_deg * kDegToRad;
    const double lat2 = lat2_deg * kDegToRad;
    const double dlat = (lat2_deg - lat1_deg) * kDegToRad;
    const double dlon = (lon2_deg - lon1_deg) * kDegToRad;

    const double a = std::sin(dlat / 2) * std::sin(dlat / 2) +
                     std::cos(lat1) * std::cos(lat2) *
                     std::sin(dlon / 2) * std::sin(dlon / 2);
    const double c = 2.0 * std::atan2(std::sqrt(a), std::sqrt(1.0 - a));

    return kWgs84A * c;
}

}  // namespace geo
}  // namespace idr
