// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Coordinate frame definitions and documentation.
//
// FRAME CONVENTIONS (must be respected throughout the entire codebase):
//
//   Phone (P)       — Android sensor frame
//                     x = right edge of screen
//                     y = top edge of screen
//                     z = out of screen face
//                     Origin: device center
//
//   Vehicle (V)     — SAE-style vehicle body frame
//                     x = forward (direction of travel)
//                     y = left
//                     z = up
//                     Origin: approximate vehicle center of mass
//
//   Navigation (N)  — Local tangent plane, ENU
//                     x = East
//                     y = North
//                     z = Up
//                     Origin: first valid GNSS fix or configured origin
//
//   ECEF            — Earth-Centered Earth-Fixed (WGS84)
//                     Standard definition
//
//   LLA             — Latitude / Longitude / Altitude (WGS84)
//                     lat in degrees [-90, 90]
//                     lon in degrees [-180, 180]
//                     alt in meters above ellipsoid
//
// ROTATION CONVENTIONS:
//
//   All rotations stored as unit quaternions (Hamilton, scalar-first: [w, x, y, z]).
//   Euler angles used ONLY for human-readable output, NEVER for integration.
//
//   R_AB means: rotation FROM frame B TO frame A.
//   v_A = R_AB * v_B
//
//   Example: R_VP transforms a vector from Phone frame to Vehicle frame.
//            a_vehicle = R_VP * a_phone
//
// UNITS:
//
//   Acceleration:   m/s²
//   Angular rate:   rad/s
//   Position:       m (ENU) or degrees (LLA)
//   Velocity:       m/s
//   Heading:        radians (internal); degrees (output, clockwise from North)
//   Pressure:       hPa
//   Magnetic field: µT
//   Timestamps:     nanoseconds (int64_t, monotonic)
//   Temperature:    °C (when relevant)

#pragma once

#include <cstdint>

namespace idr {

/// Coordinate frame identifier.
/// Used for documentation and runtime assertions, not for implicit conversion.
enum class Frame : uint8_t {
    PHONE      = 0,  ///< Android sensor frame (P)
    VEHICLE    = 1,  ///< Vehicle body frame (V)
    NAVIGATION = 2,  ///< Local ENU tangent frame (N)
    ECEF       = 3,  ///< Earth-Centered Earth-Fixed
    LLA        = 4,  ///< Geodetic latitude/longitude/altitude
};

inline const char* toString(Frame f) {
    switch (f) {
        case Frame::PHONE:      return "PHONE";
        case Frame::VEHICLE:    return "VEHICLE";
        case Frame::NAVIGATION: return "NAVIGATION";
        case Frame::ECEF:       return "ECEF";
        case Frame::LLA:        return "LLA";
    }
    return "UNKNOWN";
}

}  // namespace idr
