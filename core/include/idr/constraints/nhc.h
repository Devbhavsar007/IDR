// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Vehicle profile and adaptive Non-Holonomic Constraint (NHC) system.
// Implements the "learned slack" approach from Blueprint v1.1 §3.

#pragma once

#include "idr/types/common.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace idr {

// ────────────────────────────────────────────────────────
// Motion classification
// ────────────────────────────────────────────────────────

enum class MotionClass : uint8_t {
    STATIONARY     = 0,
    NORMAL_DRIVING = 1,
    ACCELERATION   = 2,
    BRAKING        = 3,
    SHARP_TURN     = 4,
    ROUNDABOUT     = 5,
    POTHOLE        = 6,
    SPEED_BREAKER  = 7,
    VIBRATION      = 8,
    REVERSE        = 9,
    U_TURN         = 10,
    PARKING        = 11,
    PHONE_MOVEMENT = 12,
    UNUSUAL_MOTION = 13,
    NUM_CLASSES    = 14
};

/// Full softmax distribution over motion classes.
/// Used for soft NHC weighting instead of hard argmax.
struct MotionClassDistribution {
    std::array<float, 14> probabilities = {};  // sum to 1.0

    MotionClass argmax() const {
        auto it = std::max_element(probabilities.begin(), probabilities.end());
        return static_cast<MotionClass>(std::distance(probabilities.begin(), it));
    }

    float confidence() const {
        return *std::max_element(probabilities.begin(), probabilities.end());
    }
};

// ────────────────────────────────────────────────────────
// Vehicle profile
// ────────────────────────────────────────────────────────

struct VehicleProfile {
    VehicleType type = VehicleType::UNKNOWN;
    double wheelbase_m = 2.5;          ///< Approximate wheelbase
    double max_lateral_accel = 7.0;    ///< m/s² before expected slip
    double max_speed_mps = 50.0;       ///< Sanity bound
    double nhc_lateral_sigma = 0.2;    ///< Base lateral NHC noise (m/s)
    double nhc_vertical_sigma = 0.15;  ///< Base vertical NHC noise (m/s)
    bool has_lean = false;             ///< Two-wheelers lean in turns
};

namespace profiles {
    inline constexpr VehicleProfile kCar          = {VehicleType::CAR,         2.7, 8.0, 55.0, 0.10, 0.10, false};
    inline constexpr VehicleProfile kTruckBus     = {VehicleType::TRUCK_BUS,   6.0, 5.0, 35.0, 0.15, 0.15, false};
    inline constexpr VehicleProfile kTruck        = kTruckBus;
    inline constexpr VehicleProfile kBus          = kTruckBus;
    inline constexpr VehicleProfile kTwoWheeler   = {VehicleType::TWO_WHEELER, 1.4, 6.0, 45.0, 0.30, 0.20, true};
    inline constexpr VehicleProfile kBike         = kTwoWheeler;
    inline constexpr VehicleProfile kAutoRickshaw = {VehicleType::CAR,         2.0, 5.5, 25.0, 0.25, 0.18, false};
    inline constexpr VehicleProfile kUnknown      = {VehicleType::UNKNOWN,     2.5, 7.0, 50.0, 0.20, 0.15, false};
}

// ────────────────────────────────────────────────────────
// Alignment state (for NHC modulation)
// ────────────────────────────────────────────────────────

struct AlignmentState {
    double confidence = 0.0;  ///< [0, 1] alignment quality
};

// ────────────────────────────────────────────────────────
// Adaptive NHC output
// ────────────────────────────────────────────────────────

struct AdaptiveNhcParams {
    double lateral_sigma;     ///< Adapted lateral velocity sigma (m/s)
    double vertical_sigma;    ///< Adapted vertical velocity sigma (m/s)
    double nhc_confidence;    ///< [0, 1] how trustworthy NHC is right now
};

// ────────────────────────────────────────────────────────
// Adaptive NHC computation (learned slack, Blueprint v1.1 §3)
// ────────────────────────────────────────────────────────

/// Compute adaptive NHC parameters using soft motion-class weighting
/// and physics-based modulation factors.
inline AdaptiveNhcParams computeAdaptiveNhc(
    const VehicleProfile& vp,
    const MotionClassDistribution& mc_dist,
    double yaw_rate_rad_s,
    double speed_mps,
    double lateral_accel_mps2,
    const AlignmentState& align) {

    double base_lat = vp.nhc_lateral_sigma;
    double base_vert = vp.nhc_vertical_sigma;

    // ── Soft motion-class weighting ──
    // Each class contributes a sigma multiplier weighted by its probability
    static constexpr double CLASS_MULTIPLIERS[14] = {
        // STAT  NORM  ACCEL BRAKE SHARP ROUND POTHL SPDBRK VIB   REV   UTRN  PARK  PHONE UNUSUAL
        0.1,    1.0,  1.0,  1.2,  5.0,  7.0,  1.5,  1.5,   2.0,  5.0,  10.0, 10.0, 20.0, 20.0
    };
    double weighted_multiplier = 0.0;
    for (size_t i = 0; i < 14; i++) {
        weighted_multiplier += static_cast<double>(mc_dist.probabilities[i]) * CLASS_MULTIPLIERS[i];
    }

    // ── Physics-based adaptation ──
    double slip_factor = 1.0 + 2.0 * std::abs(lateral_accel_mps2) / vp.max_lateral_accel;
    double turn_factor = 1.0 + 3.0 * std::abs(yaw_rate_rad_s);
    double lean_factor = vp.has_lean ? (1.0 + 5.0 * std::abs(yaw_rate_rad_s)) : 1.0;
    double speed_factor = (speed_mps < 2.0) ? 5.0 :
                          (speed_mps < 5.0) ? 2.0 : 1.0;

    // ── Alignment confidence modulation ──
    double align_factor = 1.0 / std::max(align.confidence, 0.1);

    // ── Combine ──
    double final_lat = base_lat * weighted_multiplier * slip_factor
                     * turn_factor * lean_factor * speed_factor
                     * std::min(align_factor, 10.0);
    double final_vert = base_vert * weighted_multiplier * speed_factor;

    // ── NHC confidence ──
    double nhc_conf = static_cast<double>(
        mc_dist.probabilities[0] + mc_dist.probabilities[1] + mc_dist.probabilities[2]);
    nhc_conf *= align.confidence;
    nhc_conf = std::clamp(nhc_conf, 0.0, 1.0);

    // ── Clamp ──
    final_lat  = std::clamp(final_lat, 0.001, 100.0);
    final_vert = std::clamp(final_vert, 0.001, 50.0);

    return {final_lat, final_vert, nhc_conf};
}

}  // namespace idr
