// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// NavigationState — the final output of the IDR engine.
// This is the contract between the C++ core and consumers
// (Android UI via JNI, edge output adapters, replay tools).

#pragma once

#include "idr/types/common.h"

namespace idr {

/// Final navigation state output from the IDR engine.
/// All fields are always populated; uncertainty/confidence fields
/// allow consumers to judge trustworthiness.
struct NavigationState {
    // ── Core state ──

    /// WGS84 position
    double latitude_deg  = 0.0;
    double longitude_deg = 0.0;
    double altitude_m    = 0.0;

    /// Velocity in NED frame (m/s)
    double velocity_north_mps = 0.0;
    double velocity_east_mps  = 0.0;
    double velocity_down_mps  = 0.0;

    /// Scalar speed (m/s, ≥ 0)
    double speed_mps = 0.0;

    /// Heading (degrees, clockwise from North, [0, 360))
    double heading_deg = 0.0;

    /// Attitude (degrees)
    double pitch_deg = 0.0;
    double roll_deg  = 0.0;

    // ── Accuracy / uncertainty (1σ) ──

    double horizontal_accuracy_m  = std::numeric_limits<double>::max();
    double vertical_accuracy_m    = std::numeric_limits<double>::max();
    double velocity_accuracy_mps  = std::numeric_limits<double>::max();
    double heading_accuracy_deg   = std::numeric_limits<double>::max();

    // ── Mode and confidence ──

    NavigationMode mode       = NavigationMode::INIT;
    IntegrityStatus integrity = IntegrityStatus::NOMINAL;

    /// Source-specific confidence scores [0.0, 1.0]
    double gnss_confidence      = 0.0;
    double ai_confidence        = 0.0;
    double map_confidence       = 0.0;
    double alignment_confidence = 0.0;

    // ── Map matching ──

    int64_t matched_road_id          = -1;   ///< -1 = no match
    double  matched_road_confidence  = 0.0;
    int     road_level               = 0;    ///< 0 = ground, 1 = flyover, -1 = underpass

    // ── Vehicle state ──

    bool    is_stationary            = false; ///< ZUPT-detected stationarity
    uint8_t alignment_quality        = 0;     ///< 0=NONE, 1=GRAVITY, 2=COARSE, 3=CONVERGED

    // ── Metadata ──

    Timestamp timestamp_ns = kInvalidTimestamp;
    uint32_t update_count  = 0;

    /// Processing latencies (milliseconds)
    double ins_latency_ms       = 0.0;
    double ai_latency_ms        = 0.0;
    double ekf_latency_ms       = 0.0;
    double map_match_latency_ms = 0.0;
    double total_latency_ms     = 0.0;

    // ── Convenience ──

    bool isInitialized() const {
        return mode != NavigationMode::INIT;
    }

    GeoCoord toGeoCoord() const {
        return {latitude_deg, longitude_deg, altitude_m};
    }
};

}  // namespace idr
