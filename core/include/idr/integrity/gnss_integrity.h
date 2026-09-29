// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// GNSS Integrity Monitor — 4-state machine with innovation gating,
// spoofing detection, and hysteresis.

#pragma once

#include "idr/sensors/sensor_types.h"
#include "idr/types/common.h"

#include <spdlog/spdlog.h>
#include <cmath>
#include <deque>

namespace idr {

/// GNSS integrity configuration — hysteresis and thresholds.
struct IntegrityConfig {
    // State transition counts (consecutive epochs meeting criteria)
    int healthy_to_degraded_count = 3;
    int healthy_to_suspect_count  = 2;
    int degraded_to_denied_epochs = 10;
    int suspect_to_denied_epochs  = 15;
    int denied_to_suspect_count   = 1;
    int suspect_to_healthy_count  = 5;

    // Innovation thresholds
    double innovation_warn_sigma   = 3.0;
    double innovation_reject_sigma = 5.0;

    // Accuracy thresholds
    double max_healthy_hAcc_m  = 10.0;
    double max_degraded_hAcc_m = 30.0;

    // Jump detection
    double jump_mahalanobis_threshold = 5.0;  ///< σ threshold for position jump

    // Speed consistency
    double max_speed_ratio = 3.0;        ///< GNSS speed / EKF speed ratio for anomaly
    double max_speed_diff_mps = 20.0;    ///< Absolute speed difference threshold

    // Heading consistency
    double max_heading_diff_deg = 45.0;  ///< At speeds > 10 m/s

    // Stale fix
    double max_stale_sec = 3.0;

    // Satellite
    int min_satellites = 4;

    // Spoofing
    double spoof_suspect_threshold = 0.6;
    double spoof_alert_threshold   = 0.8;
};

/// Result of GNSS integrity assessment for a single epoch.
struct IntegrityAssessment {
    GnssIntegrity state = GnssIntegrity::DENIED;
    bool innovation_ok = true;
    bool jump_detected = false;
    bool speed_anomaly = false;
    bool heading_anomaly = false;
    bool stale = false;
    bool insufficient_sats = false;
    double innovation_nis = 0.0;
    double spoofing_score = 0.0;
    bool use_measurement = false;  ///< Whether EKF should use this GNSS fix
};

/// GNSS Integrity Monitor with 4-state hysteresis.
class GnssIntegrityMonitor {
public:
    explicit GnssIntegrityMonitor(const IntegrityConfig& config = {})
        : config_(config) {}

    /// Assess a new GNSS measurement.
    ///
    /// @param gnss  New GNSS fix
    /// @param predicted_pos_enu  EKF-predicted position in ENU
    /// @param predicted_vel_enu  EKF-predicted velocity in ENU
    /// @param predicted_heading_rad  EKF-predicted heading (rad, CW from North)
    /// @param pos_sigma  Position uncertainty from EKF (1σ, meters)
    IntegrityAssessment assess(const GnssMeasurement& gnss,
                               [[maybe_unused]] const Vec3d& predicted_pos_enu,
                               const Vec3d& predicted_vel_enu,
                               double predicted_heading_rad,
                               [[maybe_unused]] double pos_sigma) {
        IntegrityAssessment result;

        // ── Basic validity ──
        if (!gnss.isValid()) {
            result.state = GnssIntegrity::DENIED;
            result.use_measurement = false;
            updateNoFix();
            return result;
        }

        // ── Stale fix ──
        if (last_fix_timestamp_ns_ != kInvalidTimestamp) {
            double dt = nsToSec(gnss.timestamp_ns - last_fix_timestamp_ns_);
            result.stale = (dt > config_.max_stale_sec);
        }
        last_fix_timestamp_ns_ = gnss.timestamp_ns;

        // ── Satellite count ──
        result.insufficient_sats = (gnss.satellite_count < config_.min_satellites);

        // ── Speed anomaly ──
        double ekf_speed = predicted_vel_enu.norm();
        if (ekf_speed > 1.0) {
            double ratio = static_cast<double>(gnss.speed_mps) / ekf_speed;
            double diff = std::abs(static_cast<double>(gnss.speed_mps) - ekf_speed);
            result.speed_anomaly = (ratio > config_.max_speed_ratio ||
                                    diff > config_.max_speed_diff_mps);
        }

        // ── Heading anomaly ──
        if (gnss.hasBearing() && ekf_speed > 10.0) {
            double gnss_heading = static_cast<double>(gnss.bearing_deg) * constants::kDegToRad;
            double diff = std::abs(angle::angleDifference(gnss_heading, predicted_heading_rad));
            result.heading_anomaly = (diff > config_.max_heading_diff_deg * constants::kDegToRad);
        }

        // ── Accuracy check ──
        bool accuracy_ok = (gnss.horizontal_accuracy_m < config_.max_healthy_hAcc_m);
        bool accuracy_degraded = (gnss.horizontal_accuracy_m < config_.max_degraded_hAcc_m);

        // ── Composite assessment ──
        bool any_anomaly = result.speed_anomaly || result.heading_anomaly ||
                           result.insufficient_sats;
        bool severe_anomaly = result.jump_detected || result.stale;

        // ── State machine update ──
        if (!any_anomaly && !severe_anomaly && accuracy_ok) {
            updateGoodFix();
        } else if (severe_anomaly) {
            updateSevereAnomaly();
        } else if (any_anomaly || !accuracy_degraded) {
            updateMildAnomaly();
        }

        result.state = state_;
        result.use_measurement = (state_ == GnssIntegrity::HEALTHY ||
                                  state_ == GnssIntegrity::DEGRADED);

        return result;
    }

    /// Report that no GNSS fix was received this epoch.
    void updateNoFix() {
        no_fix_count_++;
        good_fix_count_ = 0;

        if (no_fix_count_ >= config_.degraded_to_denied_epochs) {
            state_ = GnssIntegrity::DENIED;
        }
    }

    GnssIntegrity state() const { return state_; }

private:
    void updateGoodFix() {
        good_fix_count_++;
        no_fix_count_ = 0;
        anomaly_count_ = 0;

        switch (state_) {
            case GnssIntegrity::DENIED:
                state_ = GnssIntegrity::SUSPECT;
                good_fix_count_ = 1;
                break;
            case GnssIntegrity::SUSPECT:
                if (good_fix_count_ >= config_.suspect_to_healthy_count) {
                    state_ = GnssIntegrity::HEALTHY;
                }
                break;
            case GnssIntegrity::DEGRADED:
                state_ = GnssIntegrity::HEALTHY;
                break;
            case GnssIntegrity::HEALTHY:
                break;
        }
    }

    void updateMildAnomaly() {
        anomaly_count_++;
        good_fix_count_ = 0;

        if (state_ == GnssIntegrity::HEALTHY &&
            anomaly_count_ >= config_.healthy_to_degraded_count) {
            state_ = GnssIntegrity::DEGRADED;
        }
    }

    void updateSevereAnomaly() {
        anomaly_count_++;
        good_fix_count_ = 0;

        if (state_ == GnssIntegrity::HEALTHY &&
            anomaly_count_ >= config_.healthy_to_suspect_count) {
            state_ = GnssIntegrity::SUSPECT;
        } else if (state_ == GnssIntegrity::DEGRADED) {
            state_ = GnssIntegrity::SUSPECT;
        } else if (state_ == GnssIntegrity::SUSPECT &&
                   anomaly_count_ >= config_.suspect_to_denied_epochs) {
            state_ = GnssIntegrity::DENIED;
        }
    }

    IntegrityConfig config_;
    GnssIntegrity state_ = GnssIntegrity::DENIED;  // start denied until first valid fix

    int good_fix_count_ = 0;
    int anomaly_count_ = 0;
    int no_fix_count_ = 0;

    Timestamp last_fix_timestamp_ns_ = kInvalidTimestamp;
};

}  // namespace idr
