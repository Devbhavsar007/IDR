// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Explicit GNSS Spoofing Defense Engine
// Detects sophisticated false GNSS signals using kinematic consistency,
// trajectory curvature cross-correlation, and RF satellite C/N0 signatures.

#pragma once

#include "idr/sensors/sensor_types.h"
#include "idr/types/common.h"
#include "idr/math_utils/quat_utils.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

namespace idr {

/// Satellite constellation signal information for spoofing detection.
struct SatelliteSignalInfo {
    int svid = 0;
    float elevation_deg = 0.0f;
    float azimuth_deg = 0.0f;
    float cn0_dbhz = 0.0f;
    bool used_in_fix = true;
};

/// Parameters for GNSS spoofing detector.
struct SpoofingDetectorConfig {
    double weight_velocity = 0.35;
    double weight_curvature = 0.30;
    double weight_clock = 0.15;
    double weight_cn0 = 0.20;

    double max_expected_curvature_diff_rad_s = 0.05; // ~2.8 deg/s
    double min_expected_cn0_std_dev = 2.0;          // dB-Hz (real signals typically 5-10 dB-Hz)
    double spoof_suspect_threshold = 0.6;
    double spoof_alert_threshold = 0.8;
    int suspect_consecutive_epochs = 3;
    int denied_consecutive_epochs = 5;
};

/// Multi-feature GNSS Spoofing Detector.
class SpoofingDetector {
public:
    explicit SpoofingDetector(const SpoofingDetectorConfig& config = {})
        : config_(config) {}

    /// 1. Velocity cross-check: GNSS velocity vs EKF-predicted / IMU velocity.
    /// Returns score in [0.0, 1.0].
    double checkVelocity(const Vec3d& gnss_vel_enu,
                         const Vec3d& imu_vel_enu,
                         double gnss_speed_accuracy_mps,
                         double imu_vel_sigma_mps = 0.5) const {
        Vec3d diff = gnss_vel_enu - imu_vel_enu;
        double mismatch = diff.norm();
        double sigma = std::max(gnss_speed_accuracy_mps, 0.5) + imu_vel_sigma_mps;
        return std::clamp(mismatch / (3.0 * sigma), 0.0, 1.0);
    }

    /// 2. Trajectory curvature cross-check: GNSS heading change rate vs Gyro yaw rate.
    /// Returns score in [0.0, 1.0].
    double checkCurvature(double gnss_heading_rate_rad_s,
                          double gyro_yaw_rate_rad_s) const {
        double diff = std::abs(gnss_heading_rate_rad_s - gyro_yaw_rate_rad_s);
        return std::clamp(diff / (3.0 * config_.max_expected_curvature_diff_rad_s), 0.0, 1.0);
    }

    /// 3. C/N0 uniformity check: Synthesized/spoofed RF signals often produce
    /// unnaturally uniform power levels across all satellites regardless of elevation.
    /// Real signals exhibit significant elevation-dependent variance (std dev ~5-10 dB-Hz).
    double checkCn0Uniformity(const std::vector<SatelliteSignalInfo>& satellites) const {
        if (satellites.size() < 5) {
            return 0.0; // Insufficient data to assert spoofing based on C/N0
        }

        double sum = 0.0;
        for (const auto& sat : satellites) {
            sum += sat.cn0_dbhz;
        }
        double mean = sum / static_cast<double>(satellites.size());

        double var_sum = 0.0;
        for (const auto& sat : satellites) {
            double d = sat.cn0_dbhz - mean;
            var_sum += d * d;
        }
        double std_dev = std::sqrt(var_sum / static_cast<double>(satellites.size()));

        // If standard deviation is unrealistically low with multiple satellites, flag high spoof probability
        if (std_dev < config_.min_expected_cn0_std_dev) {
            return 0.85;
        } else if (std_dev < 3.0) {
            return 0.30;
        }
        return 0.0;
    }

    /// 4. Clock / pseudorange rate consistency.
    double checkClockConsistency(double pseudorange_rate_residual_mps,
                                 double expected_variance_mps = 0.2) const {
        return std::clamp(std::abs(pseudorange_rate_residual_mps) / (3.0 * expected_variance_mps), 0.0, 1.0);
    }

    /// Composite evaluation of GNSS fix.
    double computeSpoofingScore(double vel_score,
                                double curv_score,
                                double clk_score,
                                double cn0_score) const {
        double score = config_.weight_velocity * vel_score +
                       config_.weight_curvature * curv_score +
                       config_.weight_clock * clk_score +
                       config_.weight_cn0 * cn0_score;
        return std::clamp(score, 0.0, 1.0);
    }

    /// Update tracking history and evaluate hysteresis state transition.
    /// Returns true if GNSS is classified as compromised / spoofed.
    bool update(double composite_score) {
        last_score_ = composite_score;

        if (composite_score >= config_.spoof_alert_threshold) {
            alert_epochs_++;
            suspect_epochs_++;
        } else if (composite_score >= config_.spoof_suspect_threshold) {
            suspect_epochs_++;
            alert_epochs_ = std::max(0, alert_epochs_ - 1);
        } else {
            suspect_epochs_ = std::max(0, suspect_epochs_ - 1);
            alert_epochs_ = std::max(0, alert_epochs_ - 1);
        }

        if (alert_epochs_ >= config_.denied_consecutive_epochs) {
            is_spoofed_ = true;
        } else if (suspect_epochs_ >= config_.suspect_consecutive_epochs) {
            is_suspect_ = true;
        } else if (suspect_epochs_ == 0) {
            is_suspect_ = false;
            is_spoofed_ = false;
        }

        return is_spoofed_ || is_suspect_;
    }

    double lastScore() const { return last_score_; }
    bool isSuspect() const { return is_suspect_; }
    bool isSpoofed() const { return is_spoofed_; }
    int suspectEpochs() const { return suspect_epochs_; }
    int alertEpochs() const { return alert_epochs_; }

    void reset() {
        last_score_ = 0.0;
        suspect_epochs_ = 0;
        alert_epochs_ = 0;
        is_suspect_ = false;
        is_spoofed_ = false;
    }

private:
    SpoofingDetectorConfig config_;
    double last_score_ = 0.0;
    int suspect_epochs_ = 0;
    int alert_epochs_ = 0;
    bool is_suspect_ = false;
    bool is_spoofed_ = false;
};

} // namespace idr
