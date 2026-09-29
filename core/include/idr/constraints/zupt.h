// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Zero-Velocity Update (ZUPT) Detector.
//
// Detects when the vehicle is stationary using multiple sensor cues:
//   1. Accelerometer variance (low when stationary)
//   2. Gyroscope magnitude (near-zero when stationary)
//   3. GNSS speed (near-zero when stationary)
//   4. Combined likelihood ratio
//
// When ZUPT is triggered, the EKF receives a pseudo-measurement
// constraining all three velocity components to zero.

#pragma once

#include "idr/sensors/sensor_types.h"
#include "idr/types/common.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>

namespace idr {

/// ZUPT configuration
struct ZuptConfig {
    /// Window size for variance computation (samples)
    int window_size = 50;

    /// Accelerometer variance threshold (m/s²)²
    /// Stationary: < 0.01. Driving: >> 0.1
    double accel_var_threshold = 0.02;

    /// Gyroscope magnitude threshold (rad/s)
    double gyro_mag_threshold = 0.03;

    /// GNSS speed threshold (m/s) — below this, GNSS supports stationarity
    double gnss_speed_threshold = 0.5;

    /// Minimum consecutive detections to trigger ZUPT
    int min_consecutive = 20;

    /// Maximum consecutive to hold ZUPT (avoid infinite stationarity)
    int max_consecutive = 5000;

    /// Velocity sigma for ZUPT measurement (m/s)
    double zupt_sigma = 0.01;

    /// ZARU (Zero Angular Rate Update) sigma (rad/s)
    double zaru_sigma = 0.001;
};

/// ZUPT detection result
struct ZuptResult {
    bool is_stationary = false;          ///< Vehicle is detected as stationary
    bool zupt_available = false;          ///< ZUPT pseudo-measurement should be applied
    bool zaru_available = false;          ///< ZARU pseudo-measurement should be applied
    double stationary_confidence = 0.0;   ///< [0, 1]
    double accel_variance = 0.0;          ///< Current accel variance
    double gyro_magnitude = 0.0;          ///< Current mean gyro magnitude
    int consecutive_count = 0;            ///< How many samples stationary
};

/// Zero-Velocity Update Detector.
class ZuptDetector {
public:
    explicit ZuptDetector(const ZuptConfig& config = {})
        : config_(config) {}

    /// Process a new IMU sample and return ZUPT status.
    ZuptResult update(const ImuSample& imu) {
        // Add to sliding windows
        accel_window_.push_back(imu.accel);
        gyro_window_.push_back(imu.gyro);

        if (static_cast<int>(accel_window_.size()) > config_.window_size) {
            accel_window_.pop_front();
        }
        if (static_cast<int>(gyro_window_.size()) > config_.window_size) {
            gyro_window_.pop_front();
        }

        ZuptResult result;

        // Need full window
        if (static_cast<int>(accel_window_.size()) < config_.window_size) {
            return result;
        }

        // ── Accelerometer variance ──
        // Compute variance of acceleration magnitude
        // When stationary, accel ≈ gravity constant, so variance is tiny
        result.accel_variance = computeAccelVariance();

        // ── Gyroscope magnitude ──
        // Mean angular rate magnitude over window
        result.gyro_magnitude = computeGyroMagnitude();

        // ── Detection logic ──
        bool accel_stationary = (result.accel_variance < config_.accel_var_threshold);
        bool gyro_stationary = (result.gyro_magnitude < config_.gyro_mag_threshold);

        // Both conditions must be met
        bool detected = accel_stationary && gyro_stationary;

        // Factor in GNSS speed if available
        if (last_gnss_speed_valid_) {
            bool gnss_stationary = (last_gnss_speed_ < config_.gnss_speed_threshold);
            // If GNSS says moving but IMU says stationary, trust GNSS at higher speeds
            if (!gnss_stationary && last_gnss_speed_ > 2.0) {
                detected = false;
            }
        }

        // Update consecutive counter
        if (detected) {
            consecutive_stationary_++;
        } else {
            consecutive_stationary_ = 0;
        }

        result.consecutive_count = consecutive_stationary_;

        // Apply hysteresis
        bool was_stationary = is_stationary_;
        if (consecutive_stationary_ >= config_.min_consecutive) {
            is_stationary_ = true;
        } else if (consecutive_stationary_ == 0) {
            is_stationary_ = false;
        }
        // else: keep previous state (hysteresis)

        result.is_stationary = is_stationary_;

        // Compute confidence
        if (is_stationary_) {
            double var_ratio = std::max(0.0, 1.0 - result.accel_variance / config_.accel_var_threshold);
            double gyro_ratio = std::max(0.0, 1.0 - result.gyro_magnitude / config_.gyro_mag_threshold);
            result.stationary_confidence = var_ratio * gyro_ratio;
        }

        // ZUPT/ZARU availability
        result.zupt_available = is_stationary_ &&
                                (consecutive_stationary_ < config_.max_consecutive);
        result.zaru_available = result.zupt_available;

        // Log transition
        if (is_stationary_ && !was_stationary) {
            spdlog::debug("ZUPT: vehicle now STATIONARY (accel_var={:.4f}, gyro={:.4f})",
                          result.accel_variance, result.gyro_magnitude);
        } else if (!is_stationary_ && was_stationary) {
            spdlog::debug("ZUPT: vehicle now MOVING");
        }

        return result;
    }

    /// Feed GNSS speed for cross-validation
    void updateGnssSpeed(double speed_mps) {
        last_gnss_speed_ = speed_mps;
        last_gnss_speed_valid_ = true;
    }

    /// Get ZUPT velocity sigma (for EKF measurement)
    double zuptSigma() const { return config_.zupt_sigma; }

    /// Get ZARU angular rate sigma (for EKF measurement)
    double zaruSigma() const { return config_.zaru_sigma; }

    bool isStationary() const { return is_stationary_; }

private:
    double computeAccelVariance() const {
        // Variance of acceleration magnitude
        // This is more robust than per-axis variance because gravity
        // magnitude is constant regardless of phone orientation
        double sum = 0.0, sum_sq = 0.0;
        for (const auto& a : accel_window_) {
            double mag = a.norm();
            sum += mag;
            sum_sq += mag * mag;
        }
        double n = static_cast<double>(accel_window_.size());
        double mean = sum / n;
        return (sum_sq / n) - (mean * mean);
    }

    double computeGyroMagnitude() const {
        double sum = 0.0;
        for (const auto& g : gyro_window_) {
            sum += g.norm();
        }
        return sum / static_cast<double>(gyro_window_.size());
    }

    ZuptConfig config_;

    std::deque<Vec3d> accel_window_;
    std::deque<Vec3d> gyro_window_;

    bool is_stationary_ = false;
    int consecutive_stationary_ = 0;

    double last_gnss_speed_ = 0.0;
    bool last_gnss_speed_valid_ = false;
};

}  // namespace idr
