// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Online Sensor Calibration.
//
// Estimates accelerometer and gyroscope biases during stationary periods.
// Uses a simple averaging approach that is robust to phone orientation
// because we subtract gravity before averaging accelerometer bias.
//
// This is a "warm start" calibration — it runs continuously and feeds
// bias estimates into the INS/EKF. NOT a full factory calibration.

#pragma once

#include "idr/math_utils/quat_utils.h"
#include "idr/types/common.h"

#include <spdlog/spdlog.h>
#include <cmath>
#include <deque>

namespace idr {

/// Calibration configuration
struct CalibrationConfig {
    /// Minimum stationary samples before computing bias
    int min_stationary_samples = 200;

    /// Maximum samples to accumulate (oldest dropped)
    int max_samples = 1000;

    /// EMA alpha for bias update (slow update)
    double bias_ema_alpha = 0.01;

    /// Maximum allowable bias magnitude (m/s² for accel, rad/s for gyro)
    double max_accel_bias = 2.0;
    double max_gyro_bias = 0.1;
};

/// Calibration state
struct CalibrationState {
    Vec3d accel_bias = Vec3d::Zero();   ///< Estimated accelerometer bias (body frame)
    Vec3d gyro_bias = Vec3d::Zero();    ///< Estimated gyroscope bias (body frame)
    bool accel_calibrated = false;
    bool gyro_calibrated = false;
    int stationary_samples = 0;
};

/// Online Sensor Calibrator.
///
/// When the vehicle is stationary (as detected by ZuptDetector):
///   - Gyroscope bias = mean gyro reading (should be ~0 when stationary)
///   - Accelerometer bias = mean accel reading - gravity vector
///     (requires gravity direction from alignment)
class OnlineSensorCalibrator {
public:
    explicit OnlineSensorCalibrator(const CalibrationConfig& config = {})
        : config_(config) {}

    /// Feed a sample during a confirmed stationary period.
    /// @param accel  Raw accelerometer reading (phone frame, m/s²)
    /// @param gyro   Raw gyroscope reading (phone frame, rad/s)
    /// @param gravity_direction  Normalized gravity direction in phone frame
    void addStationarySample(const Vec3d& accel, const Vec3d& gyro,
                             const Vec3d& gravity_direction) {
        // ── Gyroscope bias ──
        // When stationary, gyro should read zero. Any reading IS the bias.
        gyro_samples_.push_back(gyro);
        if (static_cast<int>(gyro_samples_.size()) > config_.max_samples) {
            gyro_samples_.pop_front();
        }

        // ── Accelerometer bias ──
        // When stationary, accel = gravity + bias (in phone frame)
        // So: bias = accel - gravity_magnitude * gravity_direction
        Vec3d expected_accel = constants::kGravity * gravity_direction;
        Vec3d accel_residual = accel - expected_accel;
        accel_samples_.push_back(accel_residual);
        if (static_cast<int>(accel_samples_.size()) > config_.max_samples) {
            accel_samples_.pop_front();
        }

        state_.stationary_samples = static_cast<int>(gyro_samples_.size());

        // Compute bias estimates once we have enough samples
        if (state_.stationary_samples >= config_.min_stationary_samples) {
            updateBiasEstimates();
        }
    }

    /// Called when the vehicle starts moving — freeze current estimates.
    void onMotionStart() {
        // Don't clear accumulated data — keep it for cross-session averaging
        spdlog::debug("Calibration: motion started. accel_bias=[{:.4f},{:.4f},{:.4f}] "
                       "gyro_bias=[{:.5f},{:.5f},{:.5f}]",
                       state_.accel_bias.x(), state_.accel_bias.y(), state_.accel_bias.z(),
                       state_.gyro_bias.x(), state_.gyro_bias.y(), state_.gyro_bias.z());
    }

    /// Reset calibration (new session)
    void reset() {
        accel_samples_.clear();
        gyro_samples_.clear();
        state_ = CalibrationState{};
    }

    const CalibrationState& state() const { return state_; }

private:
    void updateBiasEstimates() {
        // ── Gyroscope bias ──
        Vec3d gyro_mean = Vec3d::Zero();
        for (const auto& g : gyro_samples_) {
            gyro_mean += g;
        }
        gyro_mean /= static_cast<double>(gyro_samples_.size());

        // Sanity check
        if (gyro_mean.norm() < config_.max_gyro_bias) {
            if (!state_.gyro_calibrated) {
                state_.gyro_bias = gyro_mean;
                state_.gyro_calibrated = true;
                spdlog::info("Calibration: gyro bias=[{:.5f},{:.5f},{:.5f}] rad/s",
                             gyro_mean.x(), gyro_mean.y(), gyro_mean.z());
            } else {
                // EMA update
                state_.gyro_bias = state_.gyro_bias * (1.0 - config_.bias_ema_alpha)
                                 + gyro_mean * config_.bias_ema_alpha;
            }
        }

        // ── Accelerometer bias ──
        Vec3d accel_mean = Vec3d::Zero();
        for (const auto& a : accel_samples_) {
            accel_mean += a;
        }
        accel_mean /= static_cast<double>(accel_samples_.size());

        if (accel_mean.norm() < config_.max_accel_bias) {
            if (!state_.accel_calibrated) {
                state_.accel_bias = accel_mean;
                state_.accel_calibrated = true;
                spdlog::info("Calibration: accel bias=[{:.4f},{:.4f},{:.4f}] m/s²",
                             accel_mean.x(), accel_mean.y(), accel_mean.z());
            } else {
                state_.accel_bias = state_.accel_bias * (1.0 - config_.bias_ema_alpha)
                                  + accel_mean * config_.bias_ema_alpha;
            }
        }
    }

    CalibrationConfig config_;
    CalibrationState state_;

    std::deque<Vec3d> accel_samples_;
    std::deque<Vec3d> gyro_samples_;
};

}  // namespace idr
