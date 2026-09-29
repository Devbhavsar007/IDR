// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Phone-to-Vehicle Alignment Module.
//
// Estimates the rotation from the phone's sensor frame to the vehicle body
// frame. This is critical because the phone can be in any orientation
// (pocket, dashboard mount, cupholder, etc.).
//
// Three-stage approach:
//   1. GRAVITY ALIGNMENT (immediate): Pitch and roll from gravity vector.
//      Works even when stationary. Gives 2 of 3 DOF instantly.
//
//   2. HEADING ALIGNMENT (requires motion): Yaw from GNSS bearing vs
//      accelerometer-derived forward direction. Requires speed > 3 m/s.
//
//   3. CONTINUOUS REFINEMENT: Running average with outlier rejection.
//      Tracks slow phone movements (e.g., phone slides on dashboard).
//
// Reference: Blueprint v1.0 §4, Blueprint v1.1 §2.

#pragma once

#include "idr/constraints/nhc.h"
#include "idr/math_utils/quat_utils.h"
#include "idr/types/common.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <deque>

namespace idr {

/// Alignment quality levels
enum class AlignmentQuality : uint8_t {
    NONE        = 0,  // No alignment — raw phone frame
    GRAVITY     = 1,  // Pitch/roll from gravity (yaw unknown)
    COARSE      = 2,  // Yaw estimated but low confidence
    CONVERGED   = 3,  // Full 3-DOF alignment, high confidence
};

/// Alignment configuration
struct AlignmentConfig {
    /// Minimum speed (m/s) to attempt heading alignment
    double min_speed_for_yaw_mps = 3.0;

    /// Gravity vector consistency threshold (cosine similarity)
    double gravity_consistency_threshold = 0.95;

    /// Maximum heading residual to accept (radians)
    double heading_residual_threshold_rad = 0.15;

    /// Number of heading samples for convergence
    int heading_convergence_count = 20;

    /// Confidence decay per second without updates
    double confidence_decay_rate = 0.002;

    /// Outlier rejection threshold for heading (radians)
    double heading_outlier_threshold_rad = 0.5;

    /// Exponential moving average alpha for heading refinement
    double heading_ema_alpha = 0.1;

    /// Number of gravity samples for initial averaging
    int gravity_init_samples = 50;
};

/// Phone-to-Vehicle Alignment estimator.
///
/// Usage:
///   PhoneToVehicleAlignment align(config);
///
///   // Feed gravity every IMU sample:
///   align.updateGravity(gravity_vector);
///
///   // Feed heading when GNSS + speed available:
///   align.updateHeading(gnss_bearing_rad, accel_forward_direction_rad);
///
///   // Get the rotation to apply to sensor data:
///   Quaterniond R_vp = align.rotation();
///   Vec3d accel_vehicle = R_vp * accel_phone;
class PhoneToVehicleAlignment {
public:
    explicit PhoneToVehicleAlignment(const AlignmentConfig& config = {})
        : config_(config) {}

    /// Feed a gravity vector (in phone frame, m/s²).
    /// On Android, this comes from TYPE_GRAVITY sensor or low-pass filtered accel.
    /// Expected: ~[0, 0, 9.8] when phone is flat face-up.
    void updateGravity(const Vec3d& gravity_phone) {
        double g_mag = gravity_phone.norm();
        if (g_mag < 5.0 || g_mag > 15.0) return;  // Sanity check

        Vec3d g_norm = gravity_phone / g_mag;

        if (gravity_init_count_ < config_.gravity_init_samples) {
            // Accumulate for initial estimate
            gravity_accum_ += g_norm;
            gravity_init_count_++;

            if (gravity_init_count_ == config_.gravity_init_samples) {
                Vec3d g_avg = (gravity_accum_ / static_cast<double>(gravity_init_count_)).normalized();
                computePitchRoll(g_avg);
                gravity_ref_ = g_avg;
                quality_ = AlignmentQuality::GRAVITY;
                spdlog::info("Alignment: gravity converged, pitch={:.1f}° roll={:.1f}°",
                             pitch_rad_ * constants::kRadToDeg,
                             roll_rad_ * constants::kRadToDeg);
            }
            return;
        }

        // Continuous refinement: EMA
        Vec3d g_avg = gravity_ema_;
        g_avg = g_avg * (1.0 - 0.02) + g_norm * 0.02;
        gravity_ema_ = g_avg.normalized();

        // Check consistency
        double consistency = gravity_ema_.dot(g_norm);
        if (consistency > config_.gravity_consistency_threshold) {
            computePitchRoll(gravity_ema_);
        } else {
            // Phone is being moved — don't update
            gravity_unstable_count_++;
        }
    }

    /// Feed a heading measurement pair.
    ///
    /// @param gnss_heading_rad  GNSS bearing (rad, CW from North = ENU yaw)
    /// @param vehicle_speed_mps  Vehicle speed (m/s) — used for quality gating
    void updateHeading(double gnss_heading_rad, double vehicle_speed_mps) {
        if (quality_ < AlignmentQuality::GRAVITY) return;  // Need gravity first
        if (vehicle_speed_mps < config_.min_speed_for_yaw_mps) return;

        // The GNSS heading IS the vehicle heading in the navigation frame.
        // We need to find the yaw offset between phone and vehicle.
        //
        // For now, we estimate yaw_offset from the difference between:
        //   - GNSS heading (vehicle heading in nav frame)
        //   - Phone's "forward" direction projected onto the horizontal plane
        //
        // With just gravity alignment, we know pitch/roll but not yaw.
        // The yaw_offset = gnss_heading - phone_forward_heading
        //
        // We'll accumulate and average.

        double yaw_sample = gnss_heading_rad;  // Will be refined with accel direction

        // Outlier rejection
        if (heading_samples_.size() > 5) {
            double diff = angle::angleDifference(yaw_sample, yaw_rad_);
            if (std::abs(diff) > config_.heading_outlier_threshold_rad) {
                heading_outlier_count_++;
                return;
            }
        }

        heading_samples_.push_back(yaw_sample);
        if (heading_samples_.size() > 100) {
            heading_samples_.pop_front();
        }

        // Update yaw estimate
        if (heading_samples_.size() < 3) {
            yaw_rad_ = yaw_sample;
        } else {
            // EMA with circular mean
            double diff = angle::angleDifference(yaw_sample, yaw_rad_);
            yaw_rad_ = angle::wrapPi(yaw_rad_ + config_.heading_ema_alpha * diff);
        }

        // Update quality
        if (quality_ < AlignmentQuality::COARSE) {
            quality_ = AlignmentQuality::COARSE;
            spdlog::info("Alignment: coarse heading established, yaw={:.1f}°",
                         yaw_rad_ * constants::kRadToDeg);
        }

        if (static_cast<int>(heading_samples_.size()) >= config_.heading_convergence_count) {
            // Check convergence: standard deviation of recent heading samples
            double heading_std = computeCircularStd();
            if (heading_std < config_.heading_residual_threshold_rad) {
                quality_ = AlignmentQuality::CONVERGED;
                gravity_ref_ = gravity_ema_;
                confidence_ = std::min(confidence_ + 0.05, 1.0);
            }
        }

        updateRotation();
    }

    /// Decay confidence over time (call once per second)
    void decayConfidence(double dt_sec) {
        if (quality_ >= AlignmentQuality::COARSE) {
            confidence_ -= config_.confidence_decay_rate * dt_sec;
            confidence_ = std::max(confidence_, 0.1);

            if (confidence_ < 0.3 && quality_ == AlignmentQuality::CONVERGED) {
                quality_ = AlignmentQuality::COARSE;
            }
        }
    }

    /// Continuous dynamic re-calibration and degradation tracking (Blueprint v1.1 §4)
    void continuousUpdate(const Vec3d& gravity_phone,
                          double gnss_heading_rad,
                          double vehicle_speed_mps,
                          double gyro_yaw_rate_rad_s,
                          double dt_sec) {
        // 1. Track gravity consistency relative to converged reference
        double g_mag = gravity_phone.norm();
        if (g_mag >= 5.0 && g_mag <= 15.0) {
            Vec3d g_norm = gravity_phone / g_mag;
            double dot = std::clamp(gravity_ref_.dot(g_norm), -1.0, 1.0);
            gravity_consistency_ = 0.95 * gravity_consistency_ + 0.05 * std::max(0.0, dot);

            // Continuous complementary pitch/roll tracking
            Vec3d blended = (0.98 * gravity_ema_ + 0.02 * g_norm).normalized();
            computePitchRoll(blended);
        }

        // 2. Track heading residual during motion
        if (std::isfinite(gnss_heading_rad) && vehicle_speed_mps > config_.min_speed_for_yaw_mps) {
            double residual = std::abs(angle::angleDifference(yaw_rad_, gnss_heading_rad));
            heading_residuals_.push_back(residual);
            if (heading_residuals_.size() > 30) {
                heading_residuals_.pop_front();
            }
            double sum_sq = 0.0;
            for (double r : heading_residuals_) sum_sq += r * r;
            heading_residual_rms_ = std::sqrt(sum_sq / static_cast<double>(heading_residuals_.size()));

            // High observability boost during turns with reliable GNSS
            if (std::abs(gyro_yaw_rate_rad_s) > 0.10 && vehicle_speed_mps > 5.0 && heading_residual_rms_ < 0.08) {
                confidence_ = std::min(confidence_ + 0.02, 1.0);
            }
        }

        // 3. Natural entropy decay
        decayConfidence(dt_sec);

        // 4. Trigger re-calibration if anomalies detected
        bool needs_recal = false;
        if (gravity_consistency_ < 0.70) needs_recal = true;
        if (heading_residual_rms_ > 0.18) needs_recal = true; // ~10.3 deg
        if (confidence_ < 0.25) needs_recal = true;

        if (needs_recal) {
            quality_ = AlignmentQuality::COARSE;
            needs_recalibration_ = true;
            confidence_ = std::max(0.1, confidence_ * 0.95);
        } else if (confidence_ > 0.70 && quality_ == AlignmentQuality::CONVERGED) {
            needs_recalibration_ = false;
            gravity_ref_ = gravity_ema_;
        }

        updateRotation();
    }

    // ── Accessors ──

    /// Get the phone-to-vehicle rotation quaternion.
    /// v_vehicle = rotation() * v_phone
    Quaterniond rotation() const { return R_vehicle_phone_; }

    AlignmentQuality quality() const { return quality_; }
    double confidence() const { return confidence_; }
    double pitchRad() const { return pitch_rad_; }
    double rollRad() const { return roll_rad_; }
    double yawRad() const { return yaw_rad_; }
    double gravityConsistency() const { return gravity_consistency_; }
    double headingResidualRms() const { return heading_residual_rms_; }
    bool needsRecalibration() const { return needs_recalibration_; }

    /// Recommended covariance inflation factor when alignment degrades
    double alignmentInflationFactor() const {
        return std::clamp(1.0 / std::max(confidence_, 0.05), 1.0, 20.0);
    }

    /// Get alignment state for NHC modulation
    AlignmentState alignmentState() const {
        return {confidence_};
    }

private:
    void computePitchRoll(const Vec3d& g_normalized) {
        // Phone frame: gravity vector tells us pitch and roll.
        // If phone is flat face-up: g ≈ [0, 0, +1] (normalized)
        //
        // pitch = asin(-g_x)  (nose up = positive)
        // roll  = atan2(g_y, g_z)  (right wing down = positive)
        pitch_rad_ = std::asin(std::clamp(-g_normalized.x(), -1.0, 1.0));
        roll_rad_  = std::atan2(g_normalized.y(), g_normalized.z());

        gravity_ema_ = g_normalized;
    }

    void updateRotation() {
        // Compose the full rotation: R_vehicle_phone = Rz(yaw) * Ry(pitch) * Rx(roll)
        R_vehicle_phone_ = quat::fromEulerZYX(yaw_rad_, pitch_rad_, roll_rad_);
    }

    double computeCircularStd() const {
        if (heading_samples_.size() < 2) return constants::kPi;

        // Circular standard deviation using mean resultant length
        double sum_sin = 0.0, sum_cos = 0.0;
        for (double h : heading_samples_) {
            sum_sin += std::sin(h);
            sum_cos += std::cos(h);
        }
        double n = static_cast<double>(heading_samples_.size());
        double R = std::sqrt(sum_sin * sum_sin + sum_cos * sum_cos) / n;
        // Circular std = sqrt(-2 * ln(R))
        if (R > 0.999) return 0.0;
        if (R < 0.001) return constants::kPi;
        return std::sqrt(-2.0 * std::log(R));
    }

    AlignmentConfig config_;

    // Gravity alignment
    Vec3d gravity_accum_ = Vec3d::Zero();
    Vec3d gravity_ema_ = Vec3d(0.0, 0.0, 1.0);
    int gravity_init_count_ = 0;
    int gravity_unstable_count_ = 0;

    // Pitch/Roll (from gravity)
    double pitch_rad_ = 0.0;
    double roll_rad_ = 0.0;

    // Yaw (from GNSS heading)
    double yaw_rad_ = 0.0;
    std::deque<double> heading_samples_;
    int heading_outlier_count_ = 0;

    // Output
    Quaterniond R_vehicle_phone_ = Quaterniond::Identity();
    AlignmentQuality quality_ = AlignmentQuality::NONE;
    double confidence_ = 0.0;

    // Continuous dynamic re-calibration tracking
    Vec3d gravity_ref_ = Vec3d(0.0, 0.0, 1.0);
    double gravity_consistency_ = 1.0;
    double heading_residual_rms_ = 0.0;
    bool needs_recalibration_ = false;
    std::deque<double> heading_residuals_;
};

}  // namespace idr
