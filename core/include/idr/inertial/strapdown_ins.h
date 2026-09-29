// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Strapdown INS Mechanization.
//
// Implements classical inertial navigation equations:
//   1. Attitude propagation (quaternion integration)
//   2. Velocity propagation (gravity removal + rotation compensation)
//   3. Position propagation (velocity integration)
//
// Reference: Groves, "Principles of GNSS, Inertial, and Multisensor
//            Navigation Systems" (2nd ed.), Chapter 5.
//
// All computations in the Navigation frame (ENU).
// The INS mechanization does NOT perform any corrections — it is pure
// dead reckoning from IMU measurements. Corrections come from the EKF.

#pragma once

#include "idr/math_utils/geo_utils.h"
#include "idr/math_utils/quat_utils.h"
#include "idr/sensors/sensor_types.h"
#include "idr/types/common.h"

#include <spdlog/spdlog.h>

namespace idr {

/// Nominal INS state — the "truth" state that the error-state EKF corrects.
/// All vectors are expressed in the Navigation (ENU) frame unless stated.
struct NominalState {
    /// Position in local ENU frame (meters) relative to reference origin
    Vec3d position = Vec3d::Zero();

    /// Velocity in ENU frame (m/s)
    Vec3d velocity = Vec3d::Zero();

    /// Attitude: rotation from body (vehicle) frame to navigation (ENU) frame
    /// v_nav = attitude * v_body
    Quaterniond attitude = Quaterniond::Identity();

    /// Accelerometer bias estimate (m/s², body frame)
    Vec3d accel_bias = Vec3d::Zero();

    /// Gyroscope bias estimate (rad/s, body frame)
    Vec3d gyro_bias = Vec3d::Zero();

    /// Timestamp of this state (nanoseconds)
    Timestamp timestamp_ns = kInvalidTimestamp;
};

/// Strapdown INS mechanization engine.
///
/// Propagates the nominal state forward in time using IMU measurements.
/// This is the core dead-reckoning integration that produces a position
/// estimate from acceleration and angular rate alone.
///
/// IMPORTANT: This class does NOT correct for errors. It accumulates drift.
/// The ErrorStateEkf applies corrections by resetting the nominal state.
class StrapdownIns {
public:
    StrapdownIns() = default;

    /// Initialize the INS at a known state.
    /// Typically called once when the first GNSS fix + gravity alignment is available.
    void initialize(const NominalState& initial_state) {
        state_ = initial_state;
        initialized_ = true;
        spdlog::info("INS initialized at t={:.3f}s, pos=[{:.1f}, {:.1f}, {:.1f}]",
                     nsToSec(state_.timestamp_ns),
                     state_.position.x(), state_.position.y(), state_.position.z());
    }

    /// Propagate the INS forward by one IMU sample.
    ///
    /// @param imu  IMU sample in the BODY frame (after bias compensation
    ///             and phone-to-vehicle rotation have been applied upstream).
    /// @param dt   Time step in seconds (from TimestampValidator).
    ///
    /// Uses a midpoint integration scheme for better accuracy:
    ///   1. Compute attitude at midpoint using half the rotation
    ///   2. Transform acceleration using midpoint attitude
    ///   3. Integrate velocity and position
    void propagate(const ImuSample& imu, double dt) {
        if (!initialized_) {
            spdlog::warn("INS::propagate called before initialization");
            return;
        }
        if (dt <= 0.0 || dt > 1.0) {
            spdlog::warn("INS::propagate: invalid dt={:.6f}s, skipping", dt);
            return;
        }

        // ── Step 0: Bias-compensated measurements ──
        // (Caller should have already applied phone-to-vehicle rotation)
        Vec3d accel_body = imu.accel - state_.accel_bias;
        Vec3d gyro_body  = imu.gyro  - state_.gyro_bias;

        // ── Step 1: Attitude propagation (quaternion midpoint method) ──
        //
        //   ω = gyro_body (bias-compensated angular rate in body frame)
        //   θ = ω * dt (rotation vector for this step)
        //   q(t+dt) = q(t) ⊗ q(θ)
        //
        //   For midpoint: q_mid = q(t) ⊗ q(θ/2)
        Vec3d theta = gyro_body * dt;
        Quaterniond dq = quat::fromRotationVector(theta);
        Quaterniond dq_half = quat::fromRotationVector(theta * 0.5);

        Quaterniond attitude_mid = (state_.attitude * dq_half).normalized();
        Quaterniond attitude_new = (state_.attitude * dq).normalized();

        // ── Step 2: Specific force → navigation frame ──
        //
        //   f_nav = R_nav_body * f_body
        //   a_nav = f_nav - g_nav
        //
        //   Using midpoint attitude for better accuracy during turns
        Vec3d accel_nav = attitude_mid * accel_body;

        // Remove gravity (ENU: gravity is [0, 0, -g])
        Vec3d gravity_nav(0.0, 0.0, -constants::kGravity);
        Vec3d acceleration = accel_nav + gravity_nav;  // + because accel includes +g when stationary

        // ── Step 3: Velocity integration (trapezoidal-ish with midpoint attitude) ──
        Vec3d velocity_new = state_.velocity + acceleration * dt;

        // ── Step 4: Position integration (midpoint velocity) ──
        Vec3d velocity_mid = (state_.velocity + velocity_new) * 0.5;
        Vec3d position_new = state_.position + velocity_mid * dt;

        // ── Step 5: Update state ──
        state_.attitude = attitude_new;
        state_.velocity = velocity_new;
        state_.position = position_new;
        state_.timestamp_ns = imu.timestamp_ns;
    }

    /// Apply an error-state correction from the EKF.
    ///
    /// @param dp  Position error (meters, ENU)
    /// @param dv  Velocity error (m/s, ENU)
    /// @param dtheta  Attitude error (radians, rotation vector, body frame)
    /// @param dba  Accelerometer bias correction (m/s², body frame)
    /// @param dbg  Gyroscope bias correction (rad/s, body frame)
    void applyCorrection(const Vec3d& dp, const Vec3d& dv, const Vec3d& dtheta,
                         const Vec3d& dba, const Vec3d& dbg) {
        state_.position += dp;
        state_.velocity += dv;

        // Attitude correction: q_corrected = q_nominal ⊗ q(dθ)
        Quaterniond dq = quat::fromRotationVector(dtheta);
        state_.attitude = (state_.attitude * dq).normalized();

        state_.accel_bias += dba;
        state_.gyro_bias  += dbg;
    }

    /// Get current nominal state (const reference)
    const NominalState& state() const { return state_; }

    /// Get mutable reference to state (for EKF reset)
    NominalState& mutableState() { return state_; }

    bool isInitialized() const { return initialized_; }

private:
    NominalState state_;
    bool initialized_ = false;
};

}  // namespace idr
