// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Error-State Extended Kalman Filter (ES-EKF).
//
// 15-element error state:
//   δp  [3]  — position error (ENU, meters)
//   δv  [3]  — velocity error (ENU, m/s)
//   δθ  [3]  — attitude error (body-frame rotation vector, radians)
//   δba [3]  — accelerometer bias error (body, m/s²)
//   δbg [3]  — gyroscope bias error (body, rad/s)
//
// Reference: Solà, "Quaternion kinematics for the error-state Kalman filter" (2017)
//
// The filter maintains the error-state covariance and applies measurement
// updates. After each update, the error state is applied to the nominal
// state (via StrapdownIns::applyCorrection) and the error state is reset.

#pragma once

#include "idr/inertial/strapdown_ins.h"
#include "idr/math_utils/quat_utils.h"
#include "idr/types/common.h"

#include <spdlog/spdlog.h>
#include <Eigen/Dense>
#include <cmath>

namespace idr {

/// Error-state indices into the 15-element state vector.
namespace eskf {
    constexpr int kStateSize = 15;
    constexpr int kPosIdx    = 0;   // [0..2]  position error
    constexpr int kVelIdx    = 3;   // [3..5]  velocity error
    constexpr int kAttIdx    = 6;   // [6..8]  attitude error (rotation vector)
    constexpr int kAbIdx     = 9;   // [9..11] accelerometer bias error
    constexpr int kGbIdx     = 12;  // [12..14] gyroscope bias error
}

/// Process noise configuration.
/// Controls how quickly the filter's uncertainty grows during propagation.
struct ProcessNoise {
    /// Accelerometer noise density (m/s²/√Hz)
    double accel_noise = 0.05;

    /// Gyroscope noise density (rad/s/√Hz)
    double gyro_noise = 0.005;

    /// Accelerometer bias random walk (m/s²·√Hz)
    double accel_bias_rw = 0.001;

    /// Gyroscope bias random walk (rad/s·√Hz)
    double gyro_bias_rw = 0.0001;
};

/// Error-State Extended Kalman Filter.
class ErrorStateEkf {
public:
    using StateVec   = Eigen::Matrix<double, eskf::kStateSize, 1>;
    using StateMat   = Eigen::Matrix<double, eskf::kStateSize, eskf::kStateSize>;

    ErrorStateEkf() {
        x_.setZero();
        P_.setIdentity();
        // Initial covariance: moderate uncertainty
        P_.block<3, 3>(eskf::kPosIdx, eskf::kPosIdx) *= 100.0;    // 10m 1σ per axis
        P_.block<3, 3>(eskf::kVelIdx, eskf::kVelIdx) *= 1.0;      // 1 m/s 1σ
        P_.block<3, 3>(eskf::kAttIdx, eskf::kAttIdx) *= 0.01;     // ~5.7° 1σ
        P_.block<3, 3>(eskf::kAbIdx,  eskf::kAbIdx)  *= 0.01;     // 0.1 m/s² 1σ
        P_.block<3, 3>(eskf::kGbIdx,  eskf::kGbIdx)  *= 0.0001;   // 0.01 rad/s 1σ
    }

    /// Predict (propagate) the error-state covariance.
    ///
    /// @param nominal  Current nominal state (after INS propagation)
    /// @param accel_body  Bias-compensated accelerometer reading (body frame)
    /// @param dt  Time step in seconds
    void predict(const NominalState& nominal, const Vec3d& accel_body, double dt) {
        if (dt <= 0.0 || dt > 1.0) return;

        // ── Linearized error-state transition matrix F ──
        //
        //   The continuous-time F matrix for the error state is:
        //
        //   F = | 0   I   0   0   0  |  ← δp_dot = δv
        //       | 0   0  -R[a]× -R  0  |  ← δv_dot = -R[a×]δθ - R·δba
        //       | 0   0  -[ω]×  0  -I  |  ← δθ_dot = -[ω×]δθ - δbg
        //       | 0   0   0   0   0  |  ← δba_dot = noise
        //       | 0   0   0   0   0  |  ← δbg_dot = noise
        //
        // Discretize: Φ ≈ I + F·dt (first-order)

        Mat3d R = nominal.attitude.toRotationMatrix();
        Mat3d accel_skew = quat::skewSymmetric(accel_body);

        StateMat F = StateMat::Zero();

        // δp_dot = δv
        F.block<3, 3>(eskf::kPosIdx, eskf::kVelIdx) = Mat3d::Identity();

        // δv_dot = -R·[a×]·δθ - R·δba
        F.block<3, 3>(eskf::kVelIdx, eskf::kAttIdx) = -R * accel_skew;
        F.block<3, 3>(eskf::kVelIdx, eskf::kAbIdx)  = -R;

        // δθ_dot = -[ω×]·δθ - δbg
        // For phone-grade IMU, Earth rotation rate is negligible relative to noise
        // F.block<3,3>(eskf::kAttIdx, eskf::kAttIdx) = -skew(ω_ie_body) ≈ 0
        F.block<3, 3>(eskf::kAttIdx, eskf::kGbIdx) = -Mat3d::Identity();

        // Discrete transition: Φ = I + F·dt
        StateMat Phi = StateMat::Identity() + F * dt;

        // ── Process noise covariance Q ──
        StateMat Q = StateMat::Zero();

        // Acceleration noise → velocity noise
        Q.block<3, 3>(eskf::kVelIdx, eskf::kVelIdx) =
            R * (pn_.accel_noise * pn_.accel_noise * Mat3d::Identity()) * R.transpose() * dt;

        // Gyroscope noise → attitude noise
        Q.block<3, 3>(eskf::kAttIdx, eskf::kAttIdx) =
            pn_.gyro_noise * pn_.gyro_noise * Mat3d::Identity() * dt;

        // Bias random walks
        Q.block<3, 3>(eskf::kAbIdx, eskf::kAbIdx) =
            pn_.accel_bias_rw * pn_.accel_bias_rw * Mat3d::Identity() * dt;
        Q.block<3, 3>(eskf::kGbIdx, eskf::kGbIdx) =
            pn_.gyro_bias_rw * pn_.gyro_bias_rw * Mat3d::Identity() * dt;

        // ── Covariance propagation ──
        P_ = Phi * P_ * Phi.transpose() + Q;

        // Enforce symmetry (numerical stability)
        P_ = (P_ + P_.transpose()) * 0.5;
    }

    /// Generic measurement update.
    ///
    /// @param H  Measurement Jacobian (m × 15)
    /// @param z  Innovation (measurement - predicted measurement) (m × 1)
    /// @param R  Measurement noise covariance (m × m)
    ///
    /// @return true if update was applied, false if rejected (NIS test fail)
    template <int MeasDim>
    bool update(const Eigen::Matrix<double, MeasDim, eskf::kStateSize>& H,
                const Eigen::Matrix<double, MeasDim, 1>& z,
                const Eigen::Matrix<double, MeasDim, MeasDim>& R) {

        // Innovation covariance
        Eigen::Matrix<double, MeasDim, MeasDim> S =
            H * P_ * H.transpose() + R;

        // ── NIS test (chi-squared) ──
        Eigen::Matrix<double, MeasDim, 1> z_normalized = S.ldlt().solve(z);
        double nis = z.dot(z_normalized);

        // Chi-squared threshold at 99% confidence for MeasDim DOF
        // Approximate: threshold ≈ MeasDim + 2.3 * sqrt(2 * MeasDim)
        // More precise values from table:
        double chi2_threshold = chi2Threshold(MeasDim);
        if (nis > chi2_threshold) {
            spdlog::debug("EKF update rejected: NIS={:.2f} > threshold={:.2f} ({}D)",
                          nis, chi2_threshold, MeasDim);
            return false;
        }

        // Kalman gain
        // K = P * H^T * S^{-1}
        Eigen::Matrix<double, eskf::kStateSize, MeasDim> K =
            P_ * H.transpose() * S.inverse();

        // Error state update
        x_ += K * z;

        // Covariance update (Joseph form for numerical stability)
        StateMat I_KH = StateMat::Identity() - K * H;
        P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();

        // Enforce symmetry
        P_ = (P_ + P_.transpose()) * 0.5;

        return true;
    }

    /// GNSS position measurement update.
    ///
    /// @param position_enu  GNSS position in local ENU (meters)
    /// @param nominal_position  Current nominal INS position in ENU
    /// @param sigma_h  Horizontal position sigma (meters, 1σ)
    /// @param sigma_v  Vertical position sigma (meters, 1σ)
    bool updateGnssPosition(const Vec3d& position_enu,
                            const Vec3d& nominal_position,
                            double sigma_h, double sigma_v) {
        // H: δz = H · δx → position measurement observes position error directly
        // z = gnss_pos - (nominal_pos + δp) → z = (gnss_pos - nominal_pos) - δp
        // So: z = (gnss_pos - nominal_pos), H = [I₃ 0 0 0 0]
        Eigen::Matrix<double, 3, eskf::kStateSize> H =
            Eigen::Matrix<double, 3, eskf::kStateSize>::Zero();
        H.block<3, 3>(0, eskf::kPosIdx) = Mat3d::Identity();

        // Innovation
        Vec3d z = position_enu - nominal_position;

        // Measurement noise
        Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
        R(0, 0) = sigma_h * sigma_h;  // East
        R(1, 1) = sigma_h * sigma_h;  // North
        R(2, 2) = sigma_v * sigma_v;  // Up

        return update<3>(H, z, R);
    }

    /// GNSS velocity measurement update.
    ///
    /// @param velocity_enu  GNSS velocity in ENU (m/s)
    /// @param nominal_velocity  Current nominal INS velocity
    /// @param sigma_v  Velocity sigma (m/s, 1σ)
    bool updateGnssVelocity(const Vec3d& velocity_enu,
                            const Vec3d& nominal_velocity,
                            double sigma_v) {
        Eigen::Matrix<double, 3, eskf::kStateSize> H =
            Eigen::Matrix<double, 3, eskf::kStateSize>::Zero();
        H.block<3, 3>(0, eskf::kVelIdx) = Mat3d::Identity();

        Vec3d z = velocity_enu - nominal_velocity;

        Eigen::Matrix3d R = sigma_v * sigma_v * Eigen::Matrix3d::Identity();

        return update<3>(H, z, R);
    }

    /// Non-Holonomic Constraint (NHC) pseudo-measurement.
    /// Constrains lateral (y) and vertical (z) velocity in the vehicle body frame.
    ///
    /// @param nominal  Current nominal state
    /// @param sigma_lateral  Lateral velocity sigma (m/s)
    /// @param sigma_vertical  Vertical velocity sigma (m/s)
    bool updateNhc(const NominalState& nominal,
                   double sigma_lateral, double sigma_vertical) {
        // NHC: velocity in body frame should be [v_forward, ~0, ~0]
        // v_body = R^T * v_nav
        // We observe: v_body_y ≈ 0, v_body_z ≈ 0
        //
        // δv_body = R^T * δv_nav + [R^T * v_nav]× · δθ
        //
        // We only constrain y and z components:

        Mat3d R = nominal.attitude.toRotationMatrix();
        Mat3d RT = R.transpose();
        Vec3d v_body = RT * nominal.velocity;

        // Predicted body-frame velocity (lateral + vertical)
        Eigen::Matrix<double, 2, 1> z;
        z(0) = -v_body.y();  // innovation: 0 - v_body_y
        z(1) = -v_body.z();  // innovation: 0 - v_body_z

        // Jacobian
        // H_nhc = [0 R^T(row 1,2) [R^T*v_nav]×(row 1,2) 0 0]
        Mat3d v_nav_skew = quat::skewSymmetric(nominal.velocity);

        Eigen::Matrix<double, 2, eskf::kStateSize> H =
            Eigen::Matrix<double, 2, eskf::kStateSize>::Zero();
        H.block<2, 3>(0, eskf::kVelIdx) = RT.block<2, 3>(1, 0);
        H.block<2, 3>(0, eskf::kAttIdx) = (RT * v_nav_skew).block<2, 3>(1, 0);

        // Measurement noise
        Eigen::Matrix<double, 2, 2> R_nhc;
        R_nhc << sigma_lateral * sigma_lateral, 0,
                 0, sigma_vertical * sigma_vertical;

        return update<2>(H, z, R_nhc);
    }

    /// Barometric altitude measurement update.
    ///
    /// @param baro_alt_enu  Barometric altitude converted to ENU (meters)
    /// @param nominal_alt_enu  Current nominal INS altitude in ENU (meters)
    /// @param sigma_alt  Altitude 1σ measurement noise (meters)
    bool updateBaroAltitude(double baro_alt_enu,
                            double nominal_alt_enu,
                            double sigma_alt) {
        Eigen::Matrix<double, 1, eskf::kStateSize> H =
            Eigen::Matrix<double, 1, eskf::kStateSize>::Zero();
        H(0, eskf::kPosIdx + 2) = 1.0;

        Eigen::Matrix<double, 1, 1> z;
        z(0, 0) = baro_alt_enu - nominal_alt_enu;

        Eigen::Matrix<double, 1, 1> R;
        R(0, 0) = sigma_alt * sigma_alt;

        return update<1>(H, z, R);
    }

    /// Wheel speed / longitudinal odometry measurement update.
    ///
    /// @param forward_speed_mps  Measured forward vehicle speed (m/s)
    /// @param nominal  Current nominal state
    /// @param sigma_speed  Speed 1σ uncertainty (m/s)
    bool updateWheelSpeed(double forward_speed_mps,
                          const NominalState& nominal,
                          double sigma_speed) {
        Mat3d R = nominal.attitude.toRotationMatrix();
        Mat3d RT = R.transpose();
        Vec3d v_body = RT * nominal.velocity;

        // Innovation: forward speed measurement vs predicted body x velocity
        Eigen::Matrix<double, 1, 1> z;
        z(0, 0) = forward_speed_mps - v_body.x();

        // Jacobian: row 0 of body velocity mapping
        Mat3d v_nav_skew = quat::skewSymmetric(nominal.velocity);

        Eigen::Matrix<double, 1, eskf::kStateSize> H =
            Eigen::Matrix<double, 1, eskf::kStateSize>::Zero();
        H.block<1, 3>(0, eskf::kVelIdx) = RT.row(0);
        H.block<1, 3>(0, eskf::kAttIdx) = (RT * v_nav_skew).row(0);

        Eigen::Matrix<double, 1, 1> R_mat;
        R_mat(0, 0) = sigma_speed * sigma_speed;

        return update<1>(H, z, R_mat);
    }

    /// Calibrated magnetic heading measurement update.
    ///
    /// @param heading_rad  Measured magnetic heading (radians, ENU frame: yaw angle)
    /// @param nominal_yaw_rad  Current nominal yaw in radians
    /// @param sigma_yaw  Heading 1σ uncertainty (radians)
    bool updateMagneticHeading(double heading_rad,
                               double nominal_yaw_rad,
                               double sigma_yaw) {
        Eigen::Matrix<double, 1, eskf::kStateSize> H =
            Eigen::Matrix<double, 1, eskf::kStateSize>::Zero();
        H(0, eskf::kAttIdx + 2) = 1.0;

        Eigen::Matrix<double, 1, 1> z;
        z(0, 0) = angle::wrapPi(heading_rad - nominal_yaw_rad);

        Eigen::Matrix<double, 1, 1> R;
        R(0, 0) = sigma_yaw * sigma_yaw;

        return update<1>(H, z, R);
    }

    /// Apply the accumulated error state to the nominal state,
    /// then reset the error state to zero.
    void applyToNominal(StrapdownIns& ins) {
        Vec3d dp     = x_.segment<3>(eskf::kPosIdx);
        Vec3d dv     = x_.segment<3>(eskf::kVelIdx);
        Vec3d dtheta = x_.segment<3>(eskf::kAttIdx);
        Vec3d dba    = x_.segment<3>(eskf::kAbIdx);
        Vec3d dbg    = x_.segment<3>(eskf::kGbIdx);

        ins.applyCorrection(dp, dv, dtheta, dba, dbg);

        // Reset error state
        x_.setZero();

        // Reset covariance: apply the attitude reset Jacobian
        // G = I except for the attitude block where we account for the
        // error-state reset: δθ_new = δθ_old - dθ
        // For small dθ, the Jacobian is approximately identity.
        // For larger corrections, use: G_θ = I - [dθ/2]×
        // This is important for consistency after large corrections.
        StateMat G = StateMat::Identity();
        G.block<3, 3>(eskf::kAttIdx, eskf::kAttIdx) =
            Mat3d::Identity() - quat::skewSymmetric(dtheta * 0.5);

        P_ = G * P_ * G.transpose();
        P_ = (P_ + P_.transpose()) * 0.5;
    }

    // ── Accessors ──

    const StateVec& errorState() const { return x_; }
    const StateMat& covariance() const { return P_; }
    StateMat& mutableCovariance() { return P_; }

    /// Position uncertainty (1σ, meters) — sqrt of covariance diagonal
    Vec3d positionSigma() const {
        return P_.block<3, 3>(eskf::kPosIdx, eskf::kPosIdx).diagonal().cwiseSqrt();
    }

    /// Velocity uncertainty (1σ, m/s)
    Vec3d velocitySigma() const {
        return P_.block<3, 3>(eskf::kVelIdx, eskf::kVelIdx).diagonal().cwiseSqrt();
    }

    /// Attitude uncertainty (1σ, radians)
    Vec3d attitudeSigma() const {
        return P_.block<3, 3>(eskf::kAttIdx, eskf::kAttIdx).diagonal().cwiseSqrt();
    }

    /// Horizontal accuracy estimate (meters, 1σ, 2D)
    double horizontalAccuracy() const {
        auto ps = positionSigma();
        return std::sqrt(ps.x() * ps.x() + ps.y() * ps.y());
    }

    /// Set process noise parameters
    void setProcessNoise(const ProcessNoise& pn) { pn_ = pn; }

    /// Check covariance positive definiteness
    bool isCovarianceValid() const {
        Eigen::SelfAdjointEigenSolver<StateMat> solver(P_);
        return solver.eigenvalues().minCoeff() > 0.0;
    }

private:
    /// Chi-squared threshold at 99% for given degrees of freedom
    static double chi2Threshold(int dof) {
        // Precomputed for common dimensions
        static const double table[] = {
            0.0,     // 0 (unused)
            6.635,   // 1
            9.210,   // 2
            11.345,  // 3
            13.277,  // 4
            15.086,  // 5
            16.812   // 6
        };
        if (dof >= 1 && dof <= 6) return table[dof];
        // Approximation for larger dof
        return static_cast<double>(dof) + 2.3 * std::sqrt(2.0 * static_cast<double>(dof));
    }

    StateVec x_;     ///< Error state vector (15 × 1)
    StateMat P_;     ///< Error state covariance (15 × 15)
    ProcessNoise pn_;
};

}  // namespace idr
