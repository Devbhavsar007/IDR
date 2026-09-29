// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Right-Invariant Extended Kalman Filter (RI-EKF) on the matrix Lie group SE_2(3).
//
// State space:
//   X in SE_2(3) (Rotation R in SO(3), Velocity v in R^3, Position p in R^3)
//   biases in R^6 (Accelerometer bias b_a, Gyroscope bias b_g)
//
// Key Advantage over Standard ES-EKF:
//   The error dynamics matrix A is independent of the estimated trajectory!
//   The gravity cross-product [g x] depends only on the constant gravity vector
//   in the navigation frame, preventing covariance collapse and false overconfidence
//   during aggressive yaw maneuvers or extended GNSS blackouts.
//
// References:
//   - Barrau & Bonnabel, "The Invariant Extended Kalman Filter as a Stable
//     Observer for Star Autonomous Navigation", IEEE TAC 2017.
//   - Brossard et al., "Associating Uncertainty With Three-Dimensional Poses for
//     Diversity in Robot Localization", IEEE RA-L 2021.

#pragma once

#include "idr/fusion/eskf.h"
#include "idr/inertial/strapdown_ins.h"
#include "idr/math_utils/quat_utils.h"
#include "idr/types/common.h"

#include <spdlog/spdlog.h>
#include <Eigen/Dense>
#include <cmath>

namespace idr {
namespace iekf {

constexpr int kStateDim = 15;
constexpr int kThetaIdx = 0;   // Attitude error (3)
constexpr int kVelIdx   = 3;   // Velocity error (3)
constexpr int kPosIdx   = 6;   // Position error (3)
constexpr int kAbIdx    = 9;   // Accel bias error (3)
constexpr int kGbIdx    = 12;  // Gyro bias error (3)

}  // namespace iekf

class InvariantEkf {
public:
    using StateVec = Eigen::Matrix<double, iekf::kStateDim, 1>;
    using StateMat = Eigen::Matrix<double, iekf::kStateDim, iekf::kStateDim>;

    explicit InvariantEkf(const ProcessNoise& pn = {})
        : pn_(pn) {
        reset();
    }

    void reset() {
        R_ = Mat3d::Identity();
        v_ = Vec3d::Zero();
        p_ = Vec3d::Zero();
        ba_ = Vec3d::Zero();
        bg_ = Vec3d::Zero();

        P_.setZero();
        // Initial 1-sigma uncertainties
        P_.block<3, 3>(iekf::kThetaIdx, iekf::kThetaIdx) = Mat3d::Identity() * (0.05 * 0.05);   // ~3 deg
        P_.block<3, 3>(iekf::kVelIdx,   iekf::kVelIdx)   = Mat3d::Identity() * (0.5 * 0.5);     // 0.5 m/s
        P_.block<3, 3>(iekf::kPosIdx,   iekf::kPosIdx)   = Mat3d::Identity() * (5.0 * 5.0);     // 5.0 m
        P_.block<3, 3>(iekf::kAbIdx,    iekf::kAbIdx)    = Mat3d::Identity() * (0.05 * 0.05);   // 0.05 m/s²
        P_.block<3, 3>(iekf::kGbIdx,    iekf::kGbIdx)    = Mat3d::Identity() * (0.005 * 0.005); // 0.005 rad/s
    }

    void initialize(const Vec3d& pos_enu, const Vec3d& vel_enu, const Mat3d& R,
                    double pos_sigma = 5.0, double vel_sigma = 2.0, double att_sigma = 0.2) {
        p_ = pos_enu;
        v_ = vel_enu;
        R_ = R;
        P_.block<3, 3>(iekf::kThetaIdx, iekf::kThetaIdx) = Mat3d::Identity() * (att_sigma * att_sigma);
        P_.block<3, 3>(iekf::kVelIdx,   iekf::kVelIdx)   = Mat3d::Identity() * (vel_sigma * vel_sigma);
        P_.block<3, 3>(iekf::kPosIdx,   iekf::kPosIdx)   = Mat3d::Identity() * (pos_sigma * pos_sigma);
    }

    /// Predict state and covariance using IMU measurement.
    void predict(const Vec3d& accel_meas, const Vec3d& gyro_meas, double dt) {
        if (dt <= 0.0 || dt > 1.0) return;

        // Correct raw measurements with estimated biases
        Vec3d a_corr = accel_meas - ba_;
        Vec3d w_corr = gyro_meas - bg_;

        // 1. Nominal integration on SE_2(3)
        // Gravity in navigation frame (ENU: z-up)
        const Vec3d g(0.0, 0.0, -constants::kGravity);

        // Rotation integration via Rodrigues' formula
        double angle = w_corr.norm() * dt;
        Mat3d dR = Mat3d::Identity();
        if (angle > 1e-8) {
            Vec3d axis = w_corr.normalized();
            dR = Eigen::AngleAxisd(angle, axis).toRotationMatrix();
        }

        // Acceleration in navigation frame
        Vec3d acc_nav = R_ * a_corr + g;

        // Position and velocity propagation
        p_ += v_ * dt + 0.5 * acc_nav * dt * dt;
        v_ += acc_nav * dt;
        R_ = R_ * dR;
        // Orthogonalize rotation matrix
        Eigen::JacobiSVD<Mat3d> svd(R_, Eigen::ComputeFullU | Eigen::ComputeFullV);
        R_ = svd.matrixU() * svd.matrixV().transpose();

        // 2. Continuous-time error dynamics matrix A (independent of trajectory state!)
        Eigen::Matrix<double, iekf::kStateDim, iekf::kStateDim> A =
            Eigen::Matrix<double, iekf::kStateDim, iekf::kStateDim>::Zero();

        // [g x] is constant gravity skew-symmetric matrix!
        Mat3d g_skew = quat::skewSymmetric(g);

        A.block<3, 3>(iekf::kVelIdx, iekf::kThetaIdx) = g_skew;
        A.block<3, 3>(iekf::kPosIdx, iekf::kVelIdx)   = Mat3d::Identity();

        // Biases coupling into right-invariant error:
        A.block<3, 3>(iekf::kThetaIdx, iekf::kGbIdx) = -R_;
        A.block<3, 3>(iekf::kVelIdx,   iekf::kAbIdx) = -R_;

        // First-order state transition matrix Phi = I + A * dt
        StateMat Phi = StateMat::Identity() + A * dt;

        // Discrete process noise Q_d
        StateMat Qd = StateMat::Zero();
        double dt2 = dt * dt;
        Qd.block<3, 3>(iekf::kThetaIdx, iekf::kThetaIdx) = Mat3d::Identity() * (pn_.gyro_noise * pn_.gyro_noise * dt);
        Qd.block<3, 3>(iekf::kVelIdx,   iekf::kVelIdx)   = Mat3d::Identity() * (pn_.accel_noise * pn_.accel_noise * dt);
        Qd.block<3, 3>(iekf::kPosIdx,   iekf::kPosIdx)   = Mat3d::Identity() * (pn_.accel_noise * pn_.accel_noise * dt2 * 0.25);
        Qd.block<3, 3>(iekf::kAbIdx,    iekf::kAbIdx)    = Mat3d::Identity() * (pn_.accel_bias_rw * pn_.accel_bias_rw * dt);
        Qd.block<3, 3>(iekf::kGbIdx,    iekf::kGbIdx)    = Mat3d::Identity() * (pn_.gyro_bias_rw * pn_.gyro_bias_rw * dt);

        // Covariance propagation
        P_ = Phi * P_ * Phi.transpose() + Qd;
        P_ = (P_ + P_.transpose()) * 0.5;
    }

    /// Generic measurement update helper.
    template <int MeasDim>
    bool update(const Eigen::Matrix<double, MeasDim, iekf::kStateDim>& H,
                const Eigen::Matrix<double, MeasDim, 1>& z,
                const Eigen::Matrix<double, MeasDim, MeasDim>& R) {
        Eigen::Matrix<double, MeasDim, MeasDim> S = H * P_ * H.transpose() + R;

        // Innovation Mahalanobis / NIS test
        Eigen::Matrix<double, MeasDim, 1> z_norm = S.ldlt().solve(z);
        double nis = z.dot(z_norm);
        double threshold = chi2Threshold(MeasDim);
        if (nis > threshold) {
            spdlog::debug("RI-EKF update rejected: NIS={:.2f} > threshold={:.2f}", nis, threshold);
            return false;
        }

        // Kalman gain
        Eigen::Matrix<double, iekf::kStateDim, MeasDim> K = P_ * H.transpose() * S.inverse();

        // Right-invariant Lie algebra correction xi
        StateVec xi = K * z;

        // Inject correction onto SE_2(3) group
        Vec3d xi_theta = xi.segment<3>(iekf::kThetaIdx);
        Vec3d xi_v     = xi.segment<3>(iekf::kVelIdx);
        Vec3d xi_p     = xi.segment<3>(iekf::kPosIdx);
        Vec3d dba      = xi.segment<3>(iekf::kAbIdx);
        Vec3d dbg      = xi.segment<3>(iekf::kGbIdx);

        // Exponential map for SO(3)
        double theta_norm = xi_theta.norm();
        Mat3d exp_theta = Mat3d::Identity();
        if (theta_norm > 1e-8) {
            exp_theta = Eigen::AngleAxisd(theta_norm, xi_theta.normalized()).toRotationMatrix();
        }
        R_ = exp_theta * R_;

        // Group action on SE_2(3): exp(xi) * X rotates and shifts velocity and position
        v_ = exp_theta * v_ + xi_v;
        p_ = exp_theta * p_ + xi_p;
        ba_ += dba;
        bg_ += dbg;


        // Joseph form covariance update
        StateMat I_KH = StateMat::Identity() - K * H;
        P_ = I_KH * P_ * I_KH.transpose() + K * R * K.transpose();
        P_ = (P_ + P_.transpose()) * 0.5;

        return true;
    }

    /// GNSS position update.
    bool updateGnssPosition(const Vec3d& pos_enu, double sigma_h, double sigma_v) {
        Eigen::Matrix<double, 3, iekf::kStateDim> H = Eigen::Matrix<double, 3, iekf::kStateDim>::Zero();
        H.block<3, 3>(0, iekf::kPosIdx) = Mat3d::Identity();

        Vec3d z = pos_enu - p_;

        Mat3d R_meas = Mat3d::Zero();
        R_meas(0, 0) = sigma_h * sigma_h;
        R_meas(1, 1) = sigma_h * sigma_h;
        R_meas(2, 2) = sigma_v * sigma_v;

        return update<3>(H, z, R_meas);
    }

    /// GNSS velocity update with Right-Invariant Lie algebra mapping.
    bool updateGnssVelocity(const Vec3d& vel_enu, double sigma_vel) {
        Eigen::Matrix<double, 3, iekf::kStateDim> H = Eigen::Matrix<double, 3, iekf::kStateDim>::Zero();
        H.block<3, 3>(0, iekf::kVelIdx) = Mat3d::Identity();
        // Coupling between attitude error and velocity innovation in SE_2(3)
        H.block<3, 3>(0, iekf::kThetaIdx) = -quat::skewSymmetric(v_);

        Vec3d z = vel_enu - v_;
        Mat3d R_meas = Mat3d::Identity() * (sigma_vel * sigma_vel);

        return update<3>(H, z, R_meas);
    }

    /// Non-Holonomic Constraints (NHC) on body lateral and vertical velocity.
    bool updateNhc(double sigma_lateral, double sigma_vertical) {
        Mat3d RT = R_.transpose();
        Vec3d v_body = RT * v_;

        // Body velocity should have zero y and zero z components
        Eigen::Matrix<double, 2, 1> z;
        z(0) = -v_body.y();
        z(1) = -v_body.z();

        Eigen::Matrix<double, 2, iekf::kStateDim> H = Eigen::Matrix<double, 2, iekf::kStateDim>::Zero();
        H.block<2, 3>(0, iekf::kVelIdx) = RT.block<2, 3>(1, 0);

        Eigen::Matrix<double, 2, 2> R_nhc;
        R_nhc << sigma_lateral * sigma_lateral, 0,
                 0, sigma_vertical * sigma_vertical;

        return update<2>(H, z, R_nhc);
    }

    // ── Accessors ──
    const Mat3d& rotation() const { return R_; }
    const Vec3d& velocity() const { return v_; }
    const Vec3d& position() const { return p_; }
    const Vec3d& accelBias() const { return ba_; }
    const Vec3d& gyroBias() const { return bg_; }
    const StateMat& covariance() const { return P_; }

    double horizontalAccuracy() const {
        return std::sqrt(P_(iekf::kPosIdx, iekf::kPosIdx) + P_(iekf::kPosIdx + 1, iekf::kPosIdx + 1));
    }

    Vec3d positionSigma() const {
        return P_.block<3, 3>(iekf::kPosIdx, iekf::kPosIdx).diagonal().cwiseSqrt();
    }

    Vec3d velocitySigma() const {
        return P_.block<3, 3>(iekf::kVelIdx, iekf::kVelIdx).diagonal().cwiseSqrt();
    }

    Vec3d attitudeSigma() const {
        return P_.block<3, 3>(iekf::kThetaIdx, iekf::kThetaIdx).diagonal().cwiseSqrt();
    }

private:
    static double chi2Threshold(int dof) {
        static const double table[] = { 0.0, 6.635, 9.210, 11.345, 13.277, 15.086, 16.812 };
        if (dof >= 1 && dof <= 6) return table[dof];
        return static_cast<double>(dof) + 2.3 * std::sqrt(2.0 * static_cast<double>(dof));
    }

    // Nominal State on Lie Group SE_2(3) x R^6
    Mat3d R_;   ///< Orientation in SO(3)
    Vec3d v_;   ///< Velocity in ENU (m/s)
    Vec3d p_;   ///< Position in ENU (m)
    Vec3d ba_;  ///< Accelerometer bias (m/s²)
    Vec3d bg_;  ///< Gyroscope bias (rad/s)

    StateMat P_;      ///< Error state covariance (15 x 15)
    ProcessNoise pn_; ///< Process noise parameters
};

}  // namespace idr
