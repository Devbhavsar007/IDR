// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Quaternion utilities and angle wrappers.

#pragma once

#include "idr/types/common.h"
#include <cmath>

namespace idr {
namespace quat {

/// Create a quaternion from Euler angles (ZYX intrinsic: yaw, pitch, roll).
/// All angles in radians.
inline Quaterniond fromEulerZYX(double yaw, double pitch, double roll) {
    return Quaterniond(
        Eigen::AngleAxisd(yaw,   Vec3d::UnitZ()) *
        Eigen::AngleAxisd(pitch, Vec3d::UnitY()) *
        Eigen::AngleAxisd(roll,  Vec3d::UnitX())
    );
}

/// Extract Euler angles (yaw, pitch, roll) from quaternion.
/// Returns [yaw, pitch, roll] in radians.
inline Vec3d toEulerZYX(const Quaterniond& q) {
    // Convert to rotation matrix, then extract ZYX Euler
    Mat3d R = q.normalized().toRotationMatrix();

    double pitch = -std::asin(std::clamp(R(2, 0), -1.0, 1.0));
    double yaw, roll;

    if (std::abs(R(2, 0)) < 0.999) {
        yaw  = std::atan2(R(1, 0), R(0, 0));
        roll = std::atan2(R(2, 1), R(2, 2));
    } else {
        // Gimbal lock
        yaw  = std::atan2(-R(0, 1), R(1, 1));
        roll = 0.0;
    }

    return {yaw, pitch, roll};
}

/// Small-angle quaternion from rotation vector (axis-angle where |ω| = angle).
/// Used for ES-EKF error state → quaternion update.
/// For small δθ: q ≈ [1, δθ/2]
inline Quaterniond fromRotationVector(const Vec3d& rv) {
    double half_angle = rv.norm() / 2.0;
    if (half_angle < 1e-12) {
        // First-order approximation
        return Quaterniond(1.0, rv.x() / 2.0, rv.y() / 2.0, rv.z() / 2.0).normalized();
    }
    Vec3d axis = rv.normalized();
    double angle = rv.norm();
    return Quaterniond(Eigen::AngleAxisd(angle, axis));
}

/// Convert quaternion to rotation vector (axis-angle).
inline Vec3d toRotationVector(const Quaterniond& q) {
    Eigen::AngleAxisd aa(q.normalized());
    return aa.axis() * aa.angle();
}

/// Quaternion left-multiplication matrix (for q ⊗ p as matrix * p_vec4).
/// Used in EKF Jacobian derivations.
inline Mat4d leftMultiplicationMatrix(const Quaterniond& q) {
    Mat4d L;
    L << q.w(), -q.x(), -q.y(), -q.z(),
         q.x(),  q.w(), -q.z(),  q.y(),
         q.y(),  q.z(),  q.w(), -q.x(),
         q.z(), -q.y(),  q.x(),  q.w();
    return L;
}

/// Skew-symmetric matrix from 3D vector (cross-product matrix).
/// [v]× such that [v]× * u = v × u
inline Mat3d skewSymmetric(const Vec3d& v) {
    Mat3d S;
    S <<     0, -v.z(),  v.y(),
          v.z(),     0, -v.x(),
         -v.y(),  v.x(),     0;
    return S;
}

}  // namespace quat

namespace angle {

/// Wrap angle to [-π, π]
inline double wrapPi(double angle_rad) {
    while (angle_rad > constants::kPi) angle_rad -= 2.0 * constants::kPi;
    while (angle_rad < -constants::kPi) angle_rad += 2.0 * constants::kPi;
    return angle_rad;
}

/// Wrap angle to [0, 2π)
inline double wrap2Pi(double angle_rad) {
    while (angle_rad >= 2.0 * constants::kPi) angle_rad -= 2.0 * constants::kPi;
    while (angle_rad < 0.0) angle_rad += 2.0 * constants::kPi;
    return angle_rad;
}

/// Signed angular difference (a - b), result in [-π, π]
inline double angleDifference(double a_rad, double b_rad) {
    return wrapPi(a_rad - b_rad);
}

}  // namespace angle
}  // namespace idr
