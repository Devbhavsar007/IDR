// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Unit tests for quaternion utilities.

#include "idr/math_utils/quat_utils.h"
#include <gtest/gtest.h>
#include <cmath>

namespace idr {
namespace quat {
namespace {

constexpr double kEps = 1e-10;

TEST(QuatUtilsTest, IdentityEuler) {
    auto q = fromEulerZYX(0.0, 0.0, 0.0);
    EXPECT_NEAR(q.w(), 1.0, kEps);
    EXPECT_NEAR(q.x(), 0.0, kEps);
    EXPECT_NEAR(q.y(), 0.0, kEps);
    EXPECT_NEAR(q.z(), 0.0, kEps);
}

TEST(QuatUtilsTest, Yaw90) {
    auto q = fromEulerZYX(constants::kPi / 2, 0.0, 0.0);
    Vec3d euler = toEulerZYX(q);
    EXPECT_NEAR(euler[0], constants::kPi / 2, 1e-6);  // yaw
    EXPECT_NEAR(euler[1], 0.0, 1e-6);        // pitch
    EXPECT_NEAR(euler[2], 0.0, 1e-6);        // roll
}

TEST(QuatUtilsTest, RoundTripEuler) {
    double yaw = 0.5, pitch = 0.2, roll = -0.3;
    auto q = fromEulerZYX(yaw, pitch, roll);
    Vec3d euler = toEulerZYX(q);
    EXPECT_NEAR(euler[0], yaw, 1e-10);
    EXPECT_NEAR(euler[1], pitch, 1e-10);
    EXPECT_NEAR(euler[2], roll, 1e-10);
}

TEST(QuatUtilsTest, RotationVectorSmall) {
    Vec3d rv(0.001, 0.002, 0.003);
    auto q = fromRotationVector(rv);

    // Should be very close to identity
    EXPECT_GT(q.w(), 0.999);

    // Round trip
    Vec3d rv_back = toRotationVector(q);
    EXPECT_NEAR(rv_back.x(), rv.x(), 1e-8);
    EXPECT_NEAR(rv_back.y(), rv.y(), 1e-8);
    EXPECT_NEAR(rv_back.z(), rv.z(), 1e-8);
}

TEST(QuatUtilsTest, RotationVectorLarge) {
    Vec3d rv(0.0, 0.0, constants::kPi / 4);  // 45° yaw
    auto q = fromRotationVector(rv);
    Vec3d rv_back = toRotationVector(q);
    EXPECT_NEAR(rv_back.z(), constants::kPi / 4, 1e-10);
}

TEST(QuatUtilsTest, SkewSymmetric) {
    Vec3d v(1.0, 2.0, 3.0);
    Mat3d S = skewSymmetric(v);

    // S should be antisymmetric
    EXPECT_NEAR((S + S.transpose()).norm(), 0.0, kEps);

    // S*u = v × u for arbitrary u
    Vec3d u(4.0, 5.0, 6.0);
    Vec3d cross = v.cross(u);
    Vec3d result = S * u;
    EXPECT_NEAR((cross - result).norm(), 0.0, kEps);
}

}  // namespace
}  // namespace quat
}  // namespace idr
