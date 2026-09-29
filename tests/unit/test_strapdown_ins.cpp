// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Unit tests for StrapdownIns.

#include "idr/inertial/strapdown_ins.h"
#include "idr/math_utils/quat_utils.h"
#include <gtest/gtest.h>
#include <cmath>

namespace idr {
namespace {

/// Create an ImuSample at the given time with specified accel/gyro.
ImuSample makeImu(double time_sec, const Vec3d& accel, const Vec3d& gyro) {
    ImuSample s;
    s.timestamp_ns = secToNs(time_sec);
    s.accel = accel;
    s.gyro = gyro;
    return s;
}

TEST(StrapdownInsTest, InitializationState) {
    StrapdownIns ins;
    EXPECT_FALSE(ins.isInitialized());

    NominalState init;
    init.timestamp_ns = secToNs(0.0);
    ins.initialize(init);

    EXPECT_TRUE(ins.isInitialized());
    EXPECT_NEAR(ins.state().position.norm(), 0.0, 1e-12);
    EXPECT_NEAR(ins.state().velocity.norm(), 0.0, 1e-12);
}

TEST(StrapdownInsTest, StationaryOnGround) {
    // When stationary, accelerometer reads [0, 0, +g] in body frame
    // (if body frame z is up). After removing gravity, acceleration should be ~0.
    StrapdownIns ins;
    NominalState init;
    init.timestamp_ns = secToNs(0.0);
    init.attitude = Quaterniond::Identity();  // body = nav (ENU)
    ins.initialize(init);

    // Stationary: accel = [0, 0, +g] in body (measuring gravity)
    // After INS gravity removal: a_nav = R*a_body + g_nav
    //   = I*[0,0,9.81] + [0,0,-9.81] = [0,0,0]
    double dt = 0.01;
    for (int i = 1; i <= 100; i++) {  // 1 second
        Vec3d accel(0.0, 0.0, constants::kGravity);  // Measuring +g
        Vec3d gyro(0.0, 0.0, 0.0);
        auto imu = makeImu(i * dt, accel, gyro);
        ins.propagate(imu, dt);
    }

    // Should remain near origin with near-zero velocity
    EXPECT_NEAR(ins.state().position.norm(), 0.0, 0.01);
    EXPECT_NEAR(ins.state().velocity.norm(), 0.0, 0.01);
}

TEST(StrapdownInsTest, ConstantVelocityForward) {
    // Moving at constant velocity with no rotation.
    // Accel should read only gravity (no linear acceleration).
    StrapdownIns ins;
    NominalState init;
    init.timestamp_ns = secToNs(0.0);
    init.velocity = Vec3d(10.0, 0.0, 0.0);  // 10 m/s East
    init.attitude = Quaterniond::Identity();
    ins.initialize(init);

    double dt = 0.01;
    for (int i = 1; i <= 100; i++) {
        Vec3d accel(0.0, 0.0, constants::kGravity);  // Only gravity
        Vec3d gyro(0.0, 0.0, 0.0);
        auto imu = makeImu(i * dt, accel, gyro);
        ins.propagate(imu, dt);
    }

    // After 1 second at 10 m/s East:
    EXPECT_NEAR(ins.state().position.x(), 10.0, 0.1);   // 10m East
    EXPECT_NEAR(ins.state().position.y(), 0.0, 0.01);
    EXPECT_NEAR(ins.state().velocity.x(), 10.0, 0.01);
    EXPECT_NEAR(ins.state().velocity.y(), 0.0, 0.01);
}

TEST(StrapdownInsTest, AcceleratingForward) {
    // Start from rest, apply 2 m/s² forward acceleration for 1 second.
    // Expected: v = 2 m/s, p = 1 m
    StrapdownIns ins;
    NominalState init;
    init.timestamp_ns = secToNs(0.0);
    init.attitude = Quaterniond::Identity();
    ins.initialize(init);

    double dt = 0.01;
    double accel_fwd = 2.0;  // m/s² in East (x) direction
    for (int i = 1; i <= 100; i++) {
        // Body frame accel = forward accel + gravity
        Vec3d accel(accel_fwd, 0.0, constants::kGravity);
        Vec3d gyro(0.0, 0.0, 0.0);
        auto imu = makeImu(i * dt, accel, gyro);
        ins.propagate(imu, dt);
    }

    // After 1 second: v = at = 2 m/s, p = ½at² = 1 m
    EXPECT_NEAR(ins.state().velocity.x(), 2.0, 0.05);
    EXPECT_NEAR(ins.state().position.x(), 1.0, 0.05);
}

TEST(StrapdownInsTest, CorrectionApplied) {
    StrapdownIns ins;
    NominalState init;
    init.timestamp_ns = secToNs(0.0);
    init.position = Vec3d(100.0, 200.0, 0.0);
    ins.initialize(init);

    // Apply a position correction
    Vec3d dp(5.0, -3.0, 0.0);
    ins.applyCorrection(dp, Vec3d::Zero(), Vec3d::Zero(),
                        Vec3d::Zero(), Vec3d::Zero());

    EXPECT_NEAR(ins.state().position.x(), 105.0, 1e-10);
    EXPECT_NEAR(ins.state().position.y(), 197.0, 1e-10);
}

}  // namespace
}  // namespace idr
