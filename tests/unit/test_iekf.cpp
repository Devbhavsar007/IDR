// IDR — Unit Tests for Right-Invariant EKF on SE_2(3)
// Copyright (c) 2026 IDR Project. All rights reserved.

#include "idr/fusion/iekf.h"
#include <gtest/gtest.h>

namespace idr {
namespace {

TEST(InvariantEkfTest, InitialCovariancePositiveDefinite) {
    InvariantEkf iekf;
    Eigen::SelfAdjointEigenSolver<InvariantEkf::StateMat> solver(iekf.covariance());
    EXPECT_GT(solver.eigenvalues().minCoeff(), 0.0);
}

TEST(InvariantEkfTest, PredictionPropagatesUncertainty) {
    InvariantEkf iekf;
    double initial_p_xx = iekf.covariance()(iekf::kPosIdx, iekf::kPosIdx);

    Vec3d accel(0.0, 0.0, constants::kGravity); // stationary on ground (1g up)
    Vec3d gyro(0.0, 0.0, 0.0);

    for (int i = 0; i < 50; ++i) {
        iekf.predict(accel, gyro, 0.01);
    }

    double final_p_xx = iekf.covariance()(iekf::kPosIdx, iekf::kPosIdx);
    EXPECT_GT(final_p_xx, initial_p_xx);
}

TEST(InvariantEkfTest, GnssPositionUpdateReducesCovariance) {
    InvariantEkf iekf;
    iekf.initialize(Vec3d(0, 0, 0), Vec3d(10, 0, 0), Mat3d::Identity());

    // Predict to grow covariance
    for (int i = 0; i < 20; ++i) {
        iekf.predict(Vec3d(0, 0, constants::kGravity), Vec3d::Zero(), 0.01);
    }
    double pre_p = iekf.covariance()(iekf::kPosIdx, iekf::kPosIdx);

    // Apply GNSS position update
    bool accepted = iekf.updateGnssPosition(Vec3d(2.0, 0.0, 0.0), 1.5, 2.5);
    EXPECT_TRUE(accepted);

    double post_p = iekf.covariance()(iekf::kPosIdx, iekf::kPosIdx);
    EXPECT_LT(post_p, pre_p);
}

TEST(InvariantEkfTest, TrajectoryIndependentDynamicsStability) {
    InvariantEkf iekf_stationary;
    InvariantEkf iekf_moving;

    iekf_stationary.initialize(Vec3d(0, 0, 0), Vec3d(0, 0, 0), Mat3d::Identity());
    iekf_moving.initialize(Vec3d(500, -200, 15), Vec3d(25, 10, 0), Mat3d::Identity());

    // Both should maintain stable, positive-definite covariance under identical IMU input
    Vec3d accel(0.5, 0.0, constants::kGravity);
    Vec3d gyro(0.0, 0.0, 0.05);

    for (int i = 0; i < 100; ++i) {
        iekf_stationary.predict(accel, gyro, 0.01);
        iekf_moving.predict(accel, gyro, 0.01);
    }

    Eigen::SelfAdjointEigenSolver<InvariantEkf::StateMat> solver_stat(iekf_stationary.covariance());
    Eigen::SelfAdjointEigenSolver<InvariantEkf::StateMat> solver_move(iekf_moving.covariance());

    EXPECT_GT(solver_stat.eigenvalues().minCoeff(), 0.0);
    EXPECT_GT(solver_move.eigenvalues().minCoeff(), 0.0);
}

TEST(InvariantEkfTest, RightInvariantNhcReducesLateralVelocityError) {
    InvariantEkf iekf;
    // Set nominal moving with slight lateral slip
    iekf.initialize(Vec3d(0, 0, 0), Vec3d(15.0, 1.2, 0.0), Mat3d::Identity());

    double initial_p_vy = iekf.covariance()(iekf::kVelIdx + 1, iekf::kVelIdx + 1);

    bool accepted = iekf.updateNhc(0.15, 0.20);
    EXPECT_TRUE(accepted);

    double updated_p_vy = iekf.covariance()(iekf::kVelIdx + 1, iekf::kVelIdx + 1);
    EXPECT_LT(updated_p_vy, initial_p_vy);

    // Lateral velocity should be pulled closer to 0
    EXPECT_LT(std::abs(iekf.velocity().y()), 1.2);
}

TEST(InvariantEkfTest, HeadingConvergenceWithGnssVelocity) {
    InvariantEkf iekf;
    // True velocity is East [10, 0, 0], but filter initialized with 30-degree heading offset
    double initial_heading_offset = 30.0 * constants::kDegToRad;
    Mat3d R_init = Eigen::AngleAxisd(initial_heading_offset, Vec3d::UnitZ()).toRotationMatrix();
    iekf.initialize(Vec3d(0, 0, 0),
                    Vec3d(10.0 * std::cos(initial_heading_offset), 10.0 * std::sin(initial_heading_offset), 0.0),
                    R_init, 5.0, 3.0, 0.6);


    // Apply GNSS velocity fixes pointing due East [10, 0, 0] for 2 seconds
    for (int i = 0; i < 20; ++i) {
        iekf.predict(Vec3d(0, 0, constants::kGravity), Vec3d::Zero(), 0.1);
        iekf.updateGnssVelocity(Vec3d(10.0, 0.0, 0.0), 0.2);
    }


    // Velocity should align with East
    EXPECT_GT(iekf.velocity().x(), 8.0);
    EXPECT_LT(std::abs(iekf.velocity().y()), 2.0);
}

}  // namespace
}  // namespace idr
