// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Unit tests for ErrorStateEkf.

#include "idr/fusion/eskf.h"
#include "idr/inertial/strapdown_ins.h"
#include "idr/math_utils/geo_utils.h"
#include <gtest/gtest.h>
#include <cmath>

namespace idr {
namespace {

class EskfTest : public ::testing::Test {
protected:
    void SetUp() override {
        NominalState init;
        init.timestamp_ns = secToNs(0.0);
        init.position = Vec3d(0.0, 0.0, 0.0);
        init.velocity = Vec3d(10.0, 0.0, 0.0);  // 10 m/s East
        init.attitude = Quaterniond::Identity();
        ins_.initialize(init);
    }

    StrapdownIns ins_;
    ErrorStateEkf ekf_;
};

TEST_F(EskfTest, InitialCovariancePositiveDefinite) {
    EXPECT_TRUE(ekf_.isCovarianceValid());
}

TEST_F(EskfTest, PredictGrowsCovariance) {
    double initial_pos_var = ekf_.covariance()(0, 0);
    ekf_.predict(ins_.state(), Vec3d(0, 0, constants::kGravity), 0.01);
    double after_pos_var = ekf_.covariance()(0, 0);

    // Covariance should grow after prediction
    EXPECT_GT(after_pos_var, initial_pos_var);
}

TEST_F(EskfTest, GnssPositionUpdateReducesCovariance) {
    // Predict to grow uncertainty
    for (int i = 0; i < 100; i++) {
        ekf_.predict(ins_.state(), Vec3d(0, 0, constants::kGravity), 0.01);
    }

    double before_pos_var = ekf_.covariance()(0, 0);

    // GNSS update with good accuracy
    bool accepted = ekf_.updateGnssPosition(
        Vec3d(1.0, 0.0, 0.0),  // GNSS says we're at (1, 0, 0)
        ins_.state().position,    // INS says (0, 0, 0) [drifted]
        5.0,                      // 5m horizontal accuracy
        10.0                      // 10m vertical accuracy
    );

    EXPECT_TRUE(accepted);

    double after_pos_var = ekf_.covariance()(0, 0);
    EXPECT_LT(after_pos_var, before_pos_var);
}

TEST_F(EskfTest, GnssVelocityUpdateReducesVelocityCovariance) {
    for (int i = 0; i < 50; i++) {
        ekf_.predict(ins_.state(), Vec3d(0, 0, constants::kGravity), 0.01);
    }

    double before_vel_var = ekf_.covariance()(3, 3);  // velocity_x

    bool accepted = ekf_.updateGnssVelocity(
        Vec3d(10.0, 0.0, 0.0),
        ins_.state().velocity,
        0.5  // 0.5 m/s accuracy
    );

    EXPECT_TRUE(accepted);
    double after_vel_var = ekf_.covariance()(3, 3);
    EXPECT_LT(after_vel_var, before_vel_var);
}

TEST_F(EskfTest, NhcUpdate) {
    for (int i = 0; i < 50; i++) {
        ekf_.predict(ins_.state(), Vec3d(0, 0, constants::kGravity), 0.01);
    }

    bool accepted = ekf_.updateNhc(ins_.state(), 0.1, 0.1);
    EXPECT_TRUE(accepted);
}

TEST_F(EskfTest, ApplyToNominalResetsErrorState) {
    ekf_.predict(ins_.state(), Vec3d(0, 0, constants::kGravity), 0.01);

    // Simulate a GNSS update that creates a non-zero error state
    ekf_.updateGnssPosition(
        Vec3d(5.0, 0.0, 0.0),
        ins_.state().position,
        5.0, 10.0
    );

    // Error state should be non-zero before apply
    EXPECT_GT(ekf_.errorState().norm(), 0.0);

    ekf_.applyToNominal(ins_);

    // Error state should be zero after apply
    EXPECT_NEAR(ekf_.errorState().norm(), 0.0, 1e-15);
}

TEST_F(EskfTest, LargeInnovationRejected) {
    // Very small covariance → large innovations should be rejected
    auto& P = ekf_.mutableCovariance();
    P.setIdentity();
    P *= 0.0001;  // Very confident

    bool accepted = ekf_.updateGnssPosition(
        Vec3d(1000.0, 0.0, 0.0),  // 1000m away — huge innovation
        ins_.state().position,
        5.0, 10.0
    );

    EXPECT_FALSE(accepted);  // Should be rejected by NIS test
}

TEST_F(EskfTest, CovarianceStaysSymmetric) {
    for (int i = 0; i < 100; i++) {
        ekf_.predict(ins_.state(), Vec3d(0, 0, constants::kGravity), 0.01);
        if (i % 10 == 0) {
            ekf_.updateGnssPosition(
                ins_.state().position + Vec3d(0.1, 0.0, 0.0),
                ins_.state().position, 5.0, 10.0
            );
        }
    }

    auto P = ekf_.covariance();
    double asymmetry = (P - P.transpose()).norm();
    EXPECT_LT(asymmetry, 1e-10);
}

TEST_F(EskfTest, CovarianceStaysPositiveDefinite) {
    for (int i = 0; i < 200; i++) {
        ekf_.predict(ins_.state(), Vec3d(0, 0, constants::kGravity), 0.01);
        if (i % 5 == 0) {
            ekf_.updateGnssPosition(
                ins_.state().position + Vec3d(0.5, 0.3, 0.0),
                ins_.state().position, 5.0, 10.0
            );
            ekf_.updateNhc(ins_.state(), 0.1, 0.1);
            ekf_.applyToNominal(ins_);
        }
    }

    EXPECT_TRUE(ekf_.isCovarianceValid());
}

}  // namespace
}  // namespace idr
