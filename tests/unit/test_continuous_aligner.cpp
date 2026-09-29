// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Unit tests for Phone-to-Vehicle Continuous Alignment and Re-calibration

#include <gtest/gtest.h>
#include "idr/alignment/phone_alignment.h"

using namespace idr;

TEST(ContinuousAlignerTest, InitialGravityConvergence) {
    PhoneToVehicleAlignment align;
    EXPECT_EQ(align.quality(), AlignmentQuality::NONE);

    // Feed stationary flat gravity
    Vec3d g(0.0, 0.0, 9.81);
    for (int i = 0; i < 50; i++) {
        align.updateGravity(g);
    }

    EXPECT_EQ(align.quality(), AlignmentQuality::GRAVITY);
    EXPECT_NEAR(align.pitchRad(), 0.0, 0.01);
    EXPECT_NEAR(align.rollRad(), 0.0, 0.01);
}

TEST(ContinuousAlignerTest, HeadingConvergenceUnderMotion) {
    PhoneToVehicleAlignment align;
    Vec3d g(0.0, 0.0, 9.81);
    for (int i = 0; i < 50; i++) {
        align.updateGravity(g);
    }

    // Vehicle driving East (yaw = pi/2 rad) at 12 m/s
    double heading_east = constants::kPi / 2.0;
    for (int i = 0; i < 30; i++) {
        align.updateHeading(heading_east, 12.0);
    }

    EXPECT_EQ(align.quality(), AlignmentQuality::CONVERGED);
    EXPECT_NEAR(align.yawRad(), heading_east, 0.05);
    EXPECT_GE(align.confidence(), 0.5);
}

TEST(ContinuousAlignerTest, ContinuousDynamicUpdateTracksGravity) {
    PhoneToVehicleAlignment align;
    Vec3d g(0.0, 0.0, 9.81);
    for (int i = 0; i < 50; i++) {
        align.updateGravity(g);
    }

    double heading_east = constants::kPi / 2.0;
    for (int i = 0; i < 30; i++) {
        align.updateHeading(heading_east, 12.0);
    }
    EXPECT_EQ(align.quality(), AlignmentQuality::CONVERGED);

    // Continuous update with steady gravity and heading
    for (int i = 0; i < 20; i++) {
        align.continuousUpdate(g, heading_east, 12.0, 0.0, 0.01);
    }

    EXPECT_GE(align.gravityConsistency(), 0.90);
    EXPECT_FALSE(align.needsRecalibration());
}

TEST(ContinuousAlignerTest, PhoneTiltTriggersRecalibration) {
    PhoneToVehicleAlignment align;
    Vec3d g_flat(0.0, 0.0, 9.81);
    for (int i = 0; i < 50; i++) {
        align.updateGravity(g_flat);
    }

    double heading_north = 0.0;
    for (int i = 0; i < 30; i++) {
        align.updateHeading(heading_north, 15.0);
    }
    EXPECT_EQ(align.quality(), AlignmentQuality::CONVERGED);

    // Sudden phone rotation / tilt on dashboard: gravity changes significantly to Y axis
    Vec3d g_tilted(0.0, 8.0, 5.0);
    for (int i = 0; i < 40; i++) {
        align.continuousUpdate(g_tilted, heading_north, 15.0, 0.0, 0.01);
    }

    EXPECT_TRUE(align.needsRecalibration());
    EXPECT_GT(align.alignmentInflationFactor(), 1.5);
}
