// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Unit tests for Phone-to-Vehicle Alignment.

#include "idr/alignment/phone_alignment.h"
#include <gtest/gtest.h>
#include <cmath>

namespace idr {
namespace {

TEST(AlignmentTest, InitialQualityIsNone) {
    PhoneToVehicleAlignment align;
    EXPECT_EQ(align.quality(), AlignmentQuality::NONE);
    EXPECT_NEAR(align.confidence(), 0.0, 1e-10);
}

TEST(AlignmentTest, GravityConvergenceFlat) {
    // Phone is flat face-up: gravity = [0, 0, 9.81]
    PhoneToVehicleAlignment align;
    AlignmentConfig config;
    config.gravity_init_samples = 20;
    align = PhoneToVehicleAlignment(config);

    for (int i = 0; i < 20; i++) {
        align.updateGravity(Vec3d(0.0, 0.0, 9.81));
    }

    EXPECT_EQ(align.quality(), AlignmentQuality::GRAVITY);
    EXPECT_NEAR(align.pitchRad(), 0.0, 0.01);
    EXPECT_NEAR(align.rollRad(), 0.0, 0.01);
}

TEST(AlignmentTest, GravityConvergenceTilted) {
    // Phone tilted ~30° forward: gravity has x component
    AlignmentConfig config;
    config.gravity_init_samples = 10;
    PhoneToVehicleAlignment align(config);

    double tilt = 30.0 * constants::kDegToRad;
    Vec3d g(9.81 * std::sin(tilt), 0.0, 9.81 * std::cos(tilt));

    for (int i = 0; i < 10; i++) {
        align.updateGravity(g);
    }

    EXPECT_EQ(align.quality(), AlignmentQuality::GRAVITY);
    EXPECT_NEAR(align.pitchRad(), -tilt, 0.02);  // pitch = -asin(gx/|g|)
}

TEST(AlignmentTest, HeadingRequiresGravityFirst) {
    PhoneToVehicleAlignment align;
    align.updateHeading(1.0, 10.0);
    EXPECT_EQ(align.quality(), AlignmentQuality::NONE);  // Still NONE
}

TEST(AlignmentTest, HeadingEstablishesCoarseAlignment) {
    AlignmentConfig config;
    config.gravity_init_samples = 5;
    config.min_speed_for_yaw_mps = 2.0;
    PhoneToVehicleAlignment align(config);

    // First: gravity
    for (int i = 0; i < 5; i++) {
        align.updateGravity(Vec3d(0, 0, 9.81));
    }
    EXPECT_EQ(align.quality(), AlignmentQuality::GRAVITY);

    // Then: heading
    align.updateHeading(0.5, 10.0);  // 0.5 rad heading at 10 m/s
    EXPECT_EQ(align.quality(), AlignmentQuality::COARSE);
    EXPECT_NEAR(align.yawRad(), 0.5, 0.1);
}

TEST(AlignmentTest, RotationIsUnitQuaternion) {
    AlignmentConfig config;
    config.gravity_init_samples = 5;
    PhoneToVehicleAlignment align(config);

    for (int i = 0; i < 5; i++) {
        align.updateGravity(Vec3d(0, 0, 9.81));
    }
    align.updateHeading(1.0, 10.0);

    double norm = align.rotation().norm();
    EXPECT_NEAR(norm, 1.0, 1e-10);
}

}  // namespace
}  // namespace idr
