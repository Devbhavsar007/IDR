// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Unit tests for GNSS Spoofing Detector and Integrity Defense

#include <gtest/gtest.h>
#include "idr/integrity/spoofing_detector.h"
#include "idr/integrity/gnss_integrity.h"

using namespace idr;

TEST(SpoofingDetectorTest, VelocityCrossCheckClean) {
    SpoofingDetector detector;
    Vec3d gnss_vel(10.0, 0.0, 0.0);
    Vec3d imu_vel(10.2, 0.1, 0.0);
    double score = detector.checkVelocity(gnss_vel, imu_vel, 0.5, 0.5);
    EXPECT_LT(score, 0.2);
}

TEST(SpoofingDetectorTest, VelocityCrossCheckSpoofed) {
    SpoofingDetector detector;
    Vec3d gnss_vel(25.0, 0.0, 0.0);
    Vec3d imu_vel(0.0, 0.0, 0.0); // Vehicle is stationary
    double score = detector.checkVelocity(gnss_vel, imu_vel, 0.5, 0.5);
    EXPECT_GT(score, 0.8);
}

TEST(SpoofingDetectorTest, CurvatureCrossCheckClean) {
    SpoofingDetector detector;
    // Both GNSS course rate and gyro measure ~0.02 rad/s turn
    double score = detector.checkCurvature(0.02, 0.022);
    EXPECT_LT(score, 0.1);
}

TEST(SpoofingDetectorTest, CurvatureCrossCheckSpoofed) {
    SpoofingDetector detector;
    // GNSS says turning sharply (0.25 rad/s) but gyro says completely straight (0.0 rad/s)
    double score = detector.checkCurvature(0.25, 0.0);
    EXPECT_GT(score, 0.8);
}

TEST(SpoofingDetectorTest, Cn0UniformityClean) {
    SpoofingDetector detector;
    // Real satellites have elevation-dependent variation
    std::vector<SatelliteSignalInfo> sats = {
        {1, 15.0f, 45.0f, 28.0f, true},
        {2, 35.0f, 90.0f, 36.0f, true},
        {3, 65.0f, 130.0f, 44.0f, true},
        {4, 80.0f, 210.0f, 48.0f, true},
        {5, 20.0f, 300.0f, 31.0f, true},
        {6, 50.0f, 180.0f, 41.0f, true}
    };
    double score = detector.checkCn0Uniformity(sats);
    EXPECT_LT(score, 0.1);
}

TEST(SpoofingDetectorTest, Cn0UniformitySpoofed) {
    SpoofingDetector detector;
    // Spoofed/synthesized simulator signals have unnaturally identical C/N0
    std::vector<SatelliteSignalInfo> sats = {
        {1, 15.0f, 45.0f, 42.1f, true},
        {2, 35.0f, 90.0f, 42.0f, true},
        {3, 65.0f, 130.0f, 42.3f, true},
        {4, 80.0f, 210.0f, 42.2f, true},
        {5, 20.0f, 300.0f, 42.1f, true},
        {6, 50.0f, 180.0f, 42.0f, true}
    };
    double score = detector.checkCn0Uniformity(sats);
    EXPECT_GE(score, 0.8);
}

TEST(SpoofingDetectorTest, HysteresisStateMachine) {
    SpoofingDetector detector;
    EXPECT_FALSE(detector.isSuspect());
    EXPECT_FALSE(detector.isSpoofed());

    // Send 3 epochs with high suspect score
    for (int i = 0; i < 3; i++) {
        detector.update(0.70);
    }
    EXPECT_TRUE(detector.isSuspect());
    EXPECT_FALSE(detector.isSpoofed());

    // Send 5 epochs with alert score
    for (int i = 0; i < 5; i++) {
        detector.update(0.90);
    }
    EXPECT_TRUE(detector.isSpoofed());
}
