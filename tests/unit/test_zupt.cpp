// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Unit tests for ZUPT detector.

#include "idr/constraints/zupt.h"
#include <gtest/gtest.h>
#include <cmath>

namespace idr {
namespace {

ImuSample makeStationaryImu(double time_sec) {
    ImuSample imu;
    imu.timestamp_ns = secToNs(time_sec);
    imu.accel = Vec3d(0.0, 0.0, constants::kGravity);  // Only gravity
    imu.gyro = Vec3d(0.0, 0.0, 0.0);                   // No rotation
    return imu;
}

ImuSample makeMovingImu(double time_sec) {
    ImuSample imu;
    imu.timestamp_ns = secToNs(time_sec);
    // Forward acceleration + gravity + some vibration
    imu.accel = Vec3d(2.0, 0.3, constants::kGravity + 0.5);
    imu.gyro = Vec3d(0.01, 0.02, 0.1);
    return imu;
}

TEST(ZuptTest, InitiallyNotStationary) {
    ZuptDetector det;
    auto result = det.update(makeStationaryImu(0.0));
    EXPECT_FALSE(result.is_stationary);  // Not enough samples yet
}

TEST(ZuptTest, DetectsStationaryAfterWindow) {
    ZuptConfig config;
    config.window_size = 10;
    config.min_consecutive = 5;
    ZuptDetector det(config);

    ZuptResult result;
    for (int i = 0; i < 20; i++) {
        result = det.update(makeStationaryImu(i * 0.01));
    }

    EXPECT_TRUE(result.is_stationary);
    EXPECT_TRUE(result.zupt_available);
    EXPECT_GT(result.stationary_confidence, 0.5);
    EXPECT_LT(result.accel_variance, 0.01);
    EXPECT_LT(result.gyro_magnitude, 0.01);
}

TEST(ZuptTest, DetectsMotionAfterStationary) {
    ZuptConfig config;
    config.window_size = 10;
    config.min_consecutive = 5;
    ZuptDetector det(config);

    // Stationary
    for (int i = 0; i < 20; i++) {
        det.update(makeStationaryImu(i * 0.01));
    }
    EXPECT_TRUE(det.isStationary());

    // Start moving
    ZuptResult result;
    for (int i = 20; i < 40; i++) {
        result = det.update(makeMovingImu(i * 0.01));
    }

    EXPECT_FALSE(result.is_stationary);
    EXPECT_FALSE(result.zupt_available);
}

TEST(ZuptTest, GnssSpeedCrossValidation) {
    ZuptConfig config;
    config.window_size = 10;
    config.min_consecutive = 5;
    ZuptDetector det(config);

    det.updateGnssSpeed(20.0);  // GNSS says 20 m/s

    // Even with quiet IMU, GNSS speed overrides
    for (int i = 0; i < 20; i++) {
        det.update(makeStationaryImu(i * 0.01));
    }

    EXPECT_FALSE(det.isStationary());
}

}  // namespace
}  // namespace idr
