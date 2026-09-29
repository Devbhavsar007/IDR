// IDR — Multi-Sensor Fusion Unit Tests (Barometer, Wheel Speed, Magnetometer)
// Copyright (c) 2026 IDR Project. All rights reserved.

#include "idr/engine/idr_engine.h"
#include "idr/fusion/eskf.h"
#include "idr/sensors/sensor_types.h"
#include <gtest/gtest.h>

namespace idr {
namespace {

TEST(MultiSensorFusionTest, BaroAltitudeUpdateReducesVerticalCovariance) {
    ErrorStateEkf ekf;
    double initial_p_zz = ekf.covariance()(eskf::kPosIdx + 2, eskf::kPosIdx + 2);

    double nominal_alt = 10.0;
    double baro_alt = 10.8;
    double sigma_alt = 0.5;

    bool accepted = ekf.updateBaroAltitude(baro_alt, nominal_alt, sigma_alt);
    EXPECT_TRUE(accepted);

    double updated_p_zz = ekf.covariance()(eskf::kPosIdx + 2, eskf::kPosIdx + 2);
    EXPECT_LT(updated_p_zz, initial_p_zz);

    // Error state should have positive vertical correction
    EXPECT_GT(ekf.errorState()(eskf::kPosIdx + 2), 0.0);
}

TEST(MultiSensorFusionTest, BaroAltitudeRejectsOutliers) {
    ErrorStateEkf ekf;
    double nominal_alt = 10.0;
    double wild_alt = 500.0; // 490m jump with sigma=0.5m should trigger NIS gate
    double sigma_alt = 0.5;

    bool accepted = ekf.updateBaroAltitude(wild_alt, nominal_alt, sigma_alt);
    EXPECT_FALSE(accepted);
}

TEST(MultiSensorFusionTest, WheelSpeedOdometryReducesVelocityCovariance) {
    ErrorStateEkf ekf;
    NominalState nominal;
    nominal.attitude = Quaterniond::Identity();
    nominal.velocity = Vec3d(15.0, 0.0, 0.0); // 15 m/s forward

    double initial_p_vx = ekf.covariance()(eskf::kVelIdx, eskf::kVelIdx);

    double measured_speed = 15.5; // slight speed delta
    double sigma_speed = 0.2;

    bool accepted = ekf.updateWheelSpeed(measured_speed, nominal, sigma_speed);
    EXPECT_TRUE(accepted);

    double updated_p_vx = ekf.covariance()(eskf::kVelIdx, eskf::kVelIdx);
    EXPECT_LT(updated_p_vx, initial_p_vx);
}

TEST(MultiSensorFusionTest, MagneticHeadingUpdateConstrainsYaw) {
    ErrorStateEkf ekf;
    double initial_yaw_var = ekf.covariance()(eskf::kAttIdx + 2, eskf::kAttIdx + 2);

    double nominal_yaw = 0.1;
    double measured_mag_yaw = 0.15;
    double sigma_yaw = 0.08;

    bool accepted = ekf.updateMagneticHeading(measured_mag_yaw, nominal_yaw, sigma_yaw);
    EXPECT_TRUE(accepted);

    double updated_yaw_var = ekf.covariance()(eskf::kAttIdx + 2, eskf::kAttIdx + 2);
    EXPECT_LT(updated_yaw_var, initial_yaw_var);
}

TEST(MultiSensorFusionTest, IdrEngineHandlesBaroAndWheelOdometry) {
    IdrEngine engine;

    // 1. Initial GNSS fix to initialize origin
    GnssMeasurement gnss;
    gnss.timestamp_ns = 1000000000ULL;
    gnss.latitude_deg = 28.5832;
    gnss.longitude_deg = 77.2185;
    gnss.altitude_m = 220.0;
    gnss.horizontal_accuracy_m = 1.5f;
    gnss.speed_mps = 12.0f;
    gnss.bearing_deg = 90.0f;
    gnss.fix_type = GnssFixType::FIX_3D;
    engine.processGnss(gnss);

    EXPECT_TRUE(engine.isInitialized());

    // 2. Barometer sample
    BaroSample baro;
    baro.timestamp_ns = 1010000000ULL;
    baro.pressure_hpa = 1013.25;
    engine.processBaro(baro);

    // 3. Wheel speed sample
    WheelSpeedSample wheel;
    wheel.timestamp_ns = 1020000000ULL;
    wheel.speed_mps = 12.0;
    wheel.accuracy_mps = 0.15;
    engine.processWheelSpeed(wheel);

    // 4. Magnetometer anomaly rejection
    MagSample anomalous_mag;
    anomalous_mag.timestamp_ns = 1030000000ULL;
    anomalous_mag.field_ut = Vec3d(120.0, 95.0, -110.0); // > 75 µT anomaly
    engine.processMagnetometer(anomalous_mag);

    // Verify engine is healthy and output state reflects initialization
    auto out = engine.lastOutput();
    EXPECT_NEAR(out.latitude_deg, 28.5832, 1e-4);
    EXPECT_NEAR(out.longitude_deg, 77.2185, 1e-4);
}

}  // namespace
}  // namespace idr
