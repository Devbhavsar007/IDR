// IDR — Unit tests for Android JNI Handle.

#include "android/jni/idr_jni_handle.h"
#include <gtest/gtest.h>

namespace idr {
namespace android {
namespace {

TEST(JniHandleTest, CreateAndInitialState) {
    auto handle = IdrHandle::create();
    ASSERT_NE(handle, nullptr);
    EXPECT_FALSE(handle->isRunning());

    auto diag = handle->getDiagnostics();
    EXPECT_EQ(diag.mode, NavigationMode::INIT);
    EXPECT_FALSE(diag.is_stationary);
    EXPECT_EQ(diag.imu_count, 0u);
}

TEST(JniHandleTest, StartAndStopLifecycle) {
    auto handle = IdrHandle::create();
    handle->setVehicleProfile("car");
    EXPECT_FALSE(handle->isRunning());

    handle->start();
    EXPECT_TRUE(handle->isRunning());

    handle->stop();
    EXPECT_FALSE(handle->isRunning());
}

TEST(JniHandleTest, VehicleProfiles) {
    auto handle = IdrHandle::create();
    // Valid profiles should configure without error
    handle->setVehicleProfile("car");
    handle->setVehicleProfile("bike");
    handle->setVehicleProfile("auto_rickshaw");
    handle->setVehicleProfile("bus");
    handle->setVehicleProfile("truck");
    // Unknown profile should be handled gracefully
    handle->setVehicleProfile("hovercraft");
}

TEST(JniHandleTest, FeedImuWhileRunning) {
    auto handle = IdrHandle::create();
    handle->start();

    int64_t t = 1000000000LL;
    handle->feedGnss(t, 28.613939, 77.209023, 216.0, 0.0f, 0.0f, 3.0f, 5.0f, 2, 12);

    double accel[3] = {0.0, 0.0, 9.81};
    double gyro[3] = {0.0, 0.0, 0.0};
    double gravity[3] = {0.0, 0.0, 9.81};

    for (int i = 0; i < 50; i++) {
        t += 10000000LL;  // 100 Hz (10 ms)
        handle->feedImu(t, accel, gyro, gravity);
    }

    auto diag = handle->getDiagnostics();
    EXPECT_GT(diag.imu_count, 0u);

    auto state = handle->getNavigationState();
    EXPECT_GT(state.timestamp_ns, 1000000000LL);
    EXPECT_LE(state.timestamp_ns, t);
}

TEST(JniHandleTest, FeedGnssMeasurement) {
    auto handle = IdrHandle::create();
    handle->start();

    // First feed IMU to initialize clock
    double accel[3] = {0.0, 0.0, 9.81};
    double gyro[3] = {0.0, 0.0, 0.0};
    int64_t t = 1000000000LL;
    handle->feedImu(t, accel, gyro, nullptr);

    // Feed valid GNSS fix
    handle->feedGnss(t, 28.613939, 77.209023, 216.0, 0.0f, 0.0f, 3.0f, 5.0f, 2, 12);

    auto state = handle->getNavigationState();
    // After GNSS processing with stationary IMU, position should be near India Gate
    EXPECT_NEAR(state.latitude_deg, 28.613939, 1e-4);
    EXPECT_NEAR(state.longitude_deg, 77.209023, 1e-4);
}

TEST(JniHandleTest, FeedBarometer) {
    auto handle = IdrHandle::create();
    handle->start();

    int64_t t = 1000000000LL;
    handle->feedBaro(t, 1013.25);
    // Should not crash and model_loaded remains false
    EXPECT_FALSE(handle->getDiagnostics().model_loaded);
}

TEST(JniHandleTest, IgnoresFeedsWhenStopped) {
    auto handle = IdrHandle::create();
    // Do NOT call start()

    double accel[3] = {0.0, 0.0, 9.81};
    double gyro[3] = {0.0, 0.0, 0.0};
    handle->feedImu(1000000000LL, accel, gyro, nullptr);
    handle->feedGnss(1000000000LL, 28.6, 77.2, 200.0, 0.0f, 0.0f, 5.0f, 5.0f, 2, 8);
    handle->feedBaro(1000000000LL, 1013.0);

    auto diag = handle->getDiagnostics();
    EXPECT_EQ(diag.imu_count, 0u);
}

}  // namespace
}  // namespace android
}  // namespace idr
