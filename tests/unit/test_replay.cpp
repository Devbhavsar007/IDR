// IDR — Unit tests for ReplayEngine and GNSS Outage Simulation.

#include "idr/engine/replay_engine.h"
#include <gtest/gtest.h>
#include <cstdio>
#include <fstream>

namespace idr {
namespace {

// Helper to write a temporary CSV file for replay testing
std::string createSyntheticReplayCsv(const std::string& filename, double duration_sec = 10.0) {
    std::ofstream out(filename);
    out << "timestamp,acc_x,acc_y,acc_z,gyro_x,gyro_y,gyro_z,lat,lon,alt,speed,bearing,accuracy\n";

    double lat = 28.613939;
    double lon = 77.209023;
    double t = 0.0;
    double dt = 0.01;  // 100 Hz IMU

    while (t <= duration_sec) {
        // Vehicle moving North at 10 m/s: ~0.00009 deg lat per second
        double cur_lat = lat + (t * 10.0 / 111111.0);
        double cur_lon = lon;

        // IMU: 9.81 m/s² vertical, small forward accel at start then 0
        double acc_x = 0.0;
        double acc_y = (t < 1.0) ? 1.0 : 0.0;  // North/forward
        double acc_z = 9.81;

        // GNSS output at 1 Hz
        bool is_gnss_epoch = (std::fmod(t + 0.001, 1.0) < dt);
        if (is_gnss_epoch) {
            out << t << ","
                << acc_x << "," << acc_y << "," << acc_z << ","
                << "0.0,0.0,0.0,"
                << cur_lat << "," << cur_lon << ",216.0,10.0,0.0,3.0\n";
        } else {
            out << t << ","
                << acc_x << "," << acc_y << "," << acc_z << ","
                << "0.0,0.0,0.0,,,,,\n";
        }

        t += dt;
    }
    out.close();
    return filename;
}

TEST(ReplayEngineTest, LoadMissingFileFails) {
    ReplayEngine engine;
    EXPECT_FALSE(engine.loadCsv("non_existent_file_xyz_123.csv"));
    EXPECT_EQ(engine.recordCount(), 0u);
}

TEST(ReplayEngineTest, LoadSyntheticCsvSuccess) {
    std::string csv_path = "test_synthetic_replay.csv";
    createSyntheticReplayCsv(csv_path, 2.0);

    ReplayEngine engine;
    EXPECT_TRUE(engine.loadCsv(csv_path));
    EXPECT_GT(engine.recordCount(), 100u);

    std::remove(csv_path.c_str());
}

TEST(ReplayEngineTest, RunNominalReplay) {
    std::string csv_path = "test_nominal_replay.csv";
    createSyntheticReplayCsv(csv_path, 3.0);

    ReplayEngine engine;
    ASSERT_TRUE(engine.loadCsv(csv_path));

    int callback_count = 0;
    auto metrics = engine.run([&](const NavigationState& /*state*/) {
        callback_count++;
    });

    EXPECT_GT(callback_count, 0);
    EXPECT_NEAR(metrics.total_duration_sec, 3.0, 0.2);
    EXPECT_EQ(metrics.gnss_outage_count, 0);

    std::remove(csv_path.c_str());
}

TEST(ReplayEngineTest, RunSimulatedGnssOutage) {
    std::string csv_path = "test_outage_replay.csv";
    createSyntheticReplayCsv(csv_path, 6.0);

    ReplayEngine engine;
    ASSERT_TRUE(engine.loadCsv(csv_path));

    // Mask GNSS between t=2.0s and t=4.5s (2.5 second outage)
    auto metrics = engine.run(nullptr, 2.0, 4.5);

    EXPECT_GE(metrics.gnss_outage_count, 1);
    EXPECT_GE(metrics.max_gnss_outage_sec, 2.0);

    std::remove(csv_path.c_str());
}

}  // namespace
}  // namespace idr
