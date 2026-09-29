// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// JNI Bridge — Android native interface.
//
// This C-linkage layer exposes the IdrEngine to Android/Kotlin.
// It manages the engine lifecycle, marshals sensor data from Java arrays
// to C++ types, and returns NavigationState as a Java object.
//
// Thread safety: all JNI calls serialize on the engine mutex.

#pragma once

#include "idr/engine/idr_engine.h"
#include "idr/neural/neural_inference.h"
#include "idr/map_graph/road_graph.h"
#include "idr/map_matching/hmm_matcher.h"
#include "idr/types/common.h"

#include <spdlog/spdlog.h>
#include <memory>
#include <mutex>
#include <string>

namespace idr {
namespace android {

/// JNI handle — owns the engine and all subsystems for Android.
///
/// Lifecycle:
///   1. create()       — called from Application.onCreate()
///   2. configure()    — set vehicle profile, model path
///   3. start()        — begin processing
///   4. feedImu()/feedGnss()/feedBaro() — from sensor listeners
///   5. getNavigationState() — polled from UI thread or callback
///   6. stop()         — pause processing
///   7. destroy()      — called from onDestroy()
class IdrHandle {
public:
    /// Create a new handle with default config.
    static std::unique_ptr<IdrHandle> create() {
        return std::make_unique<IdrHandle>();
    }

    IdrHandle() = default;
    ~IdrHandle() { stop(); }

    /// Configure the engine before start.
    void configure(const EngineConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);
        config_ = config;
    }

    /// Set the vehicle profile.
    void setVehicleProfile(const std::string& profile_name) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (profile_name == "car") config_.vehicle = profiles::kCar;
        else if (profile_name == "bike") config_.vehicle = profiles::kBike;
        else if (profile_name == "auto_rickshaw") config_.vehicle = profiles::kAutoRickshaw;
        else if (profile_name == "bus") config_.vehicle = profiles::kBus;
        else if (profile_name == "truck") config_.vehicle = profiles::kTruck;
        spdlog::info("JNI: vehicle profile set to '{}'", profile_name);
    }

    /// Load the ONNX model for AI inference.
    bool loadModel(const std::string& model_path) {
        std::lock_guard<std::mutex> lock(mutex_);
        return neural_.loadModel(model_path);
    }

    /// Start the engine.
    void start() {
        std::lock_guard<std::mutex> lock(mutex_);
        engine_ = std::make_unique<IdrEngine>(config_);
        running_ = true;
        spdlog::info("JNI: engine started");
    }

    /// Stop the engine.
    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
        engine_.reset();
        spdlog::info("JNI: engine stopped");
    }

    /// Feed an IMU sample.
    /// @param timestamp_ns  Monotonic nanosecond timestamp
    /// @param accel         Accelerometer [x, y, z] in m/s²
    /// @param gyro          Gyroscope [x, y, z] in rad/s
    /// @param gravity       Gravity sensor [x, y, z] in m/s² (or nullptr)
    void feedImu(int64_t timestamp_ns,
                 const double accel[3],
                 const double gyro[3],
                 const double gravity[3]) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_ || !engine_) return;

        ImuSample imu;
        imu.timestamp_ns = timestamp_ns;
        imu.accel = Vec3d(accel[0], accel[1], accel[2]);
        imu.gyro = Vec3d(gyro[0], gyro[1], gyro[2]);
        if (gravity) {
            imu.gravity = Vec3d(gravity[0], gravity[1], gravity[2]);
        }

        engine_->processImu(imu);

        // Feed neural inference buffer
        Vec3d g = imu.hasGravity() ? imu.gravity : imu.accel;
        neural_.feedSample(imu.accel, imu.gyro, g);
    }

    /// Feed a GNSS measurement.
    void feedGnss(int64_t timestamp_ns,
                  double lat_deg, double lon_deg, double alt_m,
                  float speed_mps, float bearing_deg,
                  float h_accuracy_m, float v_accuracy_m,
                  int fix_type, int sat_count) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_ || !engine_) return;

        GnssMeasurement gnss;
        gnss.timestamp_ns = timestamp_ns;
        gnss.latitude_deg = lat_deg;
        gnss.longitude_deg = lon_deg;
        gnss.altitude_m = static_cast<float>(alt_m);
        gnss.speed_mps = speed_mps;
        gnss.bearing_deg = bearing_deg;
        gnss.horizontal_accuracy_m = h_accuracy_m;
        gnss.vertical_accuracy_m = v_accuracy_m;
        gnss.fix_type = static_cast<GnssFixType>(fix_type);
        gnss.satellite_count = static_cast<uint8_t>(sat_count);

        engine_->processGnss(gnss);
    }

    /// Feed a barometer sample.
    void feedBaro(int64_t timestamp_ns, double pressure_hpa) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_ || !engine_) return;

        BaroSample baro;
        baro.timestamp_ns = timestamp_ns;
        baro.pressure_hpa = pressure_hpa;

        engine_->processBaro(baro);
    }

    /// Get the latest navigation state.
    /// Returns a copy (thread-safe).
    NavigationState getNavigationState() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!engine_) return {};
        return engine_->lastOutput();
    }

    /// Get diagnostics for debugging overlay.
    struct Diagnostics {
        NavigationMode mode = NavigationMode::INIT;
        GnssIntegrity gnss_integrity = GnssIntegrity::DENIED;
        AlignmentQuality alignment = AlignmentQuality::NONE;
        bool is_stationary = false;
        bool model_loaded = false;
        double horizontal_accuracy_m = 999.0;
        uint32_t imu_count = 0;
    };

    Diagnostics getDiagnostics() const {
        std::lock_guard<std::mutex> lock(mutex_);
        Diagnostics d;
        if (engine_) {
            d.mode = engine_->navigationMode();
            d.gnss_integrity = engine_->gnssIntegrity();
            d.alignment = engine_->alignmentQuality();
            d.is_stationary = engine_->isStationary();
            d.horizontal_accuracy_m = engine_->lastOutput().horizontal_accuracy_m;
            d.imu_count = engine_->imuCount();
        }
        d.model_loaded = neural_.isModelLoaded();
        return d;
    }

    bool isRunning() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return running_;
    }

private:
    mutable std::mutex mutex_;
    EngineConfig config_;
    std::unique_ptr<IdrEngine> engine_;
    NeuralInference neural_;
    bool running_ = false;
};

}  // namespace android
}  // namespace idr
