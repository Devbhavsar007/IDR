// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// ONNX Runtime Inference Bridge.
//
// Loads the exported multi-task model and runs inference on IMU windows.
// Outputs velocity, displacement, heading change, speed, and motion class
// as EKF measurements.
//
// Designed for both Android (NNAPI) and edge (CPU/GPU) execution providers.

#pragma once

#include "idr/constraints/nhc.h"
#include "idr/types/common.h"

#include <spdlog/spdlog.h>
#include <array>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

// Forward declarations — actual ONNX Runtime headers included only in .cpp
// This keeps the header lightweight for compilation speed.
namespace Ort {
class Env;
class Session;
class SessionOptions;
class MemoryInfo;
}  // namespace Ort

namespace idr {

/// AI model inference result — matches the 5-head model output.
struct AiInferenceResult {
    /// Predicted velocity in ENU frame (m/s)
    Vec3d velocity = Vec3d::Zero();
    /// Velocity uncertainty (1σ, m/s per axis)
    Vec3d velocity_sigma = Vec3d::Ones() * 10.0;

    /// Predicted displacement in ENU frame (meters)
    Vec3d displacement = Vec3d::Zero();
    /// Displacement uncertainty (1σ, m per axis)
    Vec3d displacement_sigma = Vec3d::Ones() * 10.0;

    /// Heading change (radians)
    double heading_change = 0.0;
    double heading_sigma = 1.0;

    /// Scalar speed (m/s)
    double speed = 0.0;
    double speed_sigma = 5.0;

    /// Motion class distribution (14 classes)
    MotionClassDistribution motion_class;

    /// Inference latency (milliseconds)
    double latency_ms = 0.0;

    /// Whether this result is valid
    bool valid = false;
};

/// Configuration for the neural inference engine.
struct NeuralConfig {
    /// Path to the ONNX model file
    std::string model_path = "models/idr_unified_v1.onnx";

    /// Inference rate (Hz) — how often to run the model
    double inference_rate_hz = 10.0;

    /// Window size (samples) — must match training window
    int window_size = 200;

    /// Input channels (accel_xyz + gyro_xyz + gravity_xyz)
    int input_channels = 9;

    /// Number of inference threads
    int num_threads = 2;

    /// Use FP16 if available
    bool use_fp16 = true;

    /// Execution provider: "cpu", "nnapi", "gpu"
    std::string provider = "cpu";
};

/// Sensor window buffer — accumulates IMU samples for inference.
class SensorWindowBuffer {
public:
    explicit SensorWindowBuffer(int window_size = 200, int channels = 9)
        : window_size_(window_size), channels_(channels) {
        buffer_.resize(static_cast<size_t>(window_size * channels), 0.0f);
    }

    /// Add a new sample (9 channels: accel_xyz + gyro_xyz + gravity_xyz).
    /// Shifts the window and appends at the end.
    void addSample(const Vec3d& accel, const Vec3d& gyro, const Vec3d& gravity) {
        // Shift left by one row
        size_t row_size = static_cast<size_t>(channels_);
        size_t total = static_cast<size_t>(window_size_) * row_size;
        for (size_t i = 0; i < total - row_size; i++) {
            buffer_[i] = buffer_[i + row_size];
        }

        // Append new sample at end
        size_t offset = total - row_size;
        buffer_[offset + 0] = static_cast<float>(accel.x());
        buffer_[offset + 1] = static_cast<float>(accel.y());
        buffer_[offset + 2] = static_cast<float>(accel.z());
        buffer_[offset + 3] = static_cast<float>(gyro.x());
        buffer_[offset + 4] = static_cast<float>(gyro.y());
        buffer_[offset + 5] = static_cast<float>(gyro.z());
        buffer_[offset + 6] = static_cast<float>(gravity.x());
        buffer_[offset + 7] = static_cast<float>(gravity.y());
        buffer_[offset + 8] = static_cast<float>(gravity.z());

        sample_count_++;
    }

    /// Whether the buffer has accumulated a full window.
    bool isFull() const { return sample_count_ >= window_size_; }

    /// Get the buffer data pointer (for ONNX input tensor).
    const float* data() const { return buffer_.data(); }

    /// Get buffer dimensions: [1, window_size, channels]
    std::array<int64_t, 3> shape() const {
        return {1, static_cast<int64_t>(window_size_), static_cast<int64_t>(channels_)};
    }

    /// Total number of elements
    size_t size() const { return buffer_.size(); }

    void reset() { sample_count_ = 0; std::fill(buffer_.begin(), buffer_.end(), 0.0f); }

private:
    int window_size_;
    int channels_;
    int sample_count_ = 0;
    std::vector<float> buffer_;
};

/// Neural inference engine — wraps ONNX Runtime.
///
/// This is a header-only interface. The actual ONNX Runtime calls
/// are in neural_inference.cpp (linked only when ONNX Runtime is available).
///
/// When ONNX Runtime is NOT available (e.g., during unit testing),
/// this class returns invalid results and the engine falls back to
/// pure INS + classical constraints.
class NeuralInference {
public:
    explicit NeuralInference(const NeuralConfig& config = {})
        : config_(config)
        , window_buffer_(config.window_size, config.input_channels) {}

    /// Load the ONNX model. Returns true on success.
    /// If ONNX Runtime is not linked, returns false gracefully.
    bool loadModel(const std::string& model_path) {
        config_.model_path = model_path;
        // Actual loading happens in .cpp implementation
        // For now, check if file exists
        model_loaded_ = false;  // Will be set by implementation
        spdlog::info("NeuralInference: model load requested: {}", model_path);
        return model_loaded_;
    }

    /// Feed a new IMU sample into the window buffer.
    void feedSample(const Vec3d& accel, const Vec3d& gyro, const Vec3d& gravity) {
        window_buffer_.addSample(accel, gyro, gravity);
        samples_since_inference_++;
    }

    /// Check if inference should run this cycle.
    bool shouldRunInference(double imu_rate_hz) const {
        if (!model_loaded_ || !window_buffer_.isFull()) return false;
        int run_every = static_cast<int>(imu_rate_hz / config_.inference_rate_hz);
        return run_every > 0 && samples_since_inference_ >= run_every;
    }

    /// Run inference on the current window buffer.
    /// Returns the AI prediction result.
    AiInferenceResult runInference() {
        samples_since_inference_ = 0;

        if (!model_loaded_) {
            return {};  // Invalid result — engine falls back to classical
        }

        // Actual ONNX Runtime inference would happen here.
        // Placeholder: return invalid so engine uses classical path.
        AiInferenceResult result;
        result.valid = false;
        return result;
    }

    /// Convert log-variance output to sigma (standard deviation).
    static Vec3d logVarToSigma(const Vec3d& log_var) {
        return Vec3d(
            std::exp(0.5 * log_var.x()),
            std::exp(0.5 * log_var.y()),
            std::exp(0.5 * log_var.z())
        );
    }

    bool isModelLoaded() const { return model_loaded_; }
    const NeuralConfig& config() const { return config_; }

private:
    NeuralConfig config_;
    SensorWindowBuffer window_buffer_;
    bool model_loaded_ = false;
    int samples_since_inference_ = 0;
};

}  // namespace idr
