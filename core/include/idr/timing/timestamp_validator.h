// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Timestamp validation — detects and handles non-ideal sensor timing:
//   - Negative dt
//   - Duplicate timestamps
//   - Excessive gaps
//   - Out-of-order samples
//   - Variable sample rates

#pragma once

#include "idr/types/common.h"
#include <cmath>

namespace idr {

/// Result of validating a new timestamp against the previous one.
struct TimestampValidation {
    double dt_sec = 0.0;         ///< Time delta in seconds (always ≥ 0 if valid)
    bool valid = false;          ///< Whether this sample should be processed
    bool gap_detected = false;   ///< dt exceeded max_gap_sec
    bool duplicate = false;      ///< Identical to previous timestamp
    bool out_of_order = false;   ///< Earlier than previous timestamp
    bool first_sample = false;   ///< This is the first sample seen
};

/// Validates monotonic timestamp sequences from a single sensor source.
/// Handles the messy reality of smartphone sensor delivery.
class TimestampValidator {
public:
    /// @param expected_rate_hz  Expected sensor rate (for gap detection)
    /// @param max_gap_sec       Maximum acceptable dt before flagging a gap
    /// @param min_dt_sec        Minimum acceptable dt (rejects duplicates/bursts)
    explicit TimestampValidator(double expected_rate_hz = 100.0,
                                double max_gap_sec = 0.1,
                                double min_dt_sec = 1e-6)
        : expected_dt_sec_(1.0 / expected_rate_hz)
        , max_gap_sec_(max_gap_sec)
        , min_dt_sec_(min_dt_sec) {}

    /// Validate a new timestamp. Returns validation result with dt.
    TimestampValidation validate(Timestamp timestamp_ns) {
        TimestampValidation result;

        if (last_timestamp_ns_ == kInvalidTimestamp) {
            // First sample
            last_timestamp_ns_ = timestamp_ns;
            sample_count_ = 1;
            result.first_sample = true;
            result.valid = true;
            result.dt_sec = expected_dt_sec_;  // Use expected dt for first sample
            return result;
        }

        int64_t delta_ns = timestamp_ns - last_timestamp_ns_;
        double dt = static_cast<double>(delta_ns) * 1e-9;

        // Check: duplicate timestamp
        if (delta_ns == 0) {
            result.duplicate = true;
            result.valid = false;
            result.dt_sec = 0.0;
            return result;
        }

        // Check: out-of-order (negative dt)
        if (delta_ns < 0) {
            result.out_of_order = true;
            result.valid = false;
            result.dt_sec = dt;
            return result;
        }

        // Check: too small dt (burst delivery — common on Android)
        if (dt < min_dt_sec_) {
            result.duplicate = true;
            result.valid = false;
            result.dt_sec = dt;
            return result;
        }

        // Check: excessive gap
        if (dt > max_gap_sec_) {
            result.gap_detected = true;
            // Still valid, but downstream should handle the gap
            // (e.g., inflate covariance, skip INS propagation)
        }

        result.dt_sec = dt;
        result.valid = true;
        last_timestamp_ns_ = timestamp_ns;
        sample_count_++;

        // Update running sample rate estimate
        updateRateEstimate(dt);

        return result;
    }

    /// Reset the validator (e.g., after pause/resume)
    void reset() {
        last_timestamp_ns_ = kInvalidTimestamp;
        sample_count_ = 0;
        running_rate_hz_ = 1.0 / expected_dt_sec_;
    }

    /// Get the last valid timestamp
    Timestamp lastTimestamp() const { return last_timestamp_ns_; }

    /// Get the estimated actual sample rate (Hz)
    double estimatedRateHz() const { return running_rate_hz_; }

    /// Total number of valid samples processed
    uint64_t sampleCount() const { return sample_count_; }

private:
    void updateRateEstimate(double dt) {
        if (dt > 0.0 && dt < max_gap_sec_) {
            double instant_rate = 1.0 / dt;
            // Exponential moving average (α = 0.05)
            running_rate_hz_ = 0.95 * running_rate_hz_ + 0.05 * instant_rate;
        }
    }

    double expected_dt_sec_;
    double max_gap_sec_;
    double min_dt_sec_;
    Timestamp last_timestamp_ns_ = kInvalidTimestamp;
    uint64_t sample_count_ = 0;
    double running_rate_hz_ = 0.0;
};

}  // namespace idr
