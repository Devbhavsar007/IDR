// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Unit tests for TimestampValidator.

#include "idr/timing/timestamp_validator.h"
#include <gtest/gtest.h>

namespace idr {
namespace {

TEST(TimestampValidatorTest, FirstSampleIsValid) {
    TimestampValidator validator(100.0, 0.1, 1e-6);
    auto result = validator.validate(1000000000);  // 1 second
    EXPECT_TRUE(result.valid);
    EXPECT_TRUE(result.first_sample);
    EXPECT_FALSE(result.gap_detected);
    EXPECT_FALSE(result.duplicate);
    EXPECT_FALSE(result.out_of_order);
}

TEST(TimestampValidatorTest, NormalSequence) {
    TimestampValidator validator(100.0, 0.1, 1e-6);
    validator.validate(secToNs(0.0));

    auto r1 = validator.validate(secToNs(0.01));  // 10 ms = 100 Hz
    EXPECT_TRUE(r1.valid);
    EXPECT_NEAR(r1.dt_sec, 0.01, 1e-9);

    auto r2 = validator.validate(secToNs(0.02));
    EXPECT_TRUE(r2.valid);
    EXPECT_NEAR(r2.dt_sec, 0.01, 1e-9);
}

TEST(TimestampValidatorTest, DuplicateTimestamp) {
    TimestampValidator validator(100.0, 0.1, 1e-6);
    validator.validate(secToNs(1.0));

    auto result = validator.validate(secToNs(1.0));  // same timestamp
    EXPECT_FALSE(result.valid);
    EXPECT_TRUE(result.duplicate);
}

TEST(TimestampValidatorTest, OutOfOrder) {
    TimestampValidator validator(100.0, 0.1, 1e-6);
    validator.validate(secToNs(2.0));

    auto result = validator.validate(secToNs(1.5));  // earlier
    EXPECT_FALSE(result.valid);
    EXPECT_TRUE(result.out_of_order);
}

TEST(TimestampValidatorTest, GapDetection) {
    TimestampValidator validator(100.0, 0.1, 1e-6);
    validator.validate(secToNs(0.0));

    auto result = validator.validate(secToNs(0.5));  // 500 ms gap > 100 ms max
    EXPECT_TRUE(result.valid);  // still valid, but gap flagged
    EXPECT_TRUE(result.gap_detected);
    EXPECT_NEAR(result.dt_sec, 0.5, 1e-9);
}

TEST(TimestampValidatorTest, BurstDelivery) {
    TimestampValidator validator(100.0, 0.1, 1e-7);  // min_dt = 100 ns
    validator.validate(secToNs(0.0));

    // Two samples 50 ns apart (burst delivery on Android)
    auto result = validator.validate(50);  // 50 ns
    EXPECT_FALSE(result.valid);
    EXPECT_TRUE(result.duplicate);
}

TEST(TimestampValidatorTest, RateEstimation) {
    TimestampValidator validator(100.0, 0.1, 1e-6);

    // Feed 100 samples at exactly 200 Hz
    for (int i = 0; i < 100; ++i) {
        validator.validate(secToNs(i * 0.005));  // 5 ms intervals = 200 Hz
    }

    // Rate estimate should converge toward 200 Hz
    EXPECT_GT(validator.estimatedRateHz(), 150.0);
    EXPECT_LT(validator.estimatedRateHz(), 250.0);
}

TEST(TimestampValidatorTest, Reset) {
    TimestampValidator validator(100.0, 0.1, 1e-6);
    validator.validate(secToNs(1.0));
    validator.validate(secToNs(1.01));
    EXPECT_EQ(validator.sampleCount(), 2u);

    validator.reset();
    EXPECT_EQ(validator.sampleCount(), 0u);

    auto result = validator.validate(secToNs(5.0));
    EXPECT_TRUE(result.first_sample);
}

}  // namespace
}  // namespace idr
