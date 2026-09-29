// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Unit tests for coordinate conversion utilities.

#include "idr/math_utils/geo_utils.h"
#include <gtest/gtest.h>
#include <cmath>

namespace idr {
namespace geo {
namespace {

// Known reference: London (51.5074° N, 0.1278° W, 11 m altitude)
constexpr double kLondonLat = 51.5074;
constexpr double kLondonLon = -0.1278;
constexpr double kLondonAlt = 11.0;

// Known reference: New Delhi (28.6139° N, 77.2090° E, 216 m altitude)
constexpr double kDelhiLat = 28.6139;
constexpr double kDelhiLon = 77.2090;
constexpr double kDelhiAlt = 216.0;

TEST(GeoUtilsTest, LlaToEcefKnownPoint) {
    // Equator at prime meridian, sea level → should be (a, 0, 0)
    Vec3d ecef = llaToEcef(0.0, 0.0, 0.0);
    EXPECT_NEAR(ecef.x(), constants::kWgs84A, 1.0);  // ~6378137 m
    EXPECT_NEAR(ecef.y(), 0.0, 1.0);
    EXPECT_NEAR(ecef.z(), 0.0, 1.0);
}

TEST(GeoUtilsTest, RoundTripLlaEcef) {
    Vec3d ecef = llaToEcef(kDelhiLat, kDelhiLon, kDelhiAlt);
    GeoCoord back = ecefToLla(ecef);

    EXPECT_NEAR(back.latitude_deg,  kDelhiLat, 1e-6);
    EXPECT_NEAR(back.longitude_deg, kDelhiLon, 1e-6);
    EXPECT_NEAR(back.altitude_m,    kDelhiAlt, 0.1);
}

TEST(GeoUtilsTest, LlaToEnuZeroOffset) {
    // A point at the reference itself → ENU should be ~(0,0,0)
    Vec3d enu = llaToEnu(kDelhiLat, kDelhiLon, kDelhiAlt,
                         kDelhiLat, kDelhiLon, kDelhiAlt);
    EXPECT_NEAR(enu.x(), 0.0, 1e-6);  // East
    EXPECT_NEAR(enu.y(), 0.0, 1e-6);  // North
    EXPECT_NEAR(enu.z(), 0.0, 1e-6);  // Up
}

TEST(GeoUtilsTest, LlaToEnuNorthOffset) {
    // Move ~111.32 m North (≈ 0.001° latitude)
    double offset_deg = 0.001;
    Vec3d enu = llaToEnu(kDelhiLat + offset_deg, kDelhiLon, kDelhiAlt,
                         kDelhiLat, kDelhiLon, kDelhiAlt);

    // North component should be ~111.3 m, East ≈ 0, Up ≈ 0
    EXPECT_NEAR(enu.y(), 111.3, 1.0);   // North
    EXPECT_NEAR(enu.x(), 0.0, 0.1);     // East
    EXPECT_NEAR(enu.z(), 0.0, 0.1);     // Up
}

TEST(GeoUtilsTest, RoundTripLlaEnu) {
    Vec3d enu(100.0, 200.0, 50.0);  // 100m E, 200m N, 50m up from Delhi
    GeoCoord back = enuToLla(enu, kDelhiLat, kDelhiLon, kDelhiAlt);

    // Convert back to ENU
    Vec3d enu2 = llaToEnu(back.latitude_deg, back.longitude_deg, back.altitude_m,
                          kDelhiLat, kDelhiLon, kDelhiAlt);

    EXPECT_NEAR(enu2.x(), enu.x(), 0.01);
    EXPECT_NEAR(enu2.y(), enu.y(), 0.01);
    EXPECT_NEAR(enu2.z(), enu.z(), 0.01);
}

TEST(GeoUtilsTest, HaversineDistance) {
    // London to Delhi: ~6718 km
    double dist = haversineDistance(kLondonLat, kLondonLon, kDelhiLat, kDelhiLon);
    EXPECT_GT(dist, 6700000.0);
    EXPECT_LT(dist, 6750000.0);
}

TEST(GeoUtilsTest, HaversineSamePoint) {
    double dist = haversineDistance(kDelhiLat, kDelhiLon, kDelhiLat, kDelhiLon);
    EXPECT_NEAR(dist, 0.0, 0.01);
}

}  // namespace
}  // namespace geo
}  // namespace idr
