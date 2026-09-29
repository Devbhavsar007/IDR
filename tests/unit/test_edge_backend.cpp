// IDR — Unit tests for Edge Backend.

#include "idr/edge/edge_backend.h"
#include <gtest/gtest.h>
#include <cstring>

namespace idr {
namespace edge {
namespace {

NavigationState makeSampleState() {
    NavigationState s;
    s.latitude_deg = 28.613939;   // India Gate
    s.longitude_deg = 77.209023;
    s.altitude_m = 216.0;
    s.speed_mps = 13.89;          // 50 km/h
    s.heading_deg = 45.0;
    s.horizontal_accuracy_m = 3.5;
    s.mode = NavigationMode::GNSS_INS;
    s.is_stationary = false;
    s.alignment_quality = 3;
    s.matched_road_id = 42;
    s.road_level = 0;
    s.timestamp_ns = 1000000000LL;
    return s;
}

TEST(EdgeBackendTest, NmeaFormat) {
    auto state = makeSampleState();
    auto nmea = EdgeBackend::formatNmea(state);

    // Starts with $PIDR
    EXPECT_EQ(nmea.substr(0, 5), "$PIDR");
    // Ends with *CS\r\n
    EXPECT_EQ(nmea.substr(nmea.size() - 2), "\r\n");
    // Contains a checksum marker
    EXPECT_NE(nmea.find('*'), std::string::npos);
    // Contains lat/lon
    EXPECT_NE(nmea.find("28.6139"), std::string::npos);
    EXPECT_NE(nmea.find("77.2090"), std::string::npos);
}

TEST(EdgeBackendTest, JsonFormat) {
    auto state = makeSampleState();
    auto json = EdgeBackend::formatJson(state);

    // Contains expected fields
    EXPECT_NE(json.find("\"lat\":28.6139"), std::string::npos);
    EXPECT_NE(json.find("\"lon\":77.2090"), std::string::npos);
    EXPECT_NE(json.find("\"spd\":13.89"), std::string::npos);
    EXPECT_NE(json.find("\"hdg\":45.0"), std::string::npos);
    EXPECT_NE(json.find("\"mode\":3"), std::string::npos);
}

TEST(EdgeBackendTest, BinaryPacketSize) {
    auto state = makeSampleState();
    auto pkt = EdgeBackend::formatBinary(state);
    EXPECT_EQ(sizeof(pkt), 40u);
}

TEST(EdgeBackendTest, BinaryPacketChecksum) {
    auto state = makeSampleState();
    auto pkt = EdgeBackend::formatBinary(state);

    // Verify checksum
    const auto* bytes = reinterpret_cast<const uint8_t*>(&pkt);
    uint8_t cs = 0;
    for (size_t i = 0; i < sizeof(pkt) - 1; i++) {
        cs ^= bytes[i];
    }
    EXPECT_EQ(cs, pkt.checksum);
}

TEST(EdgeBackendTest, BinaryLatLonScaling) {
    auto state = makeSampleState();
    auto pkt = EdgeBackend::formatBinary(state);

    // lat_deg_e7 should be close to 28.613939 * 1e7
    double reconstructed_lat = pkt.lat_deg_e7 / 1e7;
    EXPECT_NEAR(reconstructed_lat, 28.613939, 1e-6);

    double reconstructed_lon = pkt.lon_deg_e7 / 1e7;
    EXPECT_NEAR(reconstructed_lon, 77.209023, 1e-6);
}

TEST(EdgeBackendTest, SendCallbackInvoked) {
    EdgeConfig config;
    config.format = OutputFormat::NMEA_PROPRIETARY;
    EdgeBackend backend(config);

    std::vector<uint8_t> received;
    backend.setSendCallback([&](const uint8_t* data, size_t len) {
        received.assign(data, data + len);
    });

    backend.emit(makeSampleState());

    EXPECT_FALSE(received.empty());
    // Should start with '$'
    EXPECT_EQ(received[0], '$');
}

}  // namespace
}  // namespace edge
}  // namespace idr
