// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Edge Backend — serial/UDP output adapter for embedded deployment.
//
// Supports two output modes:
//   1. UART serial (for MCUs, NMEA-like sentences)
//   2. UDP broadcast (for LAN dashboards, fleet trackers)
//
// Message format: compact binary or NMEA-like text.

#pragma once

#include "idr/engine/idr_engine.h"
#include "idr/types/common.h"

#include <spdlog/spdlog.h>
#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

namespace idr {
namespace edge {

/// Output format for edge backend.
enum class OutputFormat : uint8_t {
    BINARY_COMPACT = 0,  ///< 40-byte packed struct
    NMEA_PROPRIETARY = 1,  ///< $PIDR,lat,lon,speed,...*CS
    JSON_COMPACT = 2,  ///< {"lat":28.6,"lon":77.2,...}
};

/// Edge output configuration.
struct EdgeConfig {
    OutputFormat format = OutputFormat::NMEA_PROPRIETARY;
    double output_rate_hz = 10.0;

    // Serial config
    std::string serial_port;  ///< e.g., "COM3" or "/dev/ttyUSB0"
    int baud_rate = 115200;

    // UDP config
    std::string udp_host = "255.255.255.255";  ///< broadcast
    int udp_port = 5555;

    // Enable/disable outputs
    bool serial_enabled = false;
    bool udp_enabled = false;
};

/// Compact binary output packet (40 bytes, little-endian).
/// Designed for low-bandwidth serial links.
#pragma pack(push, 1)
struct BinaryPacket {
    uint8_t  header[2] = {'I', 'D'};   ///< Magic: "ID"
    uint8_t  version = 1;
    uint8_t  mode;                        ///< NavigationMode enum
    int32_t  lat_deg_e7;                  ///< Latitude × 1e7
    int32_t  lon_deg_e7;                  ///< Longitude × 1e7
    int16_t  alt_dm;                      ///< Altitude in decimeters
    uint16_t speed_cmps;                  ///< Speed in cm/s
    uint16_t heading_cdeg;                ///< Heading in centidegrees [0, 36000)
    uint16_t h_accuracy_cm;               ///< Horizontal accuracy in cm
    uint8_t  gnss_integrity;
    uint8_t  alignment_quality;
    uint8_t  is_stationary;
    int64_t  timestamp_ns;                ///< Monotonic timestamp
    int32_t  matched_road_id = -1;        ///< Map-matched road ID
    uint8_t  road_level;
    uint8_t  reserved[3] = {};
    uint8_t  checksum;                    ///< XOR of all preceding bytes
};
#pragma pack(pop)

/// Edge output adapter.
class EdgeBackend {
public:
    /// Callback for sending data (serial write or UDP send).
    using SendCallback = std::function<void(const uint8_t* data, size_t len)>;

    explicit EdgeBackend(const EdgeConfig& config = {})
        : config_(config) {}

    /// Set the send callback (called when a packet is ready).
    void setSendCallback(SendCallback cb) { send_cb_ = std::move(cb); }

    /// Format and send a NavigationState.
    void emit(const NavigationState& state) {
        switch (config_.format) {
            case OutputFormat::BINARY_COMPACT:
                emitBinary(state);
                break;
            case OutputFormat::NMEA_PROPRIETARY:
                emitNmea(state);
                break;
            case OutputFormat::JSON_COMPACT:
                emitJson(state);
                break;
        }
    }

    /// Format as NMEA proprietary sentence.
    /// $PIDR,<lat>,<lon>,<alt>,<speed>,<heading>,<mode>,<hAcc>,<stationary>*CS\r\n
    static std::string formatNmea(const NavigationState& state) {
        char buf[256];
        int len = snprintf(buf, sizeof(buf),
            "$PIDR,%.7f,%.7f,%.1f,%.2f,%.1f,%d,%.1f,%d",
            state.latitude_deg,
            state.longitude_deg,
            state.altitude_m,
            state.speed_mps,
            state.heading_deg,
            static_cast<int>(state.mode),
            state.horizontal_accuracy_m,
            state.is_stationary ? 1 : 0
        );

        // NMEA checksum: XOR of all chars between $ and *
        uint8_t cs = 0;
        for (int i = 1; i < len; i++) {
            cs ^= static_cast<uint8_t>(buf[i]);
        }

        char out[280];
        snprintf(out, sizeof(out), "%s*%02X\r\n", buf, cs);
        return std::string(out);
    }

    /// Format as compact JSON.
    static std::string formatJson(const NavigationState& state) {
        char buf[512];
        snprintf(buf, sizeof(buf),
            "{\"lat\":%.7f,\"lon\":%.7f,\"alt\":%.1f,"
            "\"spd\":%.2f,\"hdg\":%.1f,\"mode\":%d,"
            "\"hAcc\":%.1f,\"stat\":%d,\"align\":%d,"
            "\"road\":%lld,\"lvl\":%d}",
            state.latitude_deg,
            state.longitude_deg,
            state.altitude_m,
            state.speed_mps,
            state.heading_deg,
            static_cast<int>(state.mode),
            state.horizontal_accuracy_m,
            state.is_stationary ? 1 : 0,
            state.alignment_quality,
            static_cast<long long>(state.matched_road_id),
            state.road_level
        );
        return std::string(buf);
    }

    /// Format as binary packet.
    static BinaryPacket formatBinary(const NavigationState& state) {
        BinaryPacket pkt;
        pkt.mode = static_cast<uint8_t>(state.mode);
        pkt.lat_deg_e7 = static_cast<int32_t>(state.latitude_deg * 1e7);
        pkt.lon_deg_e7 = static_cast<int32_t>(state.longitude_deg * 1e7);
        pkt.alt_dm = static_cast<int16_t>(state.altitude_m * 10.0);
        pkt.speed_cmps = static_cast<uint16_t>(state.speed_mps * 100.0);
        pkt.heading_cdeg = static_cast<uint16_t>(state.heading_deg * 100.0);
        pkt.h_accuracy_cm = static_cast<uint16_t>(
            std::min(state.horizontal_accuracy_m * 100.0, 65535.0));
        pkt.gnss_integrity = static_cast<uint8_t>(state.gnss_confidence > 0.5 ? 1 : 0);
        pkt.alignment_quality = state.alignment_quality;
        pkt.is_stationary = state.is_stationary ? 1 : 0;
        pkt.timestamp_ns = state.timestamp_ns;
        pkt.matched_road_id = static_cast<int32_t>(state.matched_road_id);
        pkt.road_level = static_cast<uint8_t>(state.road_level + 128);  // Signed → unsigned

        // Checksum
        const auto* bytes = reinterpret_cast<const uint8_t*>(&pkt);
        uint8_t cs = 0;
        for (size_t i = 0; i < sizeof(pkt) - 1; i++) {
            cs ^= bytes[i];
        }
        pkt.checksum = cs;

        return pkt;
    }

private:
    void emitBinary(const NavigationState& state) {
        auto pkt = formatBinary(state);
        if (send_cb_) {
            send_cb_(reinterpret_cast<const uint8_t*>(&pkt), sizeof(pkt));
        }
    }

    void emitNmea(const NavigationState& state) {
        auto nmea = formatNmea(state);
        if (send_cb_) {
            send_cb_(reinterpret_cast<const uint8_t*>(nmea.c_str()), nmea.size());
        }
    }

    void emitJson(const NavigationState& state) {
        auto json = formatJson(state);
        if (send_cb_) {
            send_cb_(reinterpret_cast<const uint8_t*>(json.c_str()), json.size());
        }
    }

    EdgeConfig config_;
    SendCallback send_cb_;
};

}  // namespace edge
}  // namespace idr
