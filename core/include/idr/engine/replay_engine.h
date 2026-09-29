// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Replay Engine — feeds recorded sensor data through IdrEngine
// for offline testing, benchmarking, and dataset evaluation.
//
// Supports:
//   - CSV input (IO-VNBD format and generic IMU+GNSS CSVs)
//   - Ground-truth comparison
//   - Metric logging (ATE, RTE, drift rate)
//   - Speed control (real-time, fast-forward, step)

#pragma once

#include "idr/engine/idr_engine.h"
#include "idr/math_utils/geo_utils.h"
#include "idr/types/common.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace idr {

/// A single record from a replay file (all sensor data + ground truth)
struct ReplayRecord {
    double timestamp_sec = 0.0;

    // IMU
    Vec3d accel = Vec3d::Zero();    ///< m/s²
    Vec3d gyro = Vec3d::Zero();     ///< rad/s
    Vec3d gravity = Vec3d::Zero();  ///< m/s² (from Android TYPE_GRAVITY)

    // GNSS (may be NaN if no fix)
    double lat_deg = std::numeric_limits<double>::quiet_NaN();
    double lon_deg = std::numeric_limits<double>::quiet_NaN();
    double alt_m = 0.0;
    float speed_mps = 0.0f;
    float bearing_deg = 0.0f;
    float h_accuracy_m = 0.0f;

    // Ground truth (for evaluation)
    double gt_lat_deg = std::numeric_limits<double>::quiet_NaN();
    double gt_lon_deg = std::numeric_limits<double>::quiet_NaN();
    double gt_speed_mps = 0.0;

    bool hasGnss() const { return std::isfinite(lat_deg) && std::isfinite(lon_deg); }
    bool hasGroundTruth() const { return std::isfinite(gt_lat_deg); }
};

/// Accumulated error metrics
struct ReplayMetrics {
    /// Absolute Trajectory Error (m) — RMS of position errors
    double ate_rms_m = 0.0;

    /// Maximum position error (m)
    double max_error_m = 0.0;

    /// Drift rate (%/km) — position error growth per km traveled
    double drift_pct_per_km = 0.0;

    /// 50th/95th percentile position errors (m)
    double p50_error_m = 0.0;
    double p95_error_m = 0.0;

    /// Total distance traveled (m)
    double total_distance_m = 0.0;

    /// Total duration (s)
    double total_duration_sec = 0.0;

    /// Number of GNSS outage segments
    int gnss_outage_count = 0;

    /// Longest GNSS outage (s)
    double max_gnss_outage_sec = 0.0;

    /// Navigation mode distribution (%)
    double pct_gnss_ins = 0.0;
    double pct_dead_reckoning = 0.0;
    double pct_degraded = 0.0;

    /// Number of ground-truth samples compared
    int gt_samples = 0;
};

/// Replay Engine.
class ReplayEngine {
public:
    using MetricsCallback = std::function<void(const ReplayMetrics&)>;

    explicit ReplayEngine(const EngineConfig& config = {})
        : engine_(config) {}

    /// Load replay data from a CSV file.
    /// Expected columns (flexible ordering):
    ///   timestamp, acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z,
    ///   [grav_x, grav_y, grav_z], [lat, lon, alt, speed, bearing, accuracy]
    bool loadCsv(const std::string& path) {
        std::ifstream file(path);
        if (!file.is_open()) {
            spdlog::error("Replay: cannot open {}", path);
            return false;
        }

        records_.clear();

        // Read header
        std::string header;
        std::getline(file, header);
        auto col_indices = parseHeader(header);

        // Read data
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            auto fields = splitCsv(line);
            if (fields.size() < 7) continue;  // Need at least timestamp + accel + gyro

            ReplayRecord rec;
            rec.timestamp_sec = getField(fields, col_indices, "timestamp", 0.0);
            rec.accel.x() = getField(fields, col_indices, "acc_x", 0.0);
            rec.accel.y() = getField(fields, col_indices, "acc_y", 0.0);
            rec.accel.z() = getField(fields, col_indices, "acc_z", 0.0);
            rec.gyro.x() = getField(fields, col_indices, "gyro_x", 0.0);
            rec.gyro.y() = getField(fields, col_indices, "gyro_y", 0.0);
            rec.gyro.z() = getField(fields, col_indices, "gyro_z", 0.0);

            // Optional: gravity
            rec.gravity.x() = getField(fields, col_indices, "grav_x", 0.0);
            rec.gravity.y() = getField(fields, col_indices, "grav_y", 0.0);
            rec.gravity.z() = getField(fields, col_indices, "grav_z", 0.0);

            // Optional: GNSS
            rec.lat_deg = getField(fields, col_indices, "lat",
                                   std::numeric_limits<double>::quiet_NaN());
            rec.lon_deg = getField(fields, col_indices, "lon",
                                   std::numeric_limits<double>::quiet_NaN());
            rec.alt_m = getField(fields, col_indices, "alt", 0.0);
            rec.speed_mps = static_cast<float>(getField(fields, col_indices, "speed", 0.0));
            rec.bearing_deg = static_cast<float>(getField(fields, col_indices, "bearing", 0.0));
            rec.h_accuracy_m = static_cast<float>(getField(fields, col_indices, "accuracy", 50.0));

            records_.push_back(rec);
        }

        spdlog::info("Replay: loaded {} records from {}", records_.size(), path);
        return !records_.empty();
    }

    /// Run the replay (process all records through IdrEngine).
    /// @param output_callback  Called for each NavigationState output
    /// @param gnss_mask_start  Start of simulated GNSS outage (seconds)
    /// @param gnss_mask_end    End of simulated GNSS outage (seconds)
    ReplayMetrics run(IdrEngine::OutputCallback output_callback = nullptr,
                      double gnss_mask_start = -1.0,
                      double gnss_mask_end = -1.0) {
        if (records_.empty()) {
            spdlog::error("Replay: no records loaded");
            return {};
        }

        engine_.setOutputCallback(output_callback);

        ReplayMetrics metrics;
        double start_time = records_.front().timestamp_sec;
        double last_gnss_time = start_time;
        bool in_outage = false;
        std::vector<double> position_errors;

        Vec3d prev_gt_enu = Vec3d::Zero();
        bool first_gt = true;

        // Reference origin for ground truth ENU
        double ref_lat = 0.0, ref_lon = 0.0, ref_alt = 0.0;
        bool ref_set = false;

        for (size_t i = 0; i < records_.size(); i++) {
            const auto& rec = records_[i];
            double elapsed = rec.timestamp_sec - start_time;

            // ── Feed IMU ──
            ImuSample imu;
            imu.timestamp_ns = secToNs(rec.timestamp_sec);
            imu.accel = rec.accel;
            imu.gyro = rec.gyro;
            imu.gravity = rec.gravity;
            engine_.processImu(imu);

            // ── Feed GNSS (with optional outage masking) ──
            if (rec.hasGnss()) {
                bool masked = (gnss_mask_start >= 0.0 && gnss_mask_end >= 0.0 &&
                               elapsed >= gnss_mask_start && elapsed < gnss_mask_end);

                if (!masked) {
                    GnssMeasurement gnss;
                    gnss.timestamp_ns = secToNs(rec.timestamp_sec);
                    gnss.latitude_deg = rec.lat_deg;
                    gnss.longitude_deg = rec.lon_deg;
                    gnss.altitude_m = static_cast<float>(rec.alt_m);
                    gnss.speed_mps = rec.speed_mps;
                    gnss.bearing_deg = rec.bearing_deg;
                    gnss.horizontal_accuracy_m = rec.h_accuracy_m;
                    gnss.vertical_accuracy_m = rec.h_accuracy_m * 2.0f;
                    gnss.fix_type = GnssFixType::FIX_3D;
                    gnss.satellite_count = 10;
                    engine_.processGnss(gnss);

                    last_gnss_time = rec.timestamp_sec;
                    in_outage = false;
                } else {
                    if (!in_outage) {
                        metrics.gnss_outage_count++;
                        in_outage = true;
                    }
                }
            }

            // ── Ground truth evaluation ──
            if (rec.hasGroundTruth() && engine_.isInitialized()) {
                if (!ref_set) {
                    ref_lat = rec.gt_lat_deg;
                    ref_lon = rec.gt_lon_deg;
                    ref_set = true;
                }

                Vec3d gt_enu = geo::llaToEnu(rec.gt_lat_deg, rec.gt_lon_deg, 0.0,
                                             ref_lat, ref_lon, ref_alt);

                // Distance traveled
                if (!first_gt) {
                    metrics.total_distance_m += (gt_enu - prev_gt_enu).head<2>().norm();
                }
                prev_gt_enu = gt_enu;
                first_gt = false;

                // Position error
                const auto& nav = engine_.lastOutput();
                Vec3d est_enu = geo::llaToEnu(nav.latitude_deg, nav.longitude_deg, 0.0,
                                              ref_lat, ref_lon, ref_alt);

                double error_2d = (est_enu.head<2>() - gt_enu.head<2>()).norm();
                position_errors.push_back(error_2d);
                metrics.gt_samples++;
            }

            // ── Track outage duration ──
            double current_outage = rec.timestamp_sec - last_gnss_time;
            if (current_outage > metrics.max_gnss_outage_sec) {
                metrics.max_gnss_outage_sec = current_outage;
            }
        }

        // ── Compute final metrics ──
        metrics.total_duration_sec = records_.back().timestamp_sec - start_time;

        if (!position_errors.empty()) {
            // ATE RMS
            double sum_sq = 0.0;
            for (double e : position_errors) sum_sq += e * e;
            metrics.ate_rms_m = std::sqrt(sum_sq / static_cast<double>(position_errors.size()));

            // Max
            metrics.max_error_m = *std::max_element(position_errors.begin(),
                                                     position_errors.end());

            // Percentiles
            std::sort(position_errors.begin(), position_errors.end());
            int n = static_cast<int>(position_errors.size());
            metrics.p50_error_m = position_errors[static_cast<size_t>(n * 0.50)];
            metrics.p95_error_m = position_errors[static_cast<size_t>(std::min(n - 1, static_cast<int>(n * 0.95)))];

            // Drift rate
            if (metrics.total_distance_m > 100.0) {
                metrics.drift_pct_per_km =
                    (metrics.ate_rms_m / metrics.total_distance_m) * 1000.0 * 100.0;
            }
        }

        spdlog::info("Replay complete: {:.0f}s, {:.1f}km, ATE={:.1f}m, drift={:.1f}%/km",
                     metrics.total_duration_sec,
                     metrics.total_distance_m / 1000.0,
                     metrics.ate_rms_m,
                     metrics.drift_pct_per_km);

        return metrics;
    }

    size_t recordCount() const { return records_.size(); }

private:
    using ColMap = std::unordered_map<std::string, int>;

    ColMap parseHeader(const std::string& header) {
        ColMap map;
        auto fields = splitCsv(header);
        for (int i = 0; i < static_cast<int>(fields.size()); i++) {
            std::string name = fields[static_cast<size_t>(i)];
            // Trim whitespace
            name.erase(0, name.find_first_not_of(" \t\r\n"));
            name.erase(name.find_last_not_of(" \t\r\n") + 1);
            // Lowercase
            std::transform(name.begin(), name.end(), name.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            map[name] = i;
        }
        return map;
    }

    static std::vector<std::string> splitCsv(const std::string& line) {
        std::vector<std::string> result;
        std::stringstream ss(line);
        std::string field;
        while (std::getline(ss, field, ',')) {
            result.push_back(field);
        }
        return result;
    }

    double getField(const std::vector<std::string>& fields,
                    const ColMap& cols,
                    const std::string& name,
                    double default_val) {
        auto it = cols.find(name);
        if (it == cols.end() || it->second >= static_cast<int>(fields.size())) {
            return default_val;
        }
        try {
            return std::stod(fields[static_cast<size_t>(it->second)]);
        } catch (...) {
            return default_val;
        }
    }

    IdrEngine engine_;
    std::vector<ReplayRecord> records_;
};

}  // namespace idr
