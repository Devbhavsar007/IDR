// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// idr-replay: CLI tool for dataset replay, benchmarking, and GNSS outage simulation.

#include "idr/engine/replay_engine.h"
#include <spdlog/spdlog.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n\n"
              << "Options:\n"
              << "  -i, --input <path>         Input CSV dataset (required)\n"
              << "  -v, --vehicle <type>       Vehicle profile (car|bike|auto_rickshaw|bus|truck) [default: car]\n"
              << "  --outage-start <sec>       Simulate GNSS outage start time in seconds [default: disabled]\n"
              << "  --outage-end <sec>         Simulate GNSS outage end time in seconds [default: disabled]\n"
              << "  -o, --output <path>        Export trajectory to CSV\n"
              << "  -h, --help                 Show this help message\n";
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string input_path;
    std::string output_path;
    std::string vehicle_profile = "car";
    double outage_start = -1.0;
    double outage_end = -1.0;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else if ((arg == "-i" || arg == "--input") && i + 1 < argc) {
            input_path = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if ((arg == "-v" || arg == "--vehicle") && i + 1 < argc) {
            vehicle_profile = argv[++i];
        } else if (arg == "--outage-start" && i + 1 < argc) {
            outage_start = std::stod(argv[++i]);
        } else if (arg == "--outage-end" && i + 1 < argc) {
            outage_end = std::stod(argv[++i]);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    if (input_path.empty()) {
        std::cerr << "Error: input dataset CSV is required (-i <path>)\n\n";
        printUsage(argv[0]);
        return 1;
    }

    idr::EngineConfig config;
    if (vehicle_profile == "bike") config.vehicle = idr::profiles::kBike;
    else if (vehicle_profile == "auto_rickshaw") config.vehicle = idr::profiles::kAutoRickshaw;
    else if (vehicle_profile == "bus") config.vehicle = idr::profiles::kBus;
    else if (vehicle_profile == "truck") config.vehicle = idr::profiles::kTruck;
    else config.vehicle = idr::profiles::kCar;

    idr::ReplayEngine replay(config);
    if (!replay.loadCsv(input_path)) {
        std::cerr << "Error: Failed to load dataset: " << input_path << "\n";
        return 1;
    }

    std::cout << "\n=======================================================\n"
              << " IDR Benchmark Replay Engine\n"
              << " Dataset: " << input_path << " (" << replay.recordCount() << " records)\n"
              << " Vehicle: " << vehicle_profile << "\n";
    if (outage_start >= 0.0 && outage_end >= 0.0) {
        std::cout << " Simulated GNSS Outage: [" << outage_start << "s, " << outage_end << "s]\n";
    }
    std::cout << "=======================================================\n\n";

    std::ofstream out_file;
    if (!output_path.empty()) {
        out_file.open(output_path);
        if (out_file.is_open()) {
            out_file << "timestamp_ns,latitude_deg,longitude_deg,altitude_m,speed_mps,heading_deg,mode,h_acc_m\n";
        }
    }

    auto callback = [&](const idr::NavigationState& state) {
        if (out_file.is_open()) {
            out_file << state.timestamp_ns << ","
                     << std::fixed << std::setprecision(7)
                     << state.latitude_deg << ","
                     << state.longitude_deg << ","
                     << std::setprecision(2)
                     << state.altitude_m << ","
                     << state.speed_mps << ","
                     << state.heading_deg << ","
                     << static_cast<int>(state.mode) << ","
                     << state.horizontal_accuracy_m << "\n";
        }
    };

    auto metrics = replay.run(callback, outage_start, outage_end);

    std::cout << "----------------- Benchmark Results -----------------\n"
              << " Duration:           " << std::fixed << std::setprecision(1) << metrics.total_duration_sec << " s\n"
              << " Distance Traveled:  " << std::setprecision(2) << (metrics.total_distance_m / 1000.0) << " km\n"
              << " ATE Position RMS:   " << std::setprecision(2) << metrics.ate_rms_m << " m\n"
              << " Max Position Error: " << metrics.max_error_m << " m\n"
              << " Drift Rate:         " << metrics.drift_pct_per_km << " % / km\n"
              << " 50th Percentile:    " << metrics.p50_error_m << " m\n"
              << " 95th Percentile:    " << metrics.p95_error_m << " m\n"
              << " GNSS Outages:       " << metrics.gnss_outage_count << " (longest: " << metrics.max_gnss_outage_sec << " s)\n"
              << " Mode Distribution:  GNSS+INS: " << std::setprecision(1) << metrics.pct_gnss_ins << "%, "
              << "DR: " << metrics.pct_dead_reckoning << "%, "
              << "Degraded: " << metrics.pct_degraded << "%\n"
              << "-----------------------------------------------------\n\n";

    if (out_file.is_open()) {
        std::cout << "Trajectory written to: " << output_path << "\n";
    }

    return 0;
}
