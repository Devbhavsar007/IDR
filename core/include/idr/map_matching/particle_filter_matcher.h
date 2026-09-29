// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Particle Filter Map Matcher (PF-MM)
// Implements multi-hypothesis particle filtering over road networks (Blueprint v1.0 Appendix B.3).
// Excels in dense urban canyons, multi-level flyovers/underpasses, and parallel corridor bifurcations.

#pragma once

#include "idr/map_graph/road_graph.h"
#include "idr/map_matching/hmm_matcher.h"
#include "idr/math_utils/quat_utils.h"
#include "idr/types/common.h"

#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_map>
#include <vector>

namespace idr {
namespace maps {

/// Particle state on road network
struct MapParticle {
    int64_t road_id = -1;
    size_t segment_idx = 0;
    double fraction = 0.0;              // [0, 1] along segment
    Vec3d position_enu = Vec3d::Zero();
    double heading_rad = 0.0;
    int level = 0;
    double weight = 1.0;
};

/// Configuration for Particle Filter Map Matcher
struct ParticleFilterConfig {
    int num_particles = 100;
    double search_radius_m = 60.0;
    double sigma_distance_m = 10.0;
    double sigma_heading_rad = 0.35;
    double sigma_altitude_m = 4.0;
    double resample_neff_ratio = 0.50; // Resample when Neff < num_particles * ratio
    double speed_noise_std = 0.5;      // m/s velocity dispersion
};

/// Multi-hypothesis Particle Filter Map Matcher
class ParticleFilterMapMatcher {
public:
    explicit ParticleFilterMapMatcher(const RoadGraph& graph,
                                     const ParticleFilterConfig& config = {})
        : graph_(graph), config_(config), rng_(42) {
        particles_.resize(static_cast<size_t>(config_.num_particles));
    }

    /// Update particles with motion propagation, measurement likelihood, and resampling.
    MapMatchResult update(const Vec3d& position_enu,
                          double heading_rad,
                          double speed_mps,
                          double dt_s,
                          double baro_altitude_m = std::numeric_limits<double>::quiet_NaN()) {
        if (!initialized_) {
            initializeParticles(position_enu, heading_rad);
            if (!initialized_) return {};
        }

        // 1. Prediction / Propagation along road topology
        predict(speed_mps, dt_s);

        // 2. Likelihood update
        updateWeights(position_enu, heading_rad, baro_altitude_m);

        // 3. Normalize weights and compute entropy
        normalizeAndAssess();

        // 4. Extract dominant hypothesis result
        MapMatchResult result = extractBestMatch();

        // 5. Resample if effective sample size drops
        if (effective_particles_ < static_cast<double>(config_.num_particles) * config_.resample_neff_ratio) {
            systematicResample();
        }

        return result;
    }

    double entropy() const { return entropy_; }
    double effectiveParticles() const { return effective_particles_; }
    const std::vector<MapParticle>& particles() const { return particles_; }

    void reset() {
        initialized_ = false;
        entropy_ = 0.0;
        effective_particles_ = 0.0;
        particles_.clear();
        particles_.resize(static_cast<size_t>(config_.num_particles));
    }

private:
    void initializeParticles(const Vec3d& position_enu, [[maybe_unused]] double heading_rad) {
        auto candidates = graph_.findNearest(position_enu, config_.search_radius_m, 10);
        if (candidates.empty()) return;

        double initial_weight = 1.0 / static_cast<double>(config_.num_particles);
        std::uniform_real_distribution<double> dist_noise(-3.0, 3.0);

        for (size_t i = 0; i < particles_.size(); i++) {
            const auto& cand = candidates[i % candidates.size()];
            particles_[i].road_id = cand.road_id;
            particles_[i].segment_idx = cand.segment_idx;
            particles_[i].fraction = std::clamp(cand.fraction + dist_noise(rng_) * 0.01, 0.0, 1.0);
            particles_[i].position_enu = cand.closest_point;
            particles_[i].heading_rad = cand.heading_rad;
            particles_[i].level = cand.level;
            particles_[i].weight = initial_weight;
        }

        initialized_ = true;
    }

    void predict(double speed_mps, double dt_s) {
        std::normal_distribution<double> speed_noise(0.0, config_.speed_noise_std);

        for (auto& p : particles_) {
            const RoadSegment* road = graph_.findRoad(p.road_id);
            if (!road || road->vertices.size() < 2) continue;

            double step_dist = std::max(0.0, (speed_mps + speed_noise(rng_)) * dt_s);
            if (step_dist <= 1e-4) continue;

            size_t seg_idx = p.segment_idx;
            if (seg_idx + 1 >= road->vertices.size()) {
                seg_idx = road->vertices.size() - 2;
            }

            Vec3d v0 = road->vertices[seg_idx];
            Vec3d v1 = road->vertices[seg_idx + 1];
            double seg_len = (v1 - v0).norm();

            if (seg_len < 1e-3) continue;

            double current_dist = p.fraction * seg_len;
            double new_dist = current_dist + step_dist;

            while (new_dist > seg_len && seg_idx + 2 < road->vertices.size()) {
                new_dist -= seg_len;
                seg_idx++;
                v0 = road->vertices[seg_idx];
                v1 = road->vertices[seg_idx + 1];
                seg_len = (v1 - v0).norm();
            }

            p.segment_idx = seg_idx;
            p.fraction = std::clamp(new_dist / std::max(seg_len, 1e-3), 0.0, 1.0);
            p.position_enu = v0 + (v1 - v0) * p.fraction;
            if (seg_idx < road->headings.size()) {
                p.heading_rad = road->headings[seg_idx];
            }
        }
    }

    void updateWeights(const Vec3d& pos_enu, double heading_rad, double baro_altitude_m) {
        double dist_denom = 2.0 * config_.sigma_distance_m * config_.sigma_distance_m;
        double head_denom = 2.0 * config_.sigma_heading_rad * config_.sigma_heading_rad;
        double alt_denom = 2.0 * config_.sigma_altitude_m * config_.sigma_altitude_m;

        for (auto& p : particles_) {
            // 2D distance likelihood
            double dx = p.position_enu.x() - pos_enu.x();
            double dy = p.position_enu.y() - pos_enu.y();
            double dist_sq = dx * dx + dy * dy;
            double l_dist = std::exp(-dist_sq / dist_denom);

            // Heading likelihood
            double d_head = std::abs(angle::angleDifference(p.heading_rad, heading_rad));
            double l_head = std::exp(-(d_head * d_head) / head_denom);

            // Barometric level likelihood (if available)
            double l_alt = 1.0;
            if (std::isfinite(baro_altitude_m)) {
                double expected_z = p.position_enu.z();
                if (p.level != 0) {
                    expected_z += static_cast<double>(p.level) * 5.5; // ~5.5m per layer
                }
                double dz = baro_altitude_m - expected_z;
                l_alt = std::exp(-(dz * dz) / alt_denom);
            }

            p.weight *= (l_dist * l_head * l_alt) + 1e-12;
        }
    }

    void normalizeAndAssess() {
        double total_weight = 0.0;
        for (const auto& p : particles_) total_weight += p.weight;

        if (total_weight <= 1e-15) {
            double uniform_w = 1.0 / static_cast<double>(particles_.size());
            for (auto& p : particles_) p.weight = uniform_w;
            effective_particles_ = static_cast<double>(particles_.size());
            entropy_ = std::log(static_cast<double>(particles_.size()));
            return;
        }

        double sum_sq_w = 0.0;
        double ent = 0.0;
        for (auto& p : particles_) {
            p.weight /= total_weight;
            sum_sq_w += p.weight * p.weight;
            if (p.weight > 1e-12) {
                ent -= p.weight * std::log(p.weight);
            }
        }

        effective_particles_ = (sum_sq_w > 0.0) ? (1.0 / sum_sq_w) : 1.0;
        entropy_ = ent;
    }

    MapMatchResult extractBestMatch() const {
        if (particles_.empty()) return {};

        std::unordered_map<int64_t, double> road_weights;
        for (const auto& p : particles_) {
            road_weights[p.road_id] += p.weight;
        }

        int64_t best_road = -1;
        double max_w = -1.0;
        for (const auto& [r_id, w] : road_weights) {
            if (w > max_w) {
                max_w = w;
                best_road = r_id;
            }
        }

        Vec3d weighted_pos = Vec3d::Zero();
        double w_sum = 0.0;
        double road_heading = 0.0;
        int road_level = 0;

        for (const auto& p : particles_) {
            if (p.road_id == best_road) {
                weighted_pos += p.position_enu * p.weight;
                w_sum += p.weight;
                road_heading = p.heading_rad;
                road_level = p.level;
            }
        }

        MapMatchResult res;
        res.road_id = best_road;
        res.confidence = std::clamp(max_w, 0.0, 1.0);
        if (w_sum > 0.0) {
            res.snapped_position = weighted_pos / w_sum;
        }
        res.road_heading_rad = road_heading;
        res.road_level = road_level;
        res.valid = (res.confidence >= 0.25);
        return res;
    }

    void systematicResample() {
        size_t n = particles_.size();
        std::vector<MapParticle> new_particles(n);

        std::uniform_real_distribution<double> u01(0.0, 1.0 / static_cast<double>(n));
        double r = u01(rng_);
        double c = particles_[0].weight;
        size_t i = 0;

        for (size_t m = 0; m < n; m++) {
            double u = r + static_cast<double>(m) / static_cast<double>(n);
            while (u > c && i + 1 < n) {
                i++;
                c += particles_[i].weight;
            }
            new_particles[m] = particles_[i];
            new_particles[m].weight = 1.0 / static_cast<double>(n);
        }

        particles_ = std::move(new_particles);
    }

    const RoadGraph& graph_;
    ParticleFilterConfig config_;
    std::mt19937 rng_;
    std::vector<MapParticle> particles_;
    bool initialized_ = false;
    double effective_particles_ = 0.0;
    double entropy_ = 0.0;
};

} // namespace maps
} // namespace idr
