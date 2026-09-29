// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// HMM Map Matcher.
//
// Uses a Hidden Markov Model (Viterbi algorithm) to match the vehicle's
// trajectory to the road network. Each road candidate is a hidden state,
// and the observations are position + heading from the EKF.
//
// Reference: Newson & Krumm, "Hidden Markov Map Matching Through Noise
//            and Sparseness" (ACM SIGSPATIAL 2009).

#pragma once

#include "idr/map_graph/road_graph.h"
#include "idr/math_utils/quat_utils.h"
#include "idr/types/common.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace idr {
namespace maps {

/// Map matching configuration.
struct MapMatchConfig {
    double search_radius_m = 50.0;
    int max_candidates = 10;

    /// Emission probability: sigma for distance (meters)
    double sigma_dist_m = 10.0;

    /// Emission probability: sigma for heading difference (radians)
    double sigma_heading_rad = 0.3;

    /// Transition probability: beta for route distance vs great-circle distance
    double beta_transition = 5.0;

    /// Weight for heading agreement in emission probability
    double heading_weight = 0.3;

    /// Minimum confidence to report a match
    double min_confidence = 0.3;

    /// Enable level disambiguation (flyover vs ground)
    bool level_disambiguation = true;

    /// Barometric altitude weight for level disambiguation
    double altitude_weight = 0.5;
};

/// Map match result for a single epoch.
struct MapMatchResult {
    int64_t road_id = -1;
    double confidence = 0.0;           ///< [0, 1]
    Vec3d snapped_position = Vec3d::Zero();  ///< Position snapped to road
    double road_heading_rad = 0.0;     ///< Heading of the matched road
    int road_level = 0;                ///< Level of the matched road
    double distance_to_road_m = 0.0;   ///< Perpendicular distance
    bool valid = false;
};

/// HMM Map Matcher (Viterbi).
class HmmMapMatcher {
public:
    explicit HmmMapMatcher(const RoadGraph& graph,
                           const MapMatchConfig& config = {})
        : graph_(graph), config_(config) {}

    /// Process a new position observation and return the best match.
    ///
    /// @param position_enu  Current EKF position in ENU (meters)
    /// @param heading_rad   Current EKF heading (radians, CW from North)
    /// @param speed_mps     Current speed (m/s) — used for context
    MapMatchResult update(const Vec3d& position_enu,
                          double heading_rad,
                          double /*speed_mps*/) {
        // ── Find candidates ──
        auto candidates = graph_.findNearest(
            position_enu, config_.search_radius_m, config_.max_candidates
        );

        if (candidates.empty()) {
            // No roads nearby — clear state
            prev_candidates_.clear();
            prev_probs_.clear();
            return {};
        }

        // ── Emission probabilities ──
        std::vector<double> emissions(candidates.size());
        for (size_t i = 0; i < candidates.size(); i++) {
            emissions[i] = emissionProb(candidates[i], heading_rad);
        }

        // ── Viterbi step ──
        std::vector<double> probs(candidates.size());

        if (prev_candidates_.empty()) {
            // First step: use emission probabilities directly
            probs = emissions;
        } else {
            // Transition + emission
            for (size_t j = 0; j < candidates.size(); j++) {
                double best_prev = -std::numeric_limits<double>::max();
                for (size_t i = 0; i < prev_candidates_.size(); i++) {
                    double trans = transitionProb(prev_candidates_[i], candidates[j]);
                    double score = prev_probs_[i] + trans;
                    best_prev = std::max(best_prev, score);
                }
                probs[j] = best_prev + emissions[j];
            }
        }

        // ── Normalize (log-space → probability) ──
        double max_prob = *std::max_element(probs.begin(), probs.end());
        double sum_exp = 0.0;
        for (double p : probs) {
            sum_exp += std::exp(p - max_prob);
        }

        // ── Find best ──
        size_t best_idx = 0;
        for (size_t i = 1; i < probs.size(); i++) {
            if (probs[i] > probs[best_idx]) best_idx = i;
        }

        double best_confidence = std::exp(probs[best_idx] - max_prob) / sum_exp;

        // ── Save state for next step ──
        prev_candidates_ = candidates;
        prev_probs_ = probs;

        // ── Build result ──
        MapMatchResult result;
        if (best_confidence >= config_.min_confidence) {
            const auto& best = candidates[best_idx];
            result.road_id = best.road_id;
            result.confidence = best_confidence;
            result.snapped_position = best.closest_point;
            result.road_heading_rad = best.heading_rad;
            result.road_level = best.level;
            result.distance_to_road_m = best.distance_m;
            result.valid = true;
        }

        return result;
    }

    /// Reset matcher state (e.g., after GNSS re-acquisition).
    void reset() {
        prev_candidates_.clear();
        prev_probs_.clear();
    }

private:
    /// Emission probability: how well does this candidate explain the observation?
    /// Based on distance + heading agreement.
    double emissionProb(const SegmentProjection& candidate,
                        double obs_heading_rad) const {
        // Distance component: Gaussian
        double dist_prob = -0.5 * (candidate.distance_m * candidate.distance_m) /
                           (config_.sigma_dist_m * config_.sigma_dist_m);

        // Heading component: von Mises-like
        double heading_diff = angle::angleDifference(obs_heading_rad,
                                                      candidate.heading_rad);
        double heading_prob = -0.5 * (heading_diff * heading_diff) /
                              (config_.sigma_heading_rad * config_.sigma_heading_rad);

        return dist_prob + config_.heading_weight * heading_prob;
    }

    /// Transition probability: how likely is moving from candidate i to j?
    /// Based on the difference between great-circle distance and route distance.
    double transitionProb(const SegmentProjection& from,
                          const SegmentProjection& to) const {
        // Great-circle distance between the two candidates' closest points
        double gc_dist = (to.closest_point - from.closest_point).head<2>().norm();

        // Route distance approximation (using closest points)
        // For accurate routing, this would use the road graph's shortest path.
        // Here we approximate: if same road, route ≈ gc; if different, add penalty.
        double route_dist = gc_dist;
        if (from.road_id != to.road_id) {
            route_dist *= 1.5;  // Penalty for road change
        }

        double diff = std::abs(gc_dist - route_dist);
        return -diff / config_.beta_transition;
    }

    const RoadGraph& graph_;
    MapMatchConfig config_;

    // Viterbi state
    std::vector<SegmentProjection> prev_candidates_;
    std::vector<double> prev_probs_;
};

}  // namespace maps
}  // namespace idr
