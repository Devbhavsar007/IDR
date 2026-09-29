// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Road Graph — spatial index and road network representation
// for map matching and road-level disambiguation.
//
// Uses a simple grid-based spatial index for O(1) nearest-road queries.
// Road segments are stored as polylines with metadata.

#pragma once

#include "idr/types/common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace idr {
namespace maps {

/// A single road segment in the road graph.
struct RoadSegment {
    int64_t id = -1;
    std::string name;

    /// Polyline vertices in ENU (meters) relative to map origin
    std::vector<Vec3d> vertices;

    /// Road metadata
    int level = 0;           ///< 0=ground, 1=flyover, -1=underpass, 2=bridge
    double speed_limit_mps = 16.7;  ///< ~60 km/h default
    int lanes = 2;
    bool oneway = false;
    double width_m = 7.0;

    /// Heading at each vertex (radians, CW from North)
    std::vector<double> headings;

    /// Precompute headings from vertex geometry.
    void computeHeadings() {
        headings.resize(vertices.size(), 0.0);
        for (size_t i = 0; i + 1 < vertices.size(); i++) {
            Vec3d diff = vertices[i + 1] - vertices[i];
            headings[i] = std::atan2(diff.x(), diff.y());  // atan2(East, North)
        }
        if (vertices.size() > 1) {
            headings.back() = headings[headings.size() - 2];
        }
    }
};

/// Point-to-segment projection result.
struct SegmentProjection {
    int64_t road_id = -1;
    double distance_m = std::numeric_limits<double>::max();  ///< Perpendicular distance
    Vec3d closest_point = Vec3d::Zero();  ///< Closest point on segment
    double heading_rad = 0.0;             ///< Road heading at closest point
    int level = 0;
    size_t segment_idx = 0;               ///< Index of the segment within the polyline
    double fraction = 0.0;                ///< [0,1] position along the segment
};

/// Grid-based spatial index for fast nearest-road queries.
class RoadGraph {
public:
    /// Construct with grid cell size in meters.
    explicit RoadGraph(double cell_size_m = 50.0)
        : cell_size_(cell_size_m) {}

    /// Add a road segment to the graph.
    void addRoad(const RoadSegment& road) {
        roads_.push_back(road);
        roads_.back().computeHeadings();

        // Index all cells this road passes through
        int64_t road_idx = static_cast<int64_t>(roads_.size()) - 1;
        if (road.vertices.size() == 1) {
            auto key = cellKey(road.vertices[0].x(), road.vertices[0].y());
            grid_[key].push_back(road_idx);
        } else {
            for (size_t i = 0; i + 1 < road.vertices.size(); i++) {
                const auto& a = road.vertices[i];
                const auto& b = road.vertices[i + 1];
                double seg_len = (b - a).norm();
                int steps = std::max(1, static_cast<int>(std::ceil(seg_len / (cell_size_ * 0.5))));
                for (int s = 0; s <= steps; s++) {
                    double frac = static_cast<double>(s) / static_cast<double>(steps);
                    Vec3d pt = a + (b - a) * frac;
                    auto key = cellKey(pt.x(), pt.y());
                    auto& list = grid_[key];
                    if (list.empty() || list.back() != road_idx) {
                        list.push_back(road_idx);
                    }
                }
            }
        }
    }

    /// Find nearest road candidates within search_radius_m.
    /// Returns up to max_candidates projections, sorted by distance.
    std::vector<SegmentProjection> findNearest(
        const Vec3d& position_enu,
        double search_radius_m = 50.0,
        int max_candidates = 10) const {

        std::vector<SegmentProjection> candidates;

        // Search grid cells within radius
        int r = static_cast<int>(std::ceil(search_radius_m / cell_size_));
        int cx = static_cast<int>(std::floor(position_enu.x() / cell_size_));
        int cy = static_cast<int>(std::floor(position_enu.y() / cell_size_));

        // Collect unique road indices from nearby cells
        std::vector<bool> visited(roads_.size(), false);

        for (int dx = -r; dx <= r; dx++) {
            for (int dy = -r; dy <= r; dy++) {
                auto key = cellKeyFromGrid(cx + dx, cy + dy);
                auto it = grid_.find(key);
                if (it == grid_.end()) continue;

                for (int64_t road_idx : it->second) {
                    size_t idx = static_cast<size_t>(road_idx);
                    if (visited[idx]) continue;
                    visited[idx] = true;

                    auto proj = projectToRoad(position_enu, roads_[idx]);
                    if (proj.distance_m <= search_radius_m) {
                        candidates.push_back(proj);
                    }
                }
            }
        }

        // Sort by distance
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) {
                      return a.distance_m < b.distance_m;
                  });

        // Limit candidates
        if (static_cast<int>(candidates.size()) > max_candidates) {
            candidates.resize(static_cast<size_t>(max_candidates));
        }

        return candidates;
    }

    size_t roadCount() const { return roads_.size(); }
    size_t numRoads() const { return roads_.size(); }
    const RoadSegment& road(size_t idx) const { return roads_[idx]; }

private:
    using GridKey = int64_t;

    GridKey cellKey(double x, double y) const {
        int cx = static_cast<int>(std::floor(x / cell_size_));
        int cy = static_cast<int>(std::floor(y / cell_size_));
        return cellKeyFromGrid(cx, cy);
    }

    static GridKey cellKeyFromGrid(int cx, int cy) {
        // Pack two 32-bit ints into one 64-bit key
        return (static_cast<int64_t>(cx) << 32) | (static_cast<int64_t>(cy) & 0xFFFFFFFF);
    }

    /// Project a point onto a road's polyline, returning the closest result.
    static SegmentProjection projectToRoad(const Vec3d& point,
                                           const RoadSegment& road) {
        SegmentProjection best;
        best.road_id = road.id;
        best.level = road.level;

        for (size_t i = 0; i + 1 < road.vertices.size(); i++) {
            const Vec3d& a = road.vertices[i];
            const Vec3d& b = road.vertices[i + 1];

            // Project point onto line segment a→b (2D)
            Vec3d ab = b - a;
            double ab_len2 = ab.x() * ab.x() + ab.y() * ab.y();
            if (ab_len2 < 1e-10) continue;

            double t = ((point.x() - a.x()) * ab.x() +
                        (point.y() - a.y()) * ab.y()) / ab_len2;
            t = std::clamp(t, 0.0, 1.0);

            Vec3d closest = a + ab * t;
            double dist = std::sqrt(
                (point.x() - closest.x()) * (point.x() - closest.x()) +
                (point.y() - closest.y()) * (point.y() - closest.y())
            );

            if (dist < best.distance_m) {
                best.distance_m = dist;
                best.closest_point = closest;
                best.segment_idx = i;
                best.fraction = t;

                // Interpolate heading
                if (i < road.headings.size()) {
                    best.heading_rad = road.headings[i];
                }
            }
        }

        return best;
    }

    double cell_size_;
    std::vector<RoadSegment> roads_;
    std::unordered_map<GridKey, std::vector<int64_t>> grid_;
};

}  // namespace maps
}  // namespace idr
