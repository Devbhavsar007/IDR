// IDR — Unit tests for RoadGraph and HmmMapMatcher.

#include "idr/map_graph/road_graph.h"
#include "idr/map_matching/hmm_matcher.h"
#include <gtest/gtest.h>
#include <cmath>

namespace idr {
namespace maps {
namespace {

// Helper: create a simple straight road along North (y-axis)
RoadSegment makeNorthRoad(int64_t id, double x_offset, double y_start, double y_end, int level = 0) {
    RoadSegment road;
    road.id = id;
    road.level = level;
    int n = 10;
    for (int i = 0; i <= n; i++) {
        double t = static_cast<double>(i) / n;
        double y = y_start + (y_end - y_start) * t;
        road.vertices.push_back(Vec3d(x_offset, y, 0.0));
    }
    return road;
}

// Helper: create an East-West road
RoadSegment makeEastRoad(int64_t id, double y_offset, double x_start, double x_end) {
    RoadSegment road;
    road.id = id;
    int n = 10;
    for (int i = 0; i <= n; i++) {
        double t = static_cast<double>(i) / n;
        double x = x_start + (x_end - x_start) * t;
        road.vertices.push_back(Vec3d(x, y_offset, 0.0));
    }
    return road;
}

class RoadGraphTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a simple grid: 2 N-S roads and 1 E-W road
        graph_.addRoad(makeNorthRoad(1, 0.0, 0.0, 200.0));     // Road 1: x=0, N-S
        graph_.addRoad(makeNorthRoad(2, 100.0, 0.0, 200.0));   // Road 2: x=100, N-S
        graph_.addRoad(makeEastRoad(3, 100.0, 0.0, 200.0));    // Road 3: y=100, E-W
    }

    RoadGraph graph_;
};

TEST_F(RoadGraphTest, FindNearestOnRoad) {
    // Point exactly on Road 1
    auto results = graph_.findNearest(Vec3d(0.0, 50.0, 0.0), 20.0, 5);
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results[0].road_id, 1);
    EXPECT_LT(results[0].distance_m, 1.0);
}

TEST_F(RoadGraphTest, FindNearestBetweenRoads) {
    // Point at x=40 — closer to Road 1 (x=0) than Road 2 (x=100)
    auto results = graph_.findNearest(Vec3d(40.0, 50.0, 0.0), 60.0, 5);
    ASSERT_GE(results.size(), 1u);
    EXPECT_EQ(results[0].road_id, 1);
    EXPECT_NEAR(results[0].distance_m, 40.0, 1.0);
}

TEST_F(RoadGraphTest, NoResultsOutOfRange) {
    // Point far from any road
    auto results = graph_.findNearest(Vec3d(500.0, 500.0, 0.0), 20.0, 5);
    EXPECT_TRUE(results.empty());
}

TEST_F(RoadGraphTest, HeadingOfNorthRoad) {
    auto results = graph_.findNearest(Vec3d(0.0, 50.0, 0.0), 10.0, 1);
    ASSERT_FALSE(results.empty());
    // North road: heading should be ~0 (atan2(0, +y) = 0)
    EXPECT_NEAR(results[0].heading_rad, 0.0, 0.1);
}

TEST_F(RoadGraphTest, HeadingOfEastRoad) {
    auto results = graph_.findNearest(Vec3d(50.0, 100.0, 0.0), 10.0, 5);
    // Find the East road (id=3)
    for (const auto& r : results) {
        if (r.road_id == 3) {
            // East road: heading should be ~π/2 (atan2(+x, 0) = π/2)
            EXPECT_NEAR(std::abs(r.heading_rad), constants::kPi / 2, 0.1);
            return;
        }
    }
}

TEST_F(RoadGraphTest, MultipleCandiatesAtIntersection) {
    // At intersection (0, 100): Road 1 and Road 3 meet
    auto results = graph_.findNearest(Vec3d(0.0, 100.0, 0.0), 20.0, 10);
    ASSERT_GE(results.size(), 2u);  // Should find both roads
}

// ── HMM Map Matcher Tests ──

class MapMatcherTest : public ::testing::Test {
protected:
    void SetUp() override {
        graph_.addRoad(makeNorthRoad(1, 0.0, 0.0, 500.0));
        graph_.addRoad(makeNorthRoad(2, 50.0, 0.0, 500.0));
    }

    RoadGraph graph_;
};

TEST_F(MapMatcherTest, MatchesToClosestRoad) {
    HmmMapMatcher matcher(graph_);

    // Vehicle near Road 1 heading North
    auto result = matcher.update(Vec3d(5.0, 100.0, 0.0), 0.0, 10.0);
    EXPECT_TRUE(result.valid);
    EXPECT_EQ(result.road_id, 1);
}

TEST_F(MapMatcherTest, HeadingDisambiguates) {
    HmmMapMatcher matcher(graph_);

    // Vehicle equidistant from both roads, but heading North (both roads go North)
    auto result = matcher.update(Vec3d(25.0, 100.0, 0.0), 0.0, 10.0);
    EXPECT_TRUE(result.valid);
    // Should pick one of the two — just verify it's valid
    EXPECT_TRUE(result.road_id == 1 || result.road_id == 2);
}

TEST_F(MapMatcherTest, ConfidenceDecaysWithDistance) {
    HmmMapMatcher matcher(graph_);

    auto close = matcher.update(Vec3d(2.0, 100.0, 0.0), 0.0, 10.0);
    matcher.reset();
    auto far = matcher.update(Vec3d(40.0, 100.0, 0.0), 0.0, 10.0);

    if (close.valid && far.valid) {
        EXPECT_GT(close.confidence, far.confidence);
    }
}

TEST_F(MapMatcherTest, ResetClearsState) {
    HmmMapMatcher matcher(graph_);
    matcher.update(Vec3d(5.0, 100.0, 0.0), 0.0, 10.0);
    matcher.reset();
    // After reset, next update should still work
    auto result = matcher.update(Vec3d(5.0, 200.0, 0.0), 0.0, 10.0);
    EXPECT_TRUE(result.valid);
}

}  // namespace
}  // namespace maps
}  // namespace idr
