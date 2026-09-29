// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// Unit tests for Multi-Hypothesis Particle Filter Map Matcher

#include <gtest/gtest.h>
#include "idr/map_matching/particle_filter_matcher.h"
#include "idr/map_graph/road_graph.h"

using namespace idr;
using namespace idr::maps;

class ParticleFilterMatcherTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create a 2-road graph:
        // Road 1: Surface road (level 0), heading East
        RoadSegment surface_road;
        surface_road.id = 101;
        surface_road.name = "Surface Ring Road";
        surface_road.level = 0;
        surface_road.vertices = {
            Vec3d(0.0, 0.0, 0.0),
            Vec3d(100.0, 0.0, 0.0),
            Vec3d(200.0, 0.0, 0.0)
        };
        surface_road.computeHeadings();
        graph_.addRoad(surface_road);

        // Road 2: Elevated flyover directly above (level 1, +6.0m), heading East
        RoadSegment flyover;
        flyover.id = 102;
        flyover.name = "Barapullah Elevated Flyover";
        flyover.level = 1;
        flyover.vertices = {
            Vec3d(0.0, 0.0, 6.0),
            Vec3d(100.0, 0.0, 6.0),
            Vec3d(200.0, 0.0, 6.0)
        };
        flyover.computeHeadings();
        graph_.addRoad(flyover);

        // Road 3: Cross road heading North (level 0)
        RoadSegment cross_road;
        cross_road.id = 103;
        cross_road.name = "North-South Link";
        cross_road.level = 0;
        cross_road.vertices = {
            Vec3d(50.0, -50.0, 0.0),
            Vec3d(50.0, 50.0, 0.0)
        };
        cross_road.computeHeadings();
        graph_.addRoad(cross_road);
    }

    RoadGraph graph_;
};

TEST_F(ParticleFilterMatcherTest, InitializesAndPopulatesParticles) {
    ParticleFilterConfig cfg;
    cfg.num_particles = 80;
    ParticleFilterMapMatcher matcher(graph_, cfg);

    Vec3d pos(20.0, 1.0, 0.0);
    double heading_east = constants::kPi / 2.0;
    auto match = matcher.update(pos, heading_east, 10.0, 0.1);

    EXPECT_TRUE(match.valid);
    EXPECT_GT(match.confidence, 0.3);
    EXPECT_EQ(matcher.particles().size(), 80);
}

TEST_F(ParticleFilterMatcherTest, DisambiguatesElevatedFlyoverWithBarometer) {
    ParticleFilterConfig cfg;
    cfg.num_particles = 100;
    ParticleFilterMapMatcher matcher(graph_, cfg);

    Vec3d pos(50.0, 0.0, 6.0); // Exactly on the flyover line
    double heading_east = constants::kPi / 2.0;

    // Feed with barometric altitude corresponding to flyover (+6.0m)
    MapMatchResult match;
    for (int i = 0; i < 15; i++) {
        match = matcher.update(pos + Vec3d(i * 1.5, 0.0, 0.0), heading_east, 15.0, 0.1, 6.2);
    }

    EXPECT_TRUE(match.valid);
    EXPECT_EQ(match.road_id, 102); // Must be the flyover!
    EXPECT_EQ(match.road_level, 1);
}

TEST_F(ParticleFilterMatcherTest, HeadingDisambiguatesOrthogonalRoads) {
    ParticleFilterConfig cfg;
    cfg.num_particles = 100;
    ParticleFilterMapMatcher matcher(graph_, cfg);

    // Position at intersection (50, 0), heading North (yaw = 0.0)
    Vec3d pos(50.0, 0.0, 0.0);
    double heading_north = 0.0;

    MapMatchResult match;
    for (int i = 0; i < 10; i++) {
        match = matcher.update(pos + Vec3d(0.0, i * 1.0, 0.0), heading_north, 10.0, 0.1, 0.0);
    }

    EXPECT_TRUE(match.valid);
    EXPECT_EQ(match.road_id, 103); // Must select North-South Link
}
