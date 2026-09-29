// IDR — Unit tests for OSM/GeoJSON Road Loader.

#include "idr/map_graph/osm_loader.h"
#include <gtest/gtest.h>

namespace idr {
namespace maps {
namespace {

TEST(OsmRoadLoaderTest, ParseGeoJsonValid) {
    const std::string sample_geojson = R"({
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {
            "name": "Barakhamba Road",
            "layer": 0
          },
          "geometry": {
            "type": "LineString",
            "coordinates": [
              [77.2273, 28.6297],
              [77.2273, 28.6350]
            ]
          }
        },
        {
          "type": "Feature",
          "properties": {
            "name": "Flyover Bridge",
            "layer": 1
          },
          "geometry": {
            "type": "LineString",
            "coordinates": [
              [77.2200, 28.6300],
              [77.2250, 28.6300]
            ]
          }
        }
      ]
    })";

    RoadGraph graph(50.0);
    double ref_lat = 28.6297;
    double ref_lon = 77.2273;
    double ref_alt = 216.0;

    size_t count = OsmRoadLoader::parseGeoJson(sample_geojson, ref_lat, ref_lon, ref_alt, graph);
    EXPECT_EQ(count, 2u);
    EXPECT_EQ(graph.numRoads(), 2u);

    // Query near the first road (at origin ENU = [0, 0, 0])
    auto candidates = graph.findNearest({0.0, 50.0, 0.0});
    ASSERT_FALSE(candidates.empty());
    EXPECT_EQ(candidates[0].road_id, 1);
    EXPECT_NEAR(candidates[0].distance_m, 0.0, 1.0);
}

TEST(OsmRoadLoaderTest, MalformedGeoJsonHandled) {
    RoadGraph graph(50.0);
    size_t count = OsmRoadLoader::parseGeoJson("Not valid json {{{", 28.6, 77.2, 0.0, graph);
    EXPECT_EQ(count, 0u);
    EXPECT_EQ(graph.numRoads(), 0u);
}

}  // namespace
}  // namespace maps
}  // namespace idr
