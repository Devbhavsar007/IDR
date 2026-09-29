// IDR — Intelligent Dead Reckoning with GNSS Fusion
// Copyright (c) 2026 IDR Project. All rights reserved.
//
// OSM / GeoJSON Road Loader — loads OpenStreetMap road geometry
// into a RoadGraph with coordinate projection to local ENU.

#pragma once

#include "idr/map_graph/road_graph.h"
#include "idr/math_utils/geo_utils.h"

#include <spdlog/spdlog.h>
#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace idr {
namespace maps {

/// Loader for OpenStreetMap and GeoJSON road networks.
class OsmRoadLoader {
public:
    /// Load roads from a GeoJSON FeatureCollection file.
    /// Coordinates [longitude, latitude, (optional altitude)] are converted to local ENU.
    static size_t loadGeoJson(const std::string& filepath,
                             double ref_lat_deg,
                             double ref_lon_deg,
                             double ref_alt_m,
                             RoadGraph& out_graph) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("OsmRoadLoader: failed to open {}", filepath);
            return 0;
        }

        std::stringstream buffer;
        buffer << file.rdbuf();
        return parseGeoJson(buffer.str(), ref_lat_deg, ref_lon_deg, ref_alt_m, out_graph);
    }

    /// Parse GeoJSON string directly.
    static size_t parseGeoJson(const std::string& json,
                              double ref_lat_deg,
                              double ref_lon_deg,
                              double ref_alt_m,
                              RoadGraph& out_graph) {
        size_t roads_added = 0;
        size_t pos = 0;

        while ((pos = json.find("\"LineString\"", pos)) != std::string::npos) {
            // Find "coordinates" block
            size_t coord_start = json.find("\"coordinates\"", pos);
            if (coord_start == std::string::npos) break;

            size_t arr_start = json.find('[', coord_start);
            if (arr_start == std::string::npos) break;

            // Search backwards for road name or properties
            std::string name = "Unnamed Road";
            int level = 0;
            int64_t road_id = static_cast<int64_t>(roads_added + 1);

            size_t feat_start = json.rfind('{', pos);
            if (feat_start != std::string::npos) {
                size_t name_pos = json.find("\"name\"", feat_start);
                if (name_pos != std::string::npos && name_pos < arr_start) {
                    size_t val_start = json.find('"', name_pos + 6);
                    if (val_start != std::string::npos && val_start < arr_start) {
                        size_t val_end = json.find('"', val_start + 1);
                        if (val_end != std::string::npos) {
                            name = json.substr(val_start + 1, val_end - val_start - 1);
                        }
                    }
                }

                // Check for layer / level (e.g. flyover, tunnel)
                size_t level_pos = json.find("\"layer\"", feat_start);
                if (level_pos == std::string::npos) level_pos = json.find("\"level\"", feat_start);
                if (level_pos != std::string::npos && level_pos < arr_start) {
                    size_t colon = json.find(':', level_pos);
                    if (colon != std::string::npos && colon < arr_start) {
                        try {
                            level = std::stoi(json.substr(colon + 1, 4));
                        } catch (...) {}
                    }
                }
            }

            // Parse coordinate pairs: [ [lon, lat], [lon, lat], ... ]
            std::vector<Vec3d> enu_vertices;
            size_t cur = arr_start + 1;
            while (cur < json.size()) {
                size_t pair_start = json.find('[', cur);
                if (pair_start == std::string::npos) break;

                // Stop if we exit outer array
                size_t outer_end = json.find(']', pair_start);
                size_t next_bracket = json.find(']', outer_end + 1);
                if (next_bracket == std::string::npos) break;

                size_t pair_end = json.find(']', pair_start);
                if (pair_end == std::string::npos) break;

                std::string pair_str = json.substr(pair_start + 1, pair_end - pair_start - 1);
                size_t comma = pair_str.find(',');
                if (comma != std::string::npos) {
                    try {
                        double lon = std::stod(pair_str.substr(0, comma));
                        double lat = std::stod(pair_str.substr(comma + 1));
                        double alt = 0.0;
                        size_t comma2 = pair_str.find(',', comma + 1);
                        if (comma2 != std::string::npos) {
                            alt = std::stod(pair_str.substr(comma2 + 1));
                        }

                        Vec3d enu = geo::llaToEnu(lat, lon, alt, ref_lat_deg, ref_lon_deg, ref_alt_m);
                        enu_vertices.push_back(enu);
                    } catch (...) {}
                }

                cur = pair_end + 1;
                // If closing outer list
                while (cur < json.size() && (std::isspace(json[cur]) || json[cur] == ',')) cur++;
                if (cur < json.size() && json[cur] == ']') break;
            }

            if (enu_vertices.size() >= 2) {
                RoadSegment seg;
                seg.id = road_id;
                seg.name = name;
                seg.level = level;
                seg.vertices = std::move(enu_vertices);
                seg.computeHeadings();
                out_graph.addRoad(seg);
                roads_added++;
            }

            pos = arr_start + 1;
        }

        spdlog::info("OsmRoadLoader: loaded {} road segments", roads_added);
        return roads_added;
    }

    /// Load roads from simple CSV format:
    /// id,name,level,lon1,lat1,lon2,lat2,...
    static size_t loadCsvWays(const std::string& filepath,
                             double ref_lat_deg,
                             double ref_lon_deg,
                             double ref_alt_m,
                             RoadGraph& out_graph) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            spdlog::error("OsmRoadLoader: cannot open {}", filepath);
            return 0;
        }

        size_t count = 0;
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;

            std::stringstream ss(line);
            std::string token;
            std::vector<std::string> tokens;
            while (std::getline(ss, token, ',')) {
                tokens.push_back(token);
            }

            if (tokens.size() < 7) continue;  // id, name, level, lon1, lat1, lon2, lat2

            try {
                RoadSegment seg;
                seg.id = std::stoll(tokens[0]);
                seg.name = tokens[1];
                seg.level = std::stoi(tokens[2]);

                for (size_t i = 3; i + 1 < tokens.size(); i += 2) {
                    double lon = std::stod(tokens[i]);
                    double lat = std::stod(tokens[i + 1]);
                    Vec3d enu = geo::llaToEnu(lat, lon, 0.0, ref_lat_deg, ref_lon_deg, ref_alt_m);
                    seg.vertices.push_back(enu);
                }

                if (seg.vertices.size() >= 2) {
                    seg.computeHeadings();
                    out_graph.addRoad(seg);
                    count++;
                }
            } catch (...) {}
        }

        spdlog::info("OsmRoadLoader: loaded {} CSV road segments", count);
        return count;
    }
};

}  // namespace maps
}  // namespace idr
