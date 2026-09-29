#!/usr/bin/env python3
"""
IDR — Offline Vector Tile Packager
Converts GeoJSON road networks into compact, indexed binary tile packages (.idrtile)
with spatial bounding boxes and multi-level elevation layer indexing for offline navigation.
"""

import json
import struct
import math
from pathlib import Path
from typing import Dict, List, Any, Tuple


class TilePackager:
    MAGIC = b"IDRT"  # IDR Tile format magic bytes
    VERSION = 1

    def __init__(self, cell_size_deg: float = 0.05):
        self.cell_size_deg = cell_size_deg

    def package_geojson(self, geojson_path: Path, output_dir: Path) -> Dict[str, Any]:
        with open(geojson_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        features = data.get("features", [])
        output_dir.mkdir(parents=True, exist_ok=True)

        # Compute global bounding box
        min_lon, min_lat = float("inf"), float("inf")
        max_lon, max_lat = float("-inf"), float("-inf")

        roads = []
        for feat in features:
            geom = feat.get("geometry", {})
            props = feat.get("properties", {})
            coords = geom.get("coordinates", [])

            if geom.get("type") != "LineString" or not coords:
                continue

            road_id = props.get("id", len(roads) + 1)
            name = props.get("name", f"Road_{road_id}")
            layer = props.get("layer", 0)
            speed_limit = float(props.get("speed_limit_kmh", 50.0))
            one_way = bool(props.get("one_way", False))

            for pt in coords:
                lon, lat = pt[0], pt[1]
                min_lon, min_lat = min(min_lon, lon), min(min_lat, lat)
                max_lon, max_lat = max(max_lon, lon), max(max_lat, lat)

            roads.append({
                "id": road_id,
                "name": name,
                "layer": layer,
                "speed_limit_kmh": speed_limit,
                "one_way": one_way,
                "coordinates": coords
            })

        # Serialize into binary package
        package_file = output_dir / "delhi_region.idrtile"
        with open(package_file, "wb") as f_out:
            # Header: MAGIC (4B), Version (2B), RoadCount (4B)
            f_out.write(self.MAGIC)
            f_out.write(struct.pack(">HI", self.VERSION, len(roads)))

            # Global Bounding Box: min_lat, min_lon, max_lat, max_lon (4 x double = 32B)
            f_out.write(struct.pack(">4d", min_lat, min_lon, max_lat, max_lon))

            # Encode each road
            for road in roads:
                name_bytes = road["name"].encode("utf-8")
                # Road header: ID (4B), Layer (1B signed), OneWay (1B), Speed (2B uint16 * 10), NameLen (2B)
                f_out.write(struct.pack(
                    ">IbbHH",
                    road["id"],
                    road["layer"],
                    1 if road["one_way"] else 0,
                    int(road["speed_limit_kmh"] * 10),
                    len(name_bytes)
                ))
                f_out.write(name_bytes)

                # Coordinate points: Count (2B), then (lat, lon, alt) as floats (3 x 4B)
                coords = road["coordinates"]
                f_out.write(struct.pack(">H", len(coords)))
                for pt in coords:
                    lon = float(pt[0])
                    lat = float(pt[1])
                    alt = float(pt[2]) if len(pt) > 2 else 0.0
                    f_out.write(struct.pack(">3f", lat, lon, alt))

        manifest = {
            "format": "idrtile",
            "version": self.VERSION,
            "tile_file": package_file.name,
            "total_roads": len(roads),
            "layers": sorted(list(set(r["layer"] for r in roads))),
            "bounds": {
                "min_latitude": min_lat,
                "min_longitude": min_lon,
                "max_latitude": max_lat,
                "max_longitude": max_lon
            },
            "file_size_bytes": package_file.stat().st_size
        }

        manifest_file = output_dir / "manifest.json"
        with open(manifest_file, "w", encoding="utf-8") as f_man:
            json.dump(manifest, f_man, indent=2)

        return manifest


def unpack_tile_header(tile_path: Path) -> Tuple[int, int, Tuple[float, float, float, float]]:
    with open(tile_path, "rb") as f:
        magic = f.read(4)
        if magic != TilePackager.MAGIC:
            raise ValueError(f"Invalid magic bytes: {magic}")
        version, road_count = struct.unpack(">HI", f.read(6))
        min_lat, min_lon, max_lat, max_lon = struct.unpack(">4d", f.read(32))
    return version, road_count, (min_lat, min_lon, max_lat, max_lon)


if __name__ == "__main__":
    import sys
    base_dir = Path(__file__).resolve().parent.parent
    geojson = base_dir / "schemas" / "delhi_corridor.geojson"
    out = base_dir / "packager" / "build"
    packager = TilePackager()
    manifest = packager.package_geojson(geojson, out)
    print(f"Packaged {manifest['total_roads']} roads into {out / manifest['tile_file']}")
