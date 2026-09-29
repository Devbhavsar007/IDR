import tempfile
from pathlib import Path
from maps.packager.tile_packager import TilePackager, unpack_tile_header


def test_tile_packaging_and_unpacking():
    geojson_path = Path(__file__).resolve().parent.parent / "schemas" / "delhi_corridor.geojson"
    assert geojson_path.is_file(), "Sample Delhi corridor GeoJSON not found"

    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir)
        packager = TilePackager()
        manifest = packager.package_geojson(geojson_path, out_dir)

        assert manifest["total_roads"] == 4
        assert manifest["layers"] == [-1, 0, 1]
        assert manifest["file_size_bytes"] > 0

        tile_file = out_dir / manifest["tile_file"]
        assert tile_file.is_file()

        version, road_count, bbox = unpack_tile_header(tile_file)
        assert version == 1
        assert road_count == 4
        min_lat, min_lon, max_lat, max_lon = bbox

        # Coordinates should reflect Delhi corridor roughly 28.56 - 28.59 N, 77.20 - 77.25 E
        assert 28.5 < min_lat < max_lat < 28.65
        assert 77.15 < min_lon < max_lon < 77.30
