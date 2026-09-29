import json
from pathlib import Path
import jsonschema


def test_delhi_corridor_conforms_to_schema():
    schema_path = Path(__file__).resolve().parent.parent / "schemas" / "road_network_schema.json"
    geojson_path = Path(__file__).resolve().parent.parent / "schemas" / "delhi_corridor.geojson"

    assert schema_path.is_file(), f"Schema file not found at {schema_path}"
    assert geojson_path.is_file(), f"GeoJSON file not found at {geojson_path}"

    with open(schema_path, "r", encoding="utf-8") as f:
        schema = json.load(f)

    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    # Validate against schema
    jsonschema.validate(instance=geojson, schema=schema)

    # Validate layer diversity (flyover layer 1, surface layer 0, underpass layer -1)
    layers = {feat["properties"]["layer"] for feat in geojson["features"]}
    assert 1 in layers, "Missing elevated flyover layer (1)"
    assert 0 in layers, "Missing at-grade surface layer (0)"
    assert -1 in layers, "Missing underpass layer (-1)"
    assert len(geojson["features"]) >= 3
