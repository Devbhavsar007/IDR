"""
IDR Trajectory Plotter & Visualizer.

Generates:
  1. Interactive Leaflet HTML map of estimated vs ground-truth trajectories with GNSS outage zones.
  2. Standalone SVG error & speed analysis charts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple


def parse_trajectory_csv(filepath: str | Path) -> List[Dict[str, float]]:
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row.get("latitude_deg") or row.get("lat") or 0.0)
                lon = float(row.get("longitude_deg") or row.get("lon") or 0.0)
                spd = float(row.get("speed_mps") or row.get("speed") or 0.0)
                t = float(row.get("timestamp_ns") or row.get("timestamp") or 0.0)
                mode = int(float(row.get("mode", 0)))
                acc = float(row.get("h_acc_m") or row.get("accuracy") or 0.0)

                if lat != 0.0 and lon != 0.0:
                    records.append({
                        "t": t,
                        "lat": lat,
                        "lon": lon,
                        "speed": spd,
                        "mode": mode,
                        "accuracy": acc,
                    })
            except (ValueError, TypeError):
                continue
    return records


def generate_interactive_html(
    estimated_points: List[Dict[str, float]],
    output_html_path: str | Path,
    title: str = "IDR Dead Reckoning Trajectory",
) -> Path:
    """Generate self-contained Leaflet HTML map."""
    path = Path(output_html_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not estimated_points:
        coords_json = "[]"
        center_lat, center_lon = 28.6139, 77.2090
    else:
        coords = [[p["lat"], p["lon"], p["mode"], p["speed"]] for p in estimated_points]
        coords_json = json.dumps(coords)
        center_lat = estimated_points[len(estimated_points) // 2]["lat"]
        center_lon = estimated_points[len(estimated_points) // 2]["lon"]

    html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8" />
    <title>{title}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        #map {{ width: 100vw; height: 100vh; }}
        .legend {{
            position: absolute; bottom: 30px; right: 20px; z-index: 1000;
            background: rgba(18, 18, 18, 0.9); color: white; padding: 16px 20px;
            border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.5); font-size: 13px;
        }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 8px; }}
        .color-box {{ width: 18px; height: 4px; margin-right: 10px; border-radius: 2px; }}
    </style>
</head>
<body>
    <div id="map"></div>
    <div class="legend">
        <h4 style="margin: 0 0 10px 0; font-size: 15px;">IDR Navigation Modes</h4>
        <div class="legend-item"><div class="color-box" style="background: #2E7D32;"></div> GNSS + INS Fusion</div>
        <div class="legend-item"><div class="color-box" style="background: #1565C0;"></div> Pure Dead Reckoning</div>
        <div class="legend-item"><div class="color-box" style="background: #E65100;"></div> Degraded GNSS</div>
    </div>

    <script>
        const map = L.map('map').setView([{center_lat}, {center_lon}], 16);
        L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            maxZoom: 19,
            attribution: '© OpenStreetMap contributors'
        }}).addTo(map);

        const data = {coords_json};
        if (data.length > 0) {{
            let currentSegment = [];
            let currentMode = data[0][2];

            function getModeColor(mode) {{
                if (mode === 5) return '#1565C0'; // DR (Blue)
                if (mode === 4) return '#E65100'; // Degraded (Amber)
                return '#2E7D32';                // GNSS_INS (Green)
            }}

            for (let i = 0; i < data.length; i++) {{
                const pt = [data[i][0], data[i][1]];
                const mode = data[i][2];

                if (mode !== currentMode && currentSegment.length > 0) {{
                    currentSegment.push(pt);
                    L.polyline(currentSegment, {{ color: getModeColor(currentMode), weight: 5, opacity: 0.85 }}).addTo(map);
                    currentSegment = [pt];
                    currentMode = mode;
                }} else {{
                    currentSegment.push(pt);
                }}
            }}

            if (currentSegment.length > 1) {{
                L.polyline(currentSegment, {{ color: getModeColor(currentMode), weight: 5, opacity: 0.85 }}).addTo(map);
            }}

            // Start & End markers
            L.circleMarker(data[0].slice(0, 2), {{ color: '#2E7D32', radius: 8, fillOpacity: 1 }}).bindPopup("Start").addTo(map);
            L.circleMarker(data[data.length - 1].slice(0, 2), {{ color: '#C62828', radius: 8, fillOpacity: 1 }}).bindPopup("End").addTo(map);
        }}
    </script>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")
    return path


def generate_error_chart_svg(
    points: List[Dict[str, float]],
    output_svg_path: str | Path,
) -> Path:
    """Generate standalone SVG chart showing speed and accuracy over sample index."""
    path = Path(output_svg_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width = 800
    height = 300
    padding = 50

    if not points:
        svg = f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg"><text x="100" y="150" fill="gray">No data</text></svg>'
        path.write_text(svg, encoding="utf-8")
        return path

    speeds = [p["speed"] * 3.6 for p in points]  # km/h
    max_spd = max(speeds) if speeds and max(speeds) > 0 else 50.0

    n = len(speeds)
    plot_w = width - 2 * padding
    plot_h = height - 2 * padding

    polyline_pts = []
    for i, spd in enumerate(speeds):
        x = padding + (i / max(1, n - 1)) * plot_w
        y = height - padding - (spd / max_spd) * plot_h
        polyline_pts.append(f"{x:.1f},{y:.1f}")

    pts_str = " ".join(polyline_pts)

    svg = f"""<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">
    <rect width="100%" height="100%" fill="#1E1E1E" />
    <text x="{padding}" y="30" fill="#E0E0E0" font-family="sans-serif" font-size="14" font-weight="bold">Speed Profile (km/h) across Replay</text>
    
    <!-- Axes -->
    <line x1="{padding}" y1="{height - padding}" x2="{width - padding}" y2="{height - padding}" stroke="#555" stroke-width="1.5" />
    <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height - padding}" stroke="#555" stroke-width="1.5" />
    
    <!-- Speed trace -->
    <polyline fill="none" stroke="#4CAF50" stroke-width="2.5" points="{pts_str}" />
    
    <text x="{padding - 35}" y="{padding + 10}" fill="#888" font-family="sans-serif" font-size="11">{max_spd:.0f}</text>
    <text x="{padding - 20}" y="{height - padding}" fill="#888" font-family="sans-serif" font-size="11">0</text>
</svg>"""
    path.write_text(svg, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="IDR Trajectory Plotter & Visualizer")
    parser.add_argument("input", help="Trajectory CSV file")
    parser.add_argument("-o", "--output-html", default="trajectory_map.html", help="Output HTML map")
    parser.add_argument("-s", "--output-svg", default="trajectory_speed.svg", help="Output SVG chart")

    args = parser.parse_args()
    points = parse_trajectory_csv(args.input)
    print(f"Loaded {len(points)} points from {args.input}")

    generate_interactive_html(points, args.output_html)
    generate_error_chart_svg(points, args.output_svg)
    print(f"Interactive map written to: {args.output_html}")
    print(f"Speed SVG chart written to: {args.output_svg}")


if __name__ == "__main__":
    main()
