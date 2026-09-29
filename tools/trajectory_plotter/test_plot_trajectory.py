"""Unit tests for IDR Trajectory Plotter."""

import tempfile
from pathlib import Path
from tools.trajectory_plotter.plot_trajectory import (
    parse_trajectory_csv,
    generate_interactive_html,
    generate_error_chart_svg,
)


def test_trajectory_plotter_html_and_svg_generation():
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "trajectory.csv"
        csv_path.write_text(
            "timestamp_ns,latitude_deg,longitude_deg,altitude_m,speed_mps,heading_deg,mode,h_acc_m\n"
            "1000000000,28.6139,77.2090,216.0,10.0,0.0,3,2.5\n"
            "2000000000,28.6149,77.2090,216.0,12.0,0.0,5,3.8\n"
            "3000000000,28.6159,77.2090,216.0,14.0,0.0,3,1.5\n"
        )

        points = parse_trajectory_csv(csv_path)
        assert len(points) == 3
        assert points[0]["mode"] == 3
        assert points[1]["mode"] == 5

        html_out = Path(tmpdir) / "map.html"
        svg_out = Path(tmpdir) / "chart.svg"

        generate_interactive_html(points, html_out)
        generate_error_chart_svg(points, svg_out)

        assert html_out.is_file()
        assert svg_out.is_file()

        html_content = html_out.read_text(encoding="utf-8")
        assert "L.map" in html_content
        assert "IDR Navigation Modes" in html_content

        svg_content = svg_out.read_text(encoding="utf-8")
        assert "<svg" in svg_content
        assert "<polyline" in svg_content
