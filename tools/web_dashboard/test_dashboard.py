from pathlib import Path


def test_web_dashboard_html_structure():
    dashboard_path = Path(__file__).resolve().parent / "index.html"
    assert dashboard_path.is_file(), f"Dashboard not found at {dashboard_path}"

    with open(dashboard_path, "r", encoding="utf-8") as f:
        html = f.read()

    # Structural assertions
    assert "<!DOCTYPE html>" in html
    assert "<canvas id=\"trajectoryCanvas\"></canvas>" in html
    assert "DPDP Act 2023 Compliant" in html
    assert "Barapullah Elevated" in html
    assert "Invariant EKF on SE₂(3)" in html
    assert "Standard Error-State EKF" in html
    assert "Lightweight Neural Model" in html
    assert "speedDisplay" in html
    assert "togglePlayBtn" in html
