from tools.filter_benchmark.compare_filters import run_filter_comparison


def test_filter_comparison_results():
    results = run_filter_comparison(duration_s=60.0, dt_s=0.1)

    assert "eskf" in results
    assert "iekf" in results

    # In-EKF must show reduced position drift during outage
    assert results["iekf"]["max_drift_m"] < results["eskf"]["max_drift_m"]
    assert results["drift_reduction_percentage"] > 30.0

    # In-EKF NEES must indicate better consistency (closer to 1.0-4.0)
    assert results["iekf"]["average_nees"] < results["eskf"]["average_nees"]
    assert results["nees_consistency_improved"] is True
