"""Unit tests for IDR Benchmark Runner."""

import tempfile
from pathlib import Path
from tools.benchmark_runner.benchmark_runner import BenchmarkRunner


def test_benchmark_runner_suite():
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(replay_binary_path="d:/Dev/IDR/build/simulation/idr-replay.exe")
        results = runner.run_suite(output_dir=tmpdir)

        assert len(results) >= 5
        assert all(r.ate_rms_m >= 0.0 for r in results)
        assert all(r.passed_targets for r in results)

        summary_json = Path(tmpdir) / "benchmark_summary.json"
        summary_md = Path(tmpdir) / "benchmark_summary.md"
        assert summary_json.is_file()
        assert summary_md.is_file()
