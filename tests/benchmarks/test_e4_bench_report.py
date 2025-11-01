from __future__ import annotations

from harness.e4_bench import run_bench_e4  # [python.e4_bench.run_bench_e4()](harness/e4_bench.py:367)


def test_markdown_report_generation_smoke(tmp_path):
    md_path = tmp_path / "e4_report.md"
    summary = run_bench_e4(grid_shape=(2, 2), iterations=3, slow_loop_period_us=1_000_000, md_path=str(md_path))
    assert "report_path" in summary
    assert md_path.exists()
    txt = md_path.read_text(encoding="utf-8")
    assert "# E4 Benchmark Report" in txt
    assert "Slow-plane Overhead" in txt
    assert "Control Loop Timing Stats" in txt