import importlib.util
import json
from pathlib import Path

import pytest

PLOT_PATH = Path(__file__).parent.parent / "scripts" / "benchmark_plot.py"
spec = importlib.util.spec_from_file_location("benchmark_plot", PLOT_PATH)
benchmark_plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark_plot)

BenchmarkPlot = benchmark_plot.BenchmarkPlot


def write_report(path: Path, runs: list[dict], label_config: dict | None = None) -> Path:
    report = {
        "schema_version": 1,
        "created_at": "2026-08-26T00:00:00+0000",
        "system": {"chip": "Apple M4 Max", "os": "macOS", "python": "3.13", "mlx": "0.32.0", "mflux": "0.19.1"},
        "git": {"commit": "abc123", "dirty": False},
        "config": label_config or {"prompt": "p", "seed": 42, "steps": 4, "width": 1024, "height": 1024, "quantize": 8},
        "runs": runs,
        "summary": [],
    }
    path.write_text(json.dumps(report))
    return path


def run(model: str, run_index: int, total: float) -> dict:
    return {
        "model": model,
        "run_index": run_index,
        "encode_s": 0.5,
        "denoise_s": total - 1.5,
        "decode_s": 1.0,
        "total_s": total,
        "step_times_s": [total / 4] * 4,
        "peak_memory_gb": 36.0,
    }


@pytest.mark.fast
def test_best_runs_picks_min_total_per_label_and_model(tmp_path):
    before = write_report(tmp_path / "before.json", [run("m", 0, 30.0), run("m", 1, 25.0)])
    after = write_report(tmp_path / "after.json", [run("m", 0, 20.0), run("m", 1, 22.0)])
    reports = BenchmarkPlot.load_reports([before, after], labels=["before", "after"])

    best = BenchmarkPlot.best_runs(reports)

    assert best[("before", "m")]["total_s"] == 25.0
    assert best[("after", "m")]["total_s"] == 20.0


@pytest.mark.fast
def test_delta_table_computes_percentages_against_first_report(tmp_path):
    before = write_report(tmp_path / "before.json", [run("m", 0, 40.0)])
    after = write_report(tmp_path / "after.json", [run("m", 0, 30.0)])
    reports = BenchmarkPlot.load_reports([before, after], labels=None)

    rows = BenchmarkPlot.delta_table(reports)

    assert len(rows) == 1
    delta = rows[0]["deltas"]["after"]
    assert delta["total_s"] == 30.0
    assert delta["total_pct"] == -25.0
    assert delta["denoise_pct"] == pytest.approx(100.0 * (28.5 - 38.5) / 38.5)


@pytest.mark.fast
def test_render_writes_non_empty_png(tmp_path):
    before = write_report(tmp_path / "before.json", [run("m", 0, 40.0), run("n", 0, 80.0)])
    after = write_report(tmp_path / "after.json", [run("m", 0, 30.0)])
    reports = BenchmarkPlot.load_reports([before, after], labels=None)
    out = tmp_path / "chart.png"

    BenchmarkPlot.render(reports, out)

    assert out.stat().st_size > 10_000
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
