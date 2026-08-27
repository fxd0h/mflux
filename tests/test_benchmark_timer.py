import importlib.util
from pathlib import Path

import pytest

BENCHMARK_PATH = Path(__file__).parent.parent / "scripts" / "benchmark.py"
spec = importlib.util.spec_from_file_location("benchmark", BENCHMARK_PATH)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

BenchmarkTimer = benchmark.BenchmarkTimer
summarize = benchmark.Benchmark.summarize


def driven_timer(stamps: list[float]) -> BenchmarkTimer:
    clock = iter(stamps).__next__
    timer = BenchmarkTimer(clock=clock)
    timer.mark_start()
    timer.call_before_loop(seed=0, prompt="", latents=None, config=None)
    timer.call_in_loop(t=0, seed=0, prompt="", latents=None, config=None, time_steps=None)
    timer.call_in_loop(t=1, seed=0, prompt="", latents=None, config=None, time_steps=None)
    timer.call_after_loop(seed=0, prompt="", latents=None, config=None)
    return timer


@pytest.mark.fast
def test_timer_computes_phase_durations():
    timer = driven_timer([10.0, 12.0, 13.0, 15.0, 18.0])
    result = timer.result(t_end=20.0, peak_memory_bytes=2_000_000_000)

    assert result["encode_s"] == 2.0
    assert result["denoise_s"] == 6.0
    assert result["decode_s"] == 2.0
    assert result["total_s"] == 10.0
    assert result["step_times_s"] == [1.0, 2.0]
    assert result["peak_memory_gb"] == 2.0


@pytest.mark.fast
def test_timer_mark_start_resets_between_runs():
    timer = driven_timer([10.0, 12.0, 13.0, 15.0, 18.0])
    timer.result(t_end=20.0, peak_memory_bytes=0)

    timer.clock = iter([100.0, 110.0, 111.0, 112.0, 120.0]).__next__
    timer.mark_start()
    timer.call_before_loop(seed=0, prompt="", latents=None, config=None)
    timer.call_in_loop(t=0, seed=0, prompt="", latents=None, config=None, time_steps=None)
    timer.call_in_loop(t=1, seed=0, prompt="", latents=None, config=None, time_steps=None)
    timer.call_after_loop(seed=0, prompt="", latents=None, config=None)
    result = timer.result(t_end=125.0, peak_memory_bytes=0)

    assert result["encode_s"] == 10.0
    assert result["step_times_s"] == [1.0, 1.0]
    assert result["total_s"] == 25.0


@pytest.mark.fast
def test_summarize_groups_per_model_with_min_and_median():
    runs = [
        {"model": "a", "total_s": 10.0, "denoise_s": 8.0},
        {"model": "a", "total_s": 12.0, "denoise_s": 10.0},
        {"model": "a", "total_s": 14.0, "denoise_s": 12.0},
        {"model": "b", "total_s": 30.0, "denoise_s": 25.0},
    ]

    summary = summarize(runs)

    assert summary == [
        {"model": "a", "runs": 3, "total_min_s": 10.0, "total_median_s": 12.0, "denoise_median_s": 10.0},
        {"model": "b", "runs": 1, "total_min_s": 30.0, "total_median_s": 30.0, "denoise_median_s": 25.0},
    ]
