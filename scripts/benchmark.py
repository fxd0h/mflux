# Per-phase generation benchmark: encode / denoise / decode wall times + peak MLX
# memory, emitted as JSON. This harness exists so the mx.compile coverage rollout
# (plan/next-improvements-proposal.md, improvement #1) can be done methodically:
# one fixed-seed A/B command per model, before and after each compile change.
#
#   uv run python scripts/benchmark.py --model z-image-turbo -q 8
#   uv run python scripts/benchmark.py --model qwen-image --steps 20 --runs 3 --json-out qwen_before.json
#
# Timing note: MLX submits work asynchronously, so `encode_s` includes any
# submission slop absorbed by the loop's first step; the per-step mx.eval in every
# denoise loop makes `denoise_s` (and `total_s`, which ends in a full host
# materialization) hard numbers. For A/B comparisons of the same model the encode
# bias is constant, which is all the compile work needs. Individual `step_times_s`
# are shifted by one sync (in_loop fires just before that step's mx.eval), so read
# them as diagnostics, not absolute per-step costs.
#
# Sustained-load caveat: long runs can thermal-throttle (observed: 9-step q8 run 1
# at ~1.7x run 0 step time on M4 Max; 4-step runs are stable to ±1.3%). For A/B
# decisions prefer min/median of several short runs over one long session.
import argparse
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import PIL.Image
import tqdm

from mflux.callbacks.callback import AfterLoopCallback, BeforeLoopCallback, InLoopCallback
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.cli.defaults.defaults import model_inference_steps
from mflux.models.boogu.variants.txt2img.boogu_image import BooguImage
from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.fibo.variants.txt2img.fibo import FIBO
from mflux.models.flux.variants.txt2img.flux import Flux1
from mflux.models.krea2.variants.txt2img.krea2 import Krea2
from mflux.models.lens.variants.txt2img.lens_image import LensImage
from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
from mflux.models.z_image.variants.z_image import ZImage

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ModelSpec:
    cls: type
    config: ModelConfig


# Keyed by canonical AVAILABLE_MODELS names so model_inference_steps() resolves the
# same step defaults the CLIs use. Registry covers the txt2img families in scope for
# the compile-coverage work; adding a row is the whole cost of covering a new one.
MODEL_SPECS: dict[str, ModelSpec] = {
    "z-image-turbo": ModelSpec(cls=ZImage, config=ModelConfig.z_image_turbo()),
    "qwen-image": ModelSpec(cls=QwenImage, config=ModelConfig.qwen_image()),
    "krea-2": ModelSpec(cls=Krea2, config=ModelConfig.krea2()),
    "boogu-image-turbo": ModelSpec(cls=BooguImage, config=ModelConfig.boogu_image_turbo()),
    "fibo": ModelSpec(cls=FIBO, config=ModelConfig.fibo()),
    "lens-turbo": ModelSpec(cls=LensImage, config=ModelConfig.lens_turbo()),
    "dev": ModelSpec(cls=Flux1, config=ModelConfig.dev()),
    "schnell": ModelSpec(cls=Flux1, config=ModelConfig.schnell()),
}


class BenchmarkTimer(BeforeLoopCallback, InLoopCallback, AfterLoopCallback):
    def __init__(self, clock=time.perf_counter):
        self.clock = clock
        self.t_start = None
        self.t_loop_start = None
        self.t_loop_end = None
        self.step_stamps = []

    def mark_start(self) -> None:
        self.t_start = self.clock()
        self.t_loop_start = None
        self.t_loop_end = None
        self.step_stamps = []

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
        control_images: list[PIL.Image.Image] | None = None,
    ) -> None:
        self.t_loop_start = self.clock()

    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: Config,
        time_steps: tqdm.tqdm,
    ) -> None:
        self.step_stamps.append(self.clock())

    def call_after_loop(self, seed: int, prompt: str, latents: mx.array, config: Config) -> None:
        self.t_loop_end = self.clock()

    def result(self, t_end: float, peak_memory_bytes: int) -> dict:
        step_times = [b - a for a, b in zip([self.t_loop_start, *self.step_stamps], self.step_stamps, strict=False)]
        return {
            "encode_s": self.t_loop_start - self.t_start,
            "denoise_s": self.t_loop_end - self.t_loop_start,
            "decode_s": t_end - self.t_loop_end,
            "total_s": t_end - self.t_start,
            "step_times_s": step_times,
            "peak_memory_gb": peak_memory_bytes / 1e9,
        }


class Benchmark:
    @staticmethod
    def run_model(model_name: str, spec: ModelSpec, args: argparse.Namespace) -> list[dict]:
        print(f"=== {model_name} (quantize={args.quantize}, {args.runs} run(s)) ===")
        model = spec.cls(
            quantize=args.quantize,
            model_path=args.model_path,
            model_config=spec.config,
        )
        # Mirrors the CLI's non-low-ram path: text encoders stay resident across the
        # run loop (num_seeds>1), transformer kept, cache cleared between runs.
        model.callbacks.register(
            MemorySaver(model=model, keep_transformer=True, cache_limit_bytes=None, num_seeds=args.runs)
        )
        # One timer for the whole run loop; mark_start() resets its state per run so
        # nothing accumulates on the registry.
        timer = BenchmarkTimer()
        model.callbacks.register(timer)

        runs = []
        for run_index in range(args.runs):
            generate_kwargs = {
                "seed": args.seed,
                "prompt": args.prompt,
                "num_inference_steps": args.steps or model_inference_steps(model_name),
                "width": args.width,
                "height": args.height,
            }
            if args.guidance is not None:
                generate_kwargs["guidance"] = args.guidance
            if args.negative_prompt is not None:
                generate_kwargs["negative_prompt"] = args.negative_prompt

            mx.reset_peak_memory()
            timer.mark_start()
            model.generate_image(**generate_kwargs)
            result = timer.result(t_end=timer.clock(), peak_memory_bytes=mx.get_peak_memory())
            result["model"] = model_name
            result["run_index"] = run_index
            runs.append(result)
            Benchmark._print_run(result)
        return runs

    @staticmethod
    def summarize(runs: list[dict]) -> list[dict]:
        summaries = []
        for model_name in dict.fromkeys(run["model"] for run in runs):
            totals = [run["total_s"] for run in runs if run["model"] == model_name]
            denoise = [run["denoise_s"] for run in runs if run["model"] == model_name]
            summaries.append(
                {
                    "model": model_name,
                    "runs": len(totals),
                    "total_min_s": min(totals),
                    "total_median_s": statistics.median(totals),
                    "denoise_median_s": statistics.median(denoise),
                }
            )
        return summaries

    @staticmethod
    def system_info() -> dict:
        chip = ""
        if platform.system() == "Darwin":
            try:
                chip = subprocess.run(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=True
                ).stdout.strip()
            except (subprocess.CalledProcessError, OSError):
                pass
        try:
            mflux_version = importlib.metadata.version("mflux")
        except importlib.metadata.PackageNotFoundError:
            mflux_version = "unknown"
        return {
            "chip": chip,
            "os": platform.platform(),
            "python": platform.python_version(),
            "mlx": mx.__version__,
            "mflux": mflux_version,
        }

    @staticmethod
    def git_info() -> dict:
        def git(*args: str) -> str:
            return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

        return {
            "commit": git("rev-parse", "--short", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }

    @staticmethod
    def _print_run(run: dict) -> None:
        print(
            f"  run {run['run_index']}: "
            f"encode {run['encode_s']:.2f}s | denoise {run['denoise_s']:.2f}s | "
            f"decode {run['decode_s']:.2f}s | total {run['total_s']:.2f}s | "
            f"peak {run['peak_memory_gb']:.1f} GB"
        )


def main():
    parser = argparse.ArgumentParser(description="Per-phase mflux generation benchmark (JSON out).")
    parser.add_argument(
        "--model",
        action="append",
        choices=list(MODEL_SPECS),
        help="Model to benchmark (repeatable). Default: z-image-turbo",
    )
    parser.add_argument("--prompt", default="A puffin standing on a cliff")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=None, help="Default: the model's CLI step count")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance", type=float, default=None, help="Default: the model's own default")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--quantize", "-q", type=int, choices=[3, 4, 5, 6, 8], default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--runs",
        type=int,
        default=2,
        help="Generations per model. Run 0 includes MLX compile warmup on compiled models — "
        "kept in the report because compile latency is itself a metric for the compile-coverage work.",
    )
    parser.add_argument("--json-out", default="benchmark_results.json")
    args = parser.parse_args()

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "system": Benchmark.system_info(),
        "git": Benchmark.git_info(),
        "config": {
            "prompt": args.prompt,
            "seed": args.seed,
            "steps": args.steps,
            "width": args.width,
            "height": args.height,
            "guidance": args.guidance,
            "quantize": args.quantize,
        },
        "runs": [],
    }
    for model_name in args.model or ["z-image-turbo"]:
        report["runs"].extend(Benchmark.run_model(model_name, MODEL_SPECS[model_name], args))

    report["summary"] = Benchmark.summarize(report["runs"])
    print("\n=== summary (median of runs; run 0 includes compile warmup) ===")
    for row in report["summary"]:
        print(
            f"  {row['model']}: total min {row['total_min_s']:.2f}s / median {row['total_median_s']:.2f}s, "
            f"denoise median {row['denoise_median_s']:.2f}s over {row['runs']} run(s)"
        )

    out = Path(args.json_out)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    sys.exit(main())
