# Render benchmark JSON reports (scripts/benchmark.py) as a comparison chart:
# stacked per-phase bars for each model's best run + per-step denoise times, one
# group per input file — the before/after view for the mx.compile coverage work.
#
#   uv run python scripts/benchmark_plot.py qwen_before.json qwen_after.json --out qwen.png
#
# Also prints a per-model delta table when given 2+ files. matplotlib is a main
# dependency (training loss plots), so no new requirements; Agg backend keeps it
# headless. *.png is gitignored, so chart files never dirty the repo.
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

PHASES = ("encode_s", "denoise_s", "decode_s")
PHASE_COLORS = {"encode_s": "#8ecae6", "denoise_s": "#219ebc", "decode_s": "#ffb703"}
MODEL_COLORS = ["#219ebc", "#d1495b", "#6a994e", "#7b2cbf", "#bc6c25", "#3a506b"]


@dataclass(frozen=True)
class Report:
    label: str
    runs: list[dict]
    system: dict
    git: dict
    config: dict


class BenchmarkPlot:
    @staticmethod
    def load_reports(paths: list[Path], labels: list[str] | None) -> list[Report]:
        reports = []
        for i, path in enumerate(paths):
            data = json.loads(path.read_text())
            label = labels[i] if labels else path.stem
            reports.append(
                Report(
                    label=label,
                    runs=data["runs"],
                    system=data.get("system", {}),
                    git=data.get("git", {}),
                    config=data.get("config", {}),
                )
            )
        return reports

    @staticmethod
    def best_runs(reports: list[Report]) -> dict[tuple[str, str], dict]:
        # Min-total run per (file, model): the thermal-throttle-robust choice (see
        # the sustained-load caveat in scripts/benchmark.py).
        best = {}
        for report in reports:
            for run in report.runs:
                key = (report.label, run["model"])
                if key not in best or run["total_s"] < best[key]["total_s"]:
                    best[key] = run
        return best

    @staticmethod
    def model_order(reports: list[Report]) -> list[str]:
        seen = []
        for report in reports:
            for run in report.runs:
                if run["model"] not in seen:
                    seen.append(run["model"])
        return seen

    @staticmethod
    def delta_table(reports: list[Report]) -> list[dict]:
        best = BenchmarkPlot.best_runs(reports)
        baseline, *others = reports
        rows = []
        for model in BenchmarkPlot.model_order(reports):
            before = best.get((baseline.label, model))
            if before is None:
                continue
            row = {"model": model, "baseline_total_s": before["total_s"], "deltas": {}}
            for report in others:
                after = best.get((report.label, model))
                if after is None:
                    continue
                row["deltas"][report.label] = {
                    "total_s": after["total_s"],
                    "total_pct": 100.0 * (after["total_s"] - before["total_s"]) / before["total_s"],
                    "denoise_pct": 100.0 * (after["denoise_s"] - before["denoise_s"]) / before["denoise_s"],
                }
            rows.append(row)
        return rows

    @staticmethod
    def render(reports: list[Report], out: Path) -> None:
        best = BenchmarkPlot.best_runs(reports)
        models = BenchmarkPlot.model_order(reports)
        n_files = len(reports)

        fig, (ax_bars, ax_steps) = plt.subplots(
            2, 1, figsize=(max(8.0, 1.6 * len(models) * n_files + 4), 9), height_ratios=[2, 1]
        )
        BenchmarkPlot._draw_bars(ax_bars, reports, models, best)
        BenchmarkPlot._draw_steps(ax_steps, reports, models, best)
        fig.suptitle(BenchmarkPlot._title(reports), fontsize=11)
        fig.tight_layout()
        fig.savefig(out, dpi=150)

    @staticmethod
    def _draw_bars(ax, reports: list[Report], models: list[str], best: dict) -> None:
        hatches = [None, "//", "xx", "\\\\", "oo"]
        width = 0.8 / len(reports)
        for i, report in enumerate(reports):
            hatch = hatches[i % len(hatches)]
            for j, model in enumerate(models):
                run = best.get((report.label, model))
                if run is None:
                    continue
                x = j - 0.4 + width * (i + 0.5)
                bottom = 0.0
                for phase in PHASES:
                    value = run[phase]
                    ax.bar(
                        x,
                        value,
                        width,
                        bottom=bottom,
                        color=PHASE_COLORS[phase],
                        hatch=hatch,
                        edgecolor="white",
                        linewidth=0.5,
                    )
                    bottom += value
                ax.text(x, bottom, f"{run['total_s']:.1f}s", ha="center", va="bottom", fontsize=9)

        phase_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=PHASE_COLORS[p]) for p in PHASES]
        file_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="black", hatch=hatches[i % len(hatches)])
            for i, r in enumerate(reports)
        ]
        ax.legend(
            phase_handles + file_handles,
            [p.removesuffix("_s") for p in PHASES] + [r.label for r in reports],
            loc="upper right",
            fontsize=8,
        )
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([BenchmarkPlot._model_tick(models, best, m) for m in models], fontsize=9)
        ax.set_ylabel("seconds (min-total run, stacked phases)")
        ax.margins(y=0.15)

    @staticmethod
    def _draw_steps(ax, reports: list[Report], models: list[str], best: dict) -> None:
        for i, report in enumerate(reports):
            for j, model in enumerate(models):
                run = best.get((report.label, model))
                if run is None or not run["step_times_s"]:
                    continue
                ax.plot(
                    range(1, len(run["step_times_s"]) + 1),
                    run["step_times_s"],
                    marker="o",
                    markersize=3,
                    color=MODEL_COLORS[j % len(MODEL_COLORS)],
                    linestyle=("-" if i == 0 else "--"),
                    alpha=0.9 if i == 0 else 0.7,
                    label=f"{model} ({report.label})",
                )
        ax.set_xlabel("denoise step (shifted by one sync — relative shape only)")
        ax.set_ylabel("seconds")
        ax.legend(fontsize=8)

    @staticmethod
    def _model_tick(models: list[str], best: dict, model: str) -> str:
        peaks = {run["peak_memory_gb"] for (label, m), run in best.items() if m == model}
        peak_note = f"\npeak {max(peaks):.0f} GB" if len(peaks) == 1 else ""
        return f"{model}{peak_note}"

    @staticmethod
    def _title(reports: list[Report]) -> str:
        first = reports[0]
        chip = first.system.get("chip") or first.system.get("os", "")
        commits = " vs ".join(
            f"{r.label}@{r.git.get('commit', '?')}{'*' if r.git.get('dirty') else ''}" for r in reports
        )
        cfg = first.config
        shape = f"{cfg.get('width')}x{cfg.get('height')}, steps={cfg.get('steps')}, q{cfg.get('quantize')}"
        return f"{chip} — {shape}\n{commits}  (* = dirty worktree)"


def main():
    parser = argparse.ArgumentParser(description="Plot benchmark JSON reports (A/B comparison) as a PNG.")
    parser.add_argument("reports", nargs="+", type=Path, help="benchmark.py JSON output(s), baseline first")
    parser.add_argument("--labels", nargs="*", default=None, help="Display label per report (default: file stem)")
    parser.add_argument("--out", type=Path, default=Path("benchmark.png"))
    args = parser.parse_args()

    reports = BenchmarkPlot.load_reports(args.reports, args.labels)
    BenchmarkPlot.render(reports, args.out)
    print(f"wrote {args.out}")

    if len(reports) > 1:
        print(f"\n=== delta vs {reports[0].label} (min-total run per model) ===")
        for row in BenchmarkPlot.delta_table(reports):
            deltas = " | ".join(
                f"{label}: total {d['total_s']:.1f}s ({d['total_pct']:+.1f}%), denoise {d['denoise_pct']:+.1f}%"
                for label, d in row["deltas"].items()
            )
            print(f"  {row['model']}: baseline {row['baseline_total_s']:.1f}s | {deltas}")


if __name__ == "__main__":
    sys.exit(main())
