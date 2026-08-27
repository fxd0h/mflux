# Justfile Guide

### What & Why
Developers use command runners like make and just as CLI macros... named shortcuts for repeatable development tasks. 

[**Just**](https://github.com/casey/just) is simpler for task automation: it has cleaner syntax, clearer errors, and avoids [**Make**](https://en.wikipedia.org/wiki/Make_(software))’s build-system quirks. [**Make**](https://en.wikipedia.org/wiki/Make_(software)) is better when you need dependency-based incremental builds; [**Just**](https://github.com/casey/just) is better for commands like `test`, `lint`, and `release`.

[@anthonywu](https://github.com/anthonywu) proposed JustFile in a [MFlux-Community Discussion](https://github.com/orgs/mflux-community/discussions/585#discussion-10610831) in August 2026: _"This is the modern alternative to Makefiles, the syntax is much more readable to humans. In my two years working with just, AI assistants big and small have had no trouble helping me write these files."_


## Installing

<sub>as MFlux is targeting - we presume you are running on M-Series Mac - ie `aarch64`</sub>

```
brew install just
```


## Common MFLux Workflow

```sh
just install
just lint
just test-fast
```

`just install` creates `.venv`, synchronizes dependencies, and installs
pre-commit hooks. If `pre-commit` is not installed, the recipe installs it as
a uv tool.

## Recipes

| Command | Purpose |
| --- | --- |
| `just` | List public recipes. |
| `just all` | Install dependencies, then run the default test selection. |
| `just install` | Create the environment, synchronize dependencies, and install pre-commit hooks. |
| `just venv-init` | Create a clean Python 3.13 virtual environment after checking the platform and uv. |
| `just lint` | Run Ruff checks without changing files. |
| `just lint-justfile` | Verify that `just --fmt` would not reformat the justfile. |
| `just format` | Run Ruff's formatter and show a summary of changed files. |
| `just check` | Run all pre-commit hooks, including auto-fixes and formatters. Review changes afterwards. |
| `just test` | Run the default pytest selection, which excludes slow model tests. |
| `just test-fast` | Run tests marked `fast`; these do not generate images. |
| `just test-slow` | Run tests marked `slow`; these generate images. |
| `just test-all` | Run all tests except those marked `high_memory_requirement`. This can download model weights. |
| `just build` | Build sdist/wheel into `dist/`, then report artifact sizes and flag any oversized files in the sdist (a check that image assets weren't accidentally bundled). |
| `just benchmark [model] [quantize] [runs] [size]` | Run the per-phase generation benchmark (`scripts/benchmark.py`; encode/denoise/decode timings + peak GB) and write a JSON report whose filename embeds the git SHA + dirty flag so A/B runs never overwrite each other. Defaults: `z-image-turbo 8 2 1024`. |
| `just benchmark-plot before.json [after.json ...]` | Render one or more benchmark JSON reports into `benchmark.png` (stacked per-phase bars + per-step times) and print a delta table; the first report is the A/B baseline. |
| `just release` | Trigger the `release.yml` GitHub Actions workflow via `gh`, then watch the run. Publishing happens on GitHub via PyPI trusted publishing (OIDC); no local credentials are used. |
| `just clean` | Remove `.venv`. Run `just install` to recreate it. |

Test recipes first synchronize the locked environment with all extras, then
run pytest with `MFLUX_PRESERVE_TEST_OUTPUT=1` so generated outputs remain
available for inspection. They do not update golden images.

`just build` removes any stale `dist/mflux-*` artifacts, builds fresh sdist and
wheel packages with `uv build`, then reports their sizes (expected under 1MB)
and lists the five largest files inside the sdist — a quick sanity check that
no image outputs got swept into the package by mistake.

## Benchmarking

`just benchmark` measures where generation time goes — text encode, denoise
loop, and VAE decode, plus peak MLX memory — and writes a JSON report whose
filename embeds the git SHA + dirty flag, so A/B runs never overwrite each
other:

```sh
just benchmark                          # defaults: z-image-turbo q8, 2 runs, 1024px
just benchmark qwen-image none 3 768    # model, quantize (or 'none'), runs, square size
```

Example output:

```text
$ just benchmark
🏗️ Benchmarking z-image-turbo (quantize=8, runs=2, 1024px) → bench_z-image-turbo_q8_1024px_0f927481-dirty.json ...
=== z-image-turbo (quantize=8, 2 run(s)) ===
100%|██████████████████████████████████| 9/9 [01:07<00:00, 7.54s/it]
  run 0: encode 0.01s | denoise 67.92s | decode 0.64s | total 68.58s | peak 36.0 GB
100%|██████████████████████████████████| 9/9 [01:34<00:00, 10.50s/it]
  run 1: encode 0.00s | denoise 94.71s | decode 0.98s | total 95.70s | peak 36.0 GB

=== summary (median of runs; run 0 includes compile warmup) ===
  z-image-turbo: total min 68.58s / median 82.14s, denoise median 81.32s over 2 run(s)

wrote bench_z-image-turbo_q8_1024px_0f927481-dirty.json
✅ Benchmark written to bench_z-image-turbo_q8_1024px_0f927481-dirty.json.
```

This example also shows why one long session can mislead: run 1 came in ~40%
slower than run 0 on identical work because sustained GPU load thermal-throttles
the machine. Prefer several short runs and compare **min / median**, which is
exactly what `summary` prints.

### When to benchmark

- **Before and after any performance change** (`mx.compile` coverage,
  CFG batching, async eval). The report filename embeds the git SHA + dirty
  flag, so one command per side of the change is enough to keep both results:
  ```sh
  git checkout main              # on the baseline...
  just benchmark                 # → bench_..._abc1234-clean.json
  # ...make your change, then
  just benchmark                 # → bench_..._def5678-dirty.json
  just benchmark-plot bench_..._abc1234-clean.json bench_..._def5678-dirty.json
  ```
- Keep every variable fixed except your code change: same model, quantize,
  size, steps, seed, prompt, and machine state (plugged in, cooled down).
- The first run in a report includes MLX compile warmup on compiled models;
  that latency is itself a metric, so it is reported rather than hidden.

Plotting a single report works too and is a quick sanity check after a run:

```text
$ just benchmark-plot bench_z-image-turbo_q8_1024px_0f927481-dirty.json
🏗️ Plotting benchmark report(s) → benchmark.png ...
wrote benchmark.png
✅ Chart written to benchmark.png.
```

With two or more reports (baseline **first**), the same command adds a
per-model delta table with percent changes to the stdout above the chart:

```sh
just benchmark-plot bench_<model>_q8_1024px_abc1234-clean.json bench_<model>_q8_1024px_def5678-dirty.json
```

Either way `benchmark.png` shows stacked per-phase bars (one group per model,
hatch pattern per report) plus per-step denoise times. The file lands in the
repo root (`*.png` is gitignored, so charts never dirty your worktree); open
it with

```sh
open benchmark.png        # macOS; Linux: xdg-open benchmark.png
```

## Releasing to PyPI

Releases publish from GitHub Actions using PyPI trusted publishing (OIDC) —
there are no PyPI credentials to configure locally. Either run `just release`,
or trigger the workflow from the GitHub UI:

1. Go to <https://github.com/mflux-community/mflux/actions/workflows/release.yml>
2. Click the **Run workflow** dropdown (top right of the runs list)
3. Leave the branch as `main`, type `publish` in the
   *Type "publish" to confirm release* field, and click **Run workflow**

## Internal Recipes
`expect-arm64`, `expect-uv`, `ensure-pre-commit`, and `_test-run` are private
helpers used by public recipes. They are not intended to be invoked directly.

<sub>Saturday 15 August -  v0.1 &nbsp; | &nbsp; by [@ianscrivener](https://github.com/ianscrivener), GPT 5.6 Terra & Claude Sonnet 5
