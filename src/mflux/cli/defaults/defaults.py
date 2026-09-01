import os
from functools import lru_cache
from pathlib import Path

import platformdirs

BATTERY_PERCENTAGE_STOP_LIMIT = 5
CONTROLNET_STRENGTH = 0.4
DEFAULT_DEV_FILL_GUIDANCE = 30
DEFAULT_DEPTH_GUIDANCE = 10
DIMENSION_STEP_PIXELS = 16
GUIDANCE_SCALE = 3.5
GUIDANCE_SCALE_KONTEXT = 2.5
HEIGHT, WIDTH = 1024, 1024
IMAGE_STRENGTH = 0.4
DEFAULT_INFERENCE_STEPS = 25

# Keyed by the *canonical* AVAILABLE_MODELS key, never by alias: aliases are looked up
# through the registry by model_inference_steps() below. Keying this table by alias is
# what let `--model klein-4b` (and every other non-literal spelling) fall through to the
# 25-step FLUX.1-dev default.
MODEL_INFERENCE_STEPS = {
    "boogu-image-turbo": 4,
    "dev": 25,
    "dev-controlnet-canny": 25,
    "dev-controlnet-upscaler": 25,
    "dev-depth": 25,
    "dev-fill": 25,
    "dev-fill-catvton": 25,
    "dev-kontext": 25,
    "dev-redux": 25,
    "ernie-image": 50,
    "ernie-image-turbo": 8,
    "fibo": 50,
    "fibo-edit": 50,
    "fibo-edit-rmbg": 10,
    "fibo-lite": 8,
    "flux2-klein-4b": 4,
    "flux2-klein-9b": 4,
    "flux2-klein-9b-kv": 4,
    "flux2-klein-base-4b": 50,
    "flux2-klein-base-9b": 50,
    "ideogram-4-fp8": 20,
    "krea-2": 8,
    "krea-dev": 25,
    "lens-turbo": 4,
    "qwen-image": 20,
    "qwen-image-edit": 20,
    "schnell": 4,
    "schnell-controlnet-canny": 4,
    "z-image": 50,
    "z-image-turbo": 9,
    "z-image-turbo-controlnet-union-2.1": 8,
}


QUANTIZE_CHOICES = [3, 5, 4, 6, 8]

if os.environ.get("MFLUX_CACHE_DIR"):
    MFLUX_CACHE_DIR = Path(os.environ["MFLUX_CACHE_DIR"]).resolve()
else:
    MFLUX_CACHE_DIR = Path(platformdirs.user_cache_dir(appname="mflux"))

MFLUX_LORA_CACHE_DIR = MFLUX_CACHE_DIR / "loras"


@lru_cache(maxsize=1)
def model_choices() -> tuple[str, ...]:
    # Every spelling of a built-in model: canonical AVAILABLE_MODELS keys plus their
    # aliases. Hand-maintaining this list is what made `--model lens-turbo` (and ~40 other
    # valid names) resolve to model_path='lens-turbo' and die with "Model not found" —
    # anything missing here is treated as a local checkpoint directory.
    # Imported lazily for the same cycle reason as model_inference_steps() below.
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    names = set(AVAILABLE_MODELS)
    for config in AVAILABLE_MODELS.values():
        names.update(config.aliases)
    return tuple(sorted(names))


@lru_cache(maxsize=1)
def canonical_model_choices() -> tuple[str, ...]:
    # The canonical key of each registry entry, in registry (priority) order — the short
    # list worth printing in --help, where every alias would be noise.
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    return tuple(AVAILABLE_MODELS)


def model_inference_steps(model_name: str | None, base_model: str | None = None) -> int:
    # Accepts a canonical key, any alias, or a HuggingFace repo id. A third-party checkpoint
    # or local path takes the count of the entry it resolves to: --base-model when given,
    # otherwise the alias in its name, the same inference the CLIs use for the geometry
    # (#698). Only a name that says nothing about its lineage falls back to
    # DEFAULT_INFERENCE_STEPS, as does any registry entry with no declared count (the
    # SeedVR2 upscalers never step).
    if model_name is None:
        # `--base-model schnell` with no --model asks for the base itself, the same
        # promotion ConfigResolution makes.
        if base_model is None:
            return DEFAULT_INFERENCE_STEPS
        model_name = base_model

    if model_name in MODEL_INFERENCE_STEPS:
        return MODEL_INFERENCE_STEPS[model_name]

    # Imported lazily: model_config pulls in mlx, and weight_loader / lora_resolution
    # already import this module, so a module-level import would close a cycle.
    from mflux.models.common.config.model_config import AVAILABLE_MODELS

    # Aliases are unique across the registry, so match them before repo ids.
    for key, config in AVAILABLE_MODELS.items():
        if model_name in config.aliases:
            return MODEL_INFERENCE_STEPS.get(key, DEFAULT_INFERENCE_STEPS)

    # Several entries share one repo id (z-image-turbo and its ControlNet, the FLUX.1-dev
    # ControlNets). Break the tie the same way ConfigResolution's exact-match rule does —
    # base variant first, then priority — so the step count matches the config that
    # actually gets built.
    for key, config in sorted(
        AVAILABLE_MODELS.items(), key=lambda kv: (kv[1].controlnet_model is not None, kv[1].priority)
    ):
        if model_name == config.model_name:
            return MODEL_INFERENCE_STEPS.get(key, DEFAULT_INFERENCE_STEPS)

    # A custom checkpoint of a known model. --base-model is the explicit signal (an
    # invalid value keeps the default here; the parser's own validation rejects it
    # right after this). Without it, the only accepted implicit signal is the
    # checkpoint's basename STARTING with a full alias at a word boundary: unlike the
    # geometry (recoverable with --base-model, and family-scoped where it is inferred),
    # a wrong step count has no error to bounce off — a dev finetune named
    # my-schnell-style-adapter silently under-steps 6x, which is worse than the
    # generic 25 over-running. "klein-9b" buried mid-name or in a directory component
    # is not a signal.
    if base_model is not None:
        for key, config in AVAILABLE_MODELS.items():
            if base_model == config.model_name or base_model in config.aliases:
                return MODEL_INFERENCE_STEPS.get(key, DEFAULT_INFERENCE_STEPS)
        return DEFAULT_INFERENCE_STEPS

    basename = model_name.rstrip("/").rsplit("/", 1)[-1].lower()
    matches = []
    for key, config in AVAILABLE_MODELS.items():
        for alias in config.aliases:
            if alias and basename.startswith(alias.lower()):
                rest = basename[len(alias) :]
                if rest == "" or not rest[0].isalnum():
                    matches.append((key, alias))
    if not matches:
        return DEFAULT_INFERENCE_STEPS
    key = max(matches, key=lambda match: len(match[1]))[0]  # longest alias wins, as in family inference
    return MODEL_INFERENCE_STEPS.get(key, DEFAULT_INFERENCE_STEPS)
