import json
import math
import struct

import mlx.core as mx


class IrisFewStepScheduler:
    # The sampler of few-step LoRAs distilled from the base model (DMD2, as in pentacoxian-dev/iris-3b-turbo-lora):
    # steps evenly spaced from pure noise, shifted, one forward per step without CFG, and fresh noise between steps.
    # Such a LoRA records turbo_steps and shift in its safetensors metadata, which is how the reference sampler
    # (pentaCoxian/iris-3b-turbo, generate_turbo) knows to switch to it.
    NUM_TRAIN_TIMESTEPS = 1000
    MAX_STEPS = 100

    def __init__(self, num_inference_steps: int, shift: float, seed: int):
        grid = [1.0 - i / num_inference_steps for i in range(num_inference_steps)]
        self.sigmas = [shift * s / (1 + (shift - 1) * s) for s in grid]
        self.num_inference_steps = num_inference_steps
        self.key = mx.random.key(seed)

    @staticmethod
    def from_lora_files(paths: list[str] | None, scales: list[float] | None = None) -> tuple[int, float] | None:
        # (steps, shift) of the first LoRA that is a few-step adapter, read from the file header only. A LoRA at scale
        # 0 changes no weight, so it does not change the sampler either.
        scales = scales or [1.0] * len(paths or [])
        for path, scale in zip(paths or [], scales):
            extra = IrisFewStepScheduler._metadata(path).get("extra", {})
            if "turbo_steps" in extra and scale != 0:
                return IrisFewStepScheduler._settings(path, extra)
        return None

    @staticmethod
    def _settings(path: str, extra: dict) -> tuple[int, float]:
        try:
            steps, shift = int(extra["turbo_steps"]), float(extra.get("shift", 4.0))
        except (TypeError, ValueError):
            steps, shift = 0, 0.0
        if not 1 <= steps <= IrisFewStepScheduler.MAX_STEPS or not math.isfinite(shift) or shift <= 0:
            raise ValueError(f"{path} records turbo_steps={extra.get('turbo_steps')!r} and shift={extra.get('shift')!r}; a few-step LoRA needs 1 to {IrisFewStepScheduler.MAX_STEPS} steps and a positive shift.")  # fmt: off
        return steps, shift

    def model_time(self, step: int) -> mx.array:
        return mx.array([self.sigmas[step]], dtype=mx.float32) * IrisFewStepScheduler.NUM_TRAIN_TIMESTEPS

    def step(self, latents: mx.array, velocity: mx.array, step: int) -> tuple[mx.array, mx.array]:
        # Returns the next state and the clean prediction (the final image on the last step)
        clean = latents - self.sigmas[step] * velocity
        if step + 1 == self.num_inference_steps:
            return clean, clean
        self.key, noise_key = mx.random.split(self.key)
        noise = mx.random.normal(latents.shape, dtype=mx.float32, key=noise_key)
        next_sigma = self.sigmas[step + 1]
        return (1 - next_sigma) * clean + next_sigma * noise, clean

    @staticmethod
    def _metadata(path: str) -> dict:
        with open(path, "rb") as file:
            length = struct.unpack("<Q", file.read(8))[0]
            header = json.loads(file.read(length))
        metadata = dict(header.get("__metadata__") or {})
        if "extra" in metadata:
            try:
                metadata["extra"] = json.loads(metadata["extra"])
            except (TypeError, ValueError):
                metadata["extra"] = {}
        return metadata
