import json
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2Transformer
from mflux.models.flux2.latent_creator.flux2_latent_creator import Flux2LatentCreator

pytestmark = pytest.mark.fast
# A tiny diffusers BriaFibo2Transformer2DModel (4 layers, injection blocks 1 and 3, 4 to 8 gist tokens) with random
# weights, its inputs and its float32 CPU outputs, from diffusers main at 48cdbae (2026-10-21)
FIXTURE = Path(__file__).parents[1] / "resources" / "fibo2_transformer_parity.npz"
CASES = ("txt2img", "short_text", "long_text", "edit", "multi_edit")


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURE, allow_pickle=False) as archive:
        return dict(archive)


class Fibo2Parity:
    # diffusers indexes these layers inside an nn.Sequential, Z-Image's modules in mflux name them
    RENAMED = {
        "t_embedder.mlp.0.": "t_embedder.linear1.",
        "t_embedder.mlp.2.": "t_embedder.linear2.",
        "final_layer.adaLN_modulation.1.": "final_layer.adaLN_modulation.0.",
    }

    @staticmethod
    def model(data: dict[str, np.ndarray], dtype: mx.Dtype, monkeypatch: pytest.MonkeyPatch) -> Fibo2Transformer:
        monkeypatch.setattr(ModelConfig, "precision", dtype)
        model = Fibo2Transformer(**json.loads(str(data["config"])))
        weights = []
        for key, value in data.items():
            if key.startswith("weight/"):
                name = key.removeprefix("weight/")
                for old, new in Fibo2Parity.RENAMED.items():
                    name = name.replace(old, new)
                weights.append((name, mx.array(value).astype(dtype)))
        model.load_weights(weights, strict=True)
        return model

    @staticmethod
    def run(data: dict[str, np.ndarray], model: Fibo2Transformer, case: str, dtype: mx.Dtype) -> np.ndarray:
        length, num_contexts, sigma = data[f"case/{case}"].tolist()
        latents = Flux2LatentCreator.patchify_latents(mx.array(data["input/latents"]))
        contexts = [
            Flux2LatentCreator.patchify_latents(mx.array(data[f"input/context_{k}"])).astype(dtype)
            for k in range(int(num_contexts))
        ]
        text = model.encode_text(mx.array(data[f"input/text_{int(length)}"]))
        velocity = model(latents, mx.array([sigma], dtype=mx.float32), text, contexts or None)
        return np.array(velocity.astype(mx.float32))

    @staticmethod
    def expected(data: dict[str, np.ndarray], case: str) -> np.ndarray:
        return np.array(Flux2LatentCreator.patchify_latents(mx.array(data[f"expected/{case}"])))


@pytest.mark.parametrize("case", CASES)
def test_float32_on_the_cpu_matches_diffusers(reference, monkeypatch, case):
    # On the CPU float32 matmuls run in full precision. What is left is the timestep embedding: torch's and MLX's
    # float32 sin and cos differ by an ulp at its large arguments (sigma * 1000), about 1e-4 at the output. A wrong
    # token order, RoPE plane or modulation gives 1e-1 or more.
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        actual = Fibo2Parity.run(reference, Fibo2Parity.model(reference, mx.float32, monkeypatch), case, mx.float32)
    finally:
        mx.set_default_device(device)
    expected = Fibo2Parity.expected(reference, case)

    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 1e-3


@pytest.mark.parametrize("case", CASES)
def test_bfloat16_matches_diffusers_to_bfloat16_rounding(reference, monkeypatch, case):
    # diffusers' own bfloat16 run of this random network is 3.8 to 5.8% off its float32 output (CPU and MPS alike)
    actual = Fibo2Parity.run(reference, Fibo2Parity.model(reference, mx.bfloat16, monkeypatch), case, mx.bfloat16)
    expected = Fibo2Parity.expected(reference, case)

    assert np.isfinite(actual).all()
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.08


def test_the_gist_count_follows_the_text_length(reference, monkeypatch):
    # 4 to 8 gist tokens in steps of 2 over text lengths 4 to 12, rounded up: what compute_gist_count gives
    perceiver = Fibo2Parity.model(reference, mx.float32, monkeypatch).perceiver

    assert [perceiver.gist_count(n) for n in (1, 4, 5, 6, 7, 9, 12, 100)] == [4, 4, 6, 6, 6, 8, 8, 8]


def test_an_injection_block_without_its_gate_ignores_the_text(reference, monkeypatch):
    # The cross-attention is gated by tanh(gate): with the gates at zero the injected text must not matter
    model = Fibo2Parity.model(reference, mx.float32, monkeypatch)
    for layer in model.layers:
        if layer.cross_attention is not None:
            layer.cross_attn_gate = mx.zeros((1,))
    latents = Flux2LatentCreator.patchify_latents(mx.array(reference["input/latents"]))
    text = mx.array(reference["input/text_6"])
    other = mx.concatenate([text[:, :1], mx.zeros_like(text[:, 1:])], axis=1)

    first = model(latents, mx.array([0.5]), model.encode_text(text))
    second = model(latents, mx.array([0.5]), model.encode_text(other))

    assert mx.array_equal(first, second).item()
