from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.layer.linear_lora_layer import LoRALinear
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2Transformer
from mflux.models.fibo2.weights.fibo2_lora_mapping import Fibo2LoRAMapping

pytestmark = pytest.mark.fast

# The tensor names of akhaliq/Fibo-2.0-8step-smd @ d3594f8a567a update_5500/lora_student_ema.safetensors, read from
# its safetensors header: a PEFT model's state dict (base_model.model.*) over the attention and feed-forward layers
SMD_KEYS = (Path(__file__).parents[1] / "resources" / "checkpoint_keys" / "fibo2_smd_lora.txt").read_text().split()


class Fibo2LoRA:
    @staticmethod
    def transformer() -> Fibo2Transformer:
        # The real block counts and injection blocks, at tiny widths
        return Fibo2Transformer(in_channels=4, dim=64, n_heads=2, cap_feat_dim=48, axes_dims=(8, 12, 12))

    @staticmethod
    def target_of(key: str) -> str | None:
        for mapping in LoRALoader._build_pattern_mappings(Fibo2LoRAMapping.get_mapping()):
            block = LoRALoader._match_pattern(key, mapping.source_pattern)
            if block is not None:
                return mapping.target_path.replace("{block}", str(block))
        return None

    @staticmethod
    def module(transformer: Fibo2Transformer, path: str):
        return LoRALoader._get_target_module(transformer, path)


def test_every_key_of_the_published_smd_lora_has_a_layer():
    assert len(SMD_KEYS) == 728
    transformer = Fibo2LoRA.transformer()
    targets = {key: Fibo2LoRA.target_of(key) for key in SMD_KEYS}
    assert [key for key, target in targets.items() if target is None] == []
    for target in set(targets.values()):
        assert hasattr(Fibo2LoRA.module(transformer, target), "weight"), target


@pytest.mark.parametrize(
    ("key", "target"),
    [
        # diffusers' trainer adapts every linear layer and saves them under transformer.
        ("transformer.x_embedder.lora_A.weight", "x_embedder"),
        ("transformer.t_embedder.mlp.0.lora_A.weight", "t_embedder.linear1"),
        ("transformer.t_embedder.mlp.2.lora_B.weight", "t_embedder.linear2"),
        ("transformer.final_layer.adaLN_modulation.1.lora_A.weight", "final_layer.adaLN_modulation.0"),
        ("transformer.final_layer.linear.lora_B.weight", "final_layer.linear"),
        ("transformer.layers.43.adaLN_modulation.0.lora_A.weight", "layers.43.adaLN_modulation.0"),
        ("transformer.noise_refiner.1.attention.to_out.0.lora_B.weight", "noise_refiner.1.attention.to_out.0"),
        ("transformer.layers.38.cross_attention.to_v.lora_A.weight", "layers.38.cross_attention.to_v"),
        ("transformer.perceiver.layers.1.ffn.w2.lora_B.weight", "perceiver.layers.1.ffn.w2"),
        ("diffusion_model.layers.0.feed_forward.w1.lora_up.weight", "layers.0.feed_forward.w1"),
        ("layers.0.attention.to_q.alpha", "layers.0.attention.to_q"),
    ],
)
def test_diffusers_peft_and_comfyui_names_reach_their_layer(key, target):
    assert Fibo2LoRA.target_of(key) == target


@pytest.mark.parametrize("bake_lora", [True, False])
def test_a_lora_changes_exactly_the_layers_it_names(tmp_path, monkeypatch, bake_lora):
    monkeypatch.setattr(ModelConfig, "precision", mx.float32)
    rng = np.random.default_rng(0)
    paths = {
        "transformer.layers.6.cross_attention.to_k": (64, 48),
        "transformer.t_embedder.mlp.2": (64, 1024),
        "transformer.final_layer.adaLN_modulation.1": (64, 64),
    }
    tensors, expected = {}, {}
    for path, (out_dim, in_dim) in paths.items():
        down, up = rng.normal(size=(2, in_dim)).astype(np.float32), rng.normal(size=(out_dim, 2)).astype(np.float32)
        tensors[f"{path}.lora_A.weight"], tensors[f"{path}.lora_B.weight"] = mx.array(down), mx.array(up)
        expected[Fibo2LoRA.target_of(f"{path}.lora_A.weight")] = 0.5 * up @ down
    lora_file = tmp_path / "fibo2_lora.safetensors"
    mx.save_safetensors(str(lora_file), tensors)
    transformer = Fibo2LoRA.transformer()
    before = {target: np.array(Fibo2LoRA.module(transformer, target).weight) for target in expected}

    LoRALoader.load_and_apply_lora(
        lora_mapping=Fibo2LoRAMapping.get_mapping(),
        transformer=transformer,
        lora_paths=[str(lora_file)],
        lora_scales=[0.5],
        bake_lora=bake_lora,
    )

    for target, delta in expected.items():
        layer = Fibo2LoRA.module(transformer, target)
        if bake_lora:
            np.testing.assert_allclose(np.array(layer.weight), before[target] + delta, rtol=1e-5, atol=1e-5)
        else:
            assert isinstance(layer, LoRALinear)
            x = mx.array(rng.normal(size=(3, before[target].shape[1])).astype(np.float32))
            np.testing.assert_allclose(np.array(layer(x)), np.array(layer.linear(x)) + np.array(x) @ delta.T, rtol=1e-4, atol=1e-4)  # fmt: off
