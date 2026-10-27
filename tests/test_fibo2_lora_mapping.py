import pytest
from mlx import nn

from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2Transformer
from mflux.models.fibo2.weights.fibo2_lora_mapping import Fibo2LoRAMapping

pytestmark = pytest.mark.fast

# The MLX names that differ from the diffusers names a LoRA is saved with
DIFFUSERS_NAMES = {
    "t_embedder.linear1": "t_embedder.mlp.0",
    "t_embedder.linear2": "t_embedder.mlp.2",
    "final_layer.adaLN_modulation.0": "final_layer.adaLN_modulation.1",
}


class TestFibo2LoRAMapping:
    def test_every_linear_layer_is_reached_from_its_diffusers_key(self):
        # diffusers' FIBO 2 trainer adapts every linear layer ("all-linear"); each must land on its own MLX module
        for path in TestFibo2LoRAMapping._linear_paths():
            source = DIFFUSERS_NAMES.get(path, path)
            for prefix, suffix in (("transformer.", ".lora_A.weight"), ("", ".lora_B.default.weight")):
                assert TestFibo2LoRAMapping._resolve(f"{prefix}{source}{suffix}") == path

    def test_comfyui_style_keys_are_accepted(self):
        key = "diffusion_model.layers.6.cross_attention.to_q.lora_B.weight"
        assert TestFibo2LoRAMapping._resolve(key) == "layers.6.cross_attention.to_q"

    @staticmethod
    def _linear_paths() -> list[str]:
        # The full-size layout: 44 layers, 5 of them with cross-attention, 2 refiner and 2 Perceiver layers
        transformer = Fibo2Transformer()
        return [name for name, module in transformer.named_modules() if isinstance(module, nn.Linear)]

    @staticmethod
    def _resolve(key: str) -> str | None:
        for target in Fibo2LoRAMapping.get_mapping():
            for pattern in target.possible_up_patterns + target.possible_down_patterns:
                block = LoRALoader._match_pattern(key, pattern)
                if block is not None:
                    return target.model_path.format(block=block)
        return None


class TestFibo2ComfyFusedKeys:
    # ComfyUI's Fibo 2 layout (Comfy-Org/Fibo-2.0), which ai-toolkit saves its LoRAs in, fuses some projections
    FUSED = {
        "layers.3.attention.qkv": ["layers.3.attention.to_q", "layers.3.attention.to_k", "layers.3.attention.to_v"],
        "noise_refiner.1.feed_forward.w13": ["noise_refiner.1.feed_forward.w1", "noise_refiner.1.feed_forward.w3"],
        "layers.6.cross_attention.to_kv": ["layers.6.cross_attention.to_k", "layers.6.cross_attention.to_v"],
        "perceiver.layers.0.self_attention.to_kv": [
            "perceiver.layers.0.self_attention.to_k",
            "perceiver.layers.0.self_attention.to_v",
        ],  # fmt: off
        "perceiver.layers.1.ffn.w13": ["perceiver.layers.1.ffn.w1", "perceiver.layers.1.ffn.w3"],
        "layers.3.attention.out": ["layers.3.attention.to_out.0"],
        "perceiver.layers.1.cross_attention.to_out": ["perceiver.layers.1.cross_attention.to_out.0"],
    }

    def test_a_fused_key_reaches_every_part(self):
        for source, parts in TestFibo2ComfyFusedKeys.FUSED.items():
            key = f"diffusion_model.{source}.lora_B.weight"
            reached = sorted(
                target.model_path.format(block=block)
                for target in Fibo2LoRAMapping.get_mapping()
                for pattern in target.possible_up_patterns
                if (block := LoRALoader._match_pattern(key, pattern)) is not None
            )
            assert reached == sorted(parts), source

    def test_a_fused_lora_gives_each_part_its_slice(self, tmp_path):
        # Gate first in w13 and q, k, v in order in qkv: the split LoRA must change each part as the fused one would
        import mlx.core as mx

        transformer = Fibo2Transformer(dim=128, n_heads=2, n_layers=1, cap_feat_dim=128, injection_layer_ids=(0,), axes_dims=(16, 24, 24))  # fmt: off
        block = transformer.layers[0]
        rank, dim, hidden = 4, 128, block.feed_forward.w1.weight.shape[0]
        down_qkv = mx.random.normal((rank, dim), key=mx.random.key(0))
        up_qkv = mx.random.normal((3 * dim, rank), key=mx.random.key(1))
        down_ffn = mx.random.normal((rank, dim), key=mx.random.key(2))
        up_ffn = mx.random.normal((2 * hidden, rank), key=mx.random.key(3))
        path = tmp_path / "comfy_fibo2.safetensors"
        mx.save_safetensors(str(path), {
            "diffusion_model.layers.0.attention.qkv.lora_A.weight": down_qkv,
            "diffusion_model.layers.0.attention.qkv.lora_B.weight": up_qkv,
            "diffusion_model.layers.0.feed_forward.w13.lora_A.weight": down_ffn,
            "diffusion_model.layers.0.feed_forward.w13.lora_B.weight": up_ffn,
        })  # fmt: off
        x = mx.random.normal((2, dim), key=mx.random.key(4))
        before = {name: getattr(block.attention, f"to_{name}")(x) for name in "qkv"} | {n: getattr(block.feed_forward, n)(x) for n in ("w1", "w3")}  # fmt: off

        LoRALoader.load_and_apply_lora(Fibo2LoRAMapping.get_mapping(), transformer, [str(path)], [1.0], bake_lora=False)

        fused_qkv = x @ down_qkv.T @ up_qkv.T
        fused_ffn = x @ down_ffn.T @ up_ffn.T
        for i, name in enumerate("qkv"):
            delta = getattr(block.attention, f"to_{name}")(x) - before[name]
            assert mx.allclose(delta, fused_qkv[:, i * dim : (i + 1) * dim], atol=1e-3).item(), name
        for i, name in enumerate(("w1", "w3")):
            delta = getattr(block.feed_forward, name)(x) - before[name]
            assert mx.allclose(delta, fused_ffn[:, i * hidden : (i + 1) * hidden], atol=1e-3).item(), name

    def test_comfyui_padding_of_the_feed_forward_is_cut_off(self):
        # Bria's feed-forward is 9536 wide; ComfyUI's repack pads each half of w13 and the input of w2 to 9728 with zeros
        import mlx.core as mx

        hidden, padded, rank = Fibo2LoRAMapping.FFN_HIDDEN, 9728, 4
        assert hidden == 9536
        up = mx.concatenate([mx.ones((hidden, rank)), mx.zeros((padded - hidden, rank)), 2 * mx.ones((hidden, rank)), mx.zeros((padded - hidden, rank))])  # fmt: off
        gate, up_part = (Fibo2LoRAMapping._part(up, i, 2, hidden) for i in range(2))
        assert gate.shape == (hidden, rank) and float(gate.min()) == 1.0
        assert up_part.shape == (hidden, rank) and float(up_part.min()) == 2.0
        assert Fibo2LoRAMapping._unpad_down(mx.ones((rank, padded))).shape == (rank, hidden)

    def test_kohya_names_norm_and_bias_patches_load_end_to_end(self, tmp_path):
        # A kohya file (lora_down/lora_up, alpha), a norm .diff under ComfyUI's q_norm name and a .diff_b on a biased
        # layer, loaded and baked at scale 0.5 on the small transformer
        import mlx.core as mx

        transformer = Fibo2Transformer(dim=128, n_heads=2, n_layers=1, cap_feat_dim=128, injection_layer_ids=(0,), axes_dims=(16, 24, 24))  # fmt: off
        block = transformer.layers[0]
        rank, scale = 4, 0.5
        down = mx.random.normal((rank, 128), key=mx.random.key(0))
        up = mx.random.normal((128, rank), key=mx.random.key(1))
        norm_diff = mx.random.normal(block.attention.norm_q.weight.shape, key=mx.random.key(2))
        bias_diff = mx.random.normal(block.adaLN_modulation[0].bias.shape, key=mx.random.key(3))
        before_out = block.attention.to_out[0].weight
        before_norm = block.attention.norm_q.weight
        before_bias = block.adaLN_modulation[0].bias
        path = tmp_path / "kohya_fibo2.safetensors"
        mx.save_safetensors(str(path), {
            "diffusion_model.layers.0.attention.out.lora_down.weight": down,
            "diffusion_model.layers.0.attention.out.lora_up.weight": up,
            "diffusion_model.layers.0.attention.out.alpha": mx.array(float(rank)),
            "diffusion_model.layers.0.attention.q_norm.diff": norm_diff,
            "diffusion_model.layers.0.adaLN_modulation.0.diff_b": bias_diff,
        })  # fmt: off

        LoRALoader.load_and_apply_lora(
            Fibo2LoRAMapping.get_mapping(), transformer, [str(path)], [scale], bake_lora=True
        )

        block = transformer.layers[0]
        assert mx.allclose(block.attention.to_out[0].weight - before_out, scale * (up @ down), atol=2e-2).item()
        assert mx.allclose(block.attention.norm_q.weight - before_norm, scale * norm_diff, atol=2e-2).item()
        assert mx.allclose(block.adaLN_modulation[0].bias - before_bias, scale * bias_diff, atol=2e-2).item()

    def test_every_w2_target_drops_comfyui_padding(self):
        w2 = [t for t in Fibo2LoRAMapping.get_mapping() if t.model_path.endswith(("feed_forward.w2", "ffn.w2"))]
        assert w2 and all(t.down_transform is Fibo2LoRAMapping._unpad_down for t in w2)

    def test_the_final_norm_takes_no_patch(self):
        # final_layer.norm is a LayerNorm without weight: a patch target there would stop the whole load
        assert not any(t.model_path == "final_layer.norm" for t in Fibo2LoRAMapping.get_mapping())
