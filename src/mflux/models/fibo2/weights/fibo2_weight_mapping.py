from typing import List

from mflux.models.common.weights.mapping.weight_mapping import WeightTarget
from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms
from mflux.models.flux2.weights.flux2_weight_mapping import Flux2WeightMapping


class Fibo2WeightMapping:
    # Each block is listed by its index, so a checkpoint that lacks one is rejected at load time instead of
    # leaving it at its random initial values.
    NUM_LAYERS = 44
    NUM_REFINER_LAYERS = 2
    NUM_PERCEIVER_LAYERS = 2
    INJECTION_LAYER_IDS = (6, 14, 22, 30, 38)
    # Fibo 2.0 reads Qwen3-VL up to the output of layer 35 of 36, so the last layer is not loaded
    NUM_TEXT_LAYERS = 35
    NUM_VISION_BLOCKS = 24
    NUM_DEEPSTACK_MERGERS = 3

    _ATTENTION = ["to_q.weight", "to_k.weight", "to_v.weight", "to_out.0.weight", "norm_q.weight", "norm_k.weight"]
    _BLOCK = [
        *(f"attention.{name}" for name in _ATTENTION),
        "attention_norm1.weight",
        "attention_norm2.weight",
        "ffn_norm1.weight",
        "ffn_norm2.weight",
        "feed_forward.w1.weight",
        "feed_forward.w2.weight",
        "feed_forward.w3.weight",
        "adaLN_modulation.0.weight",
        "adaLN_modulation.0.bias",
    ]
    _INJECTION = [
        *(f"cross_attention.{name}" for name in _ATTENTION),
        "cross_attn_norm1.weight",
        "cross_attn_norm2.weight",
        "cross_attn_gate",
    ]
    _PERCEIVER_LAYER = [
        *(f"cross_attention.{name}" for name in _ATTENTION),
        *(f"self_attention.{name}" for name in _ATTENTION),
        "cross_attn_norm.weight",
        "self_attn_norm.weight",
        "ffn_norm.weight",
        "ffn.w1.weight",
        "ffn.w2.weight",
        "ffn.w3.weight",
    ]
    _TEXT_LAYER = [
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
    ]
    _VISION_BLOCK = [
        f"{name}.{param}"
        for name in ("norm1", "norm2", "attn.qkv", "attn.proj", "mlp.linear_fc1", "mlp.linear_fc2")
        for param in ("weight", "bias")
    ]
    _MERGER = [f"{name}.{param}" for name in ("norm", "linear_fc1", "linear_fc2") for param in ("weight", "bias")]

    @staticmethod
    def get_transformer_mapping() -> List[WeightTarget]:
        m = Fibo2WeightMapping
        names = ["x_embedder.weight", "x_embedder.bias", "final_layer.linear.weight", "final_layer.linear.bias"]
        names += ["perceiver.latent_queries", "perceiver.norm_out.weight"]
        for i in range(m.NUM_REFINER_LAYERS):
            names += [f"noise_refiner.{i}.{name}" for name in m._BLOCK]
        for i in range(m.NUM_PERCEIVER_LAYERS):
            names += [f"perceiver.layers.{i}.{name}" for name in m._PERCEIVER_LAYER]
        for i in range(m.NUM_LAYERS):
            names += [f"layers.{i}.{name}" for name in m._BLOCK]
            if i in m.INJECTION_LAYER_IDS:
                names += [f"layers.{i}.{name}" for name in m._INJECTION]
        renamed = {
            # Z-Image's TimestepEmbedder and FinalLayer name their layers instead of indexing a Sequential
            "t_embedder.linear1.weight": "t_embedder.mlp.0.weight",
            "t_embedder.linear1.bias": "t_embedder.mlp.0.bias",
            "t_embedder.linear2.weight": "t_embedder.mlp.2.weight",
            "t_embedder.linear2.bias": "t_embedder.mlp.2.bias",
            "final_layer.adaLN_modulation.0.weight": "final_layer.adaLN_modulation.1.weight",
            "final_layer.adaLN_modulation.0.bias": "final_layer.adaLN_modulation.1.bias",
        }
        return [WeightTarget(to_pattern=name, from_pattern=[name]) for name in names] + [
            WeightTarget(to_pattern=target, from_pattern=[source]) for target, source in renamed.items()
        ]

    @staticmethod
    def get_text_encoder_mapping() -> List[WeightTarget]:
        m = Fibo2WeightMapping
        names = ["language_model.embed_tokens.weight", "language_model.norm.weight"]
        for i in range(m.NUM_TEXT_LAYERS):
            names += [f"language_model.layers.{i}.{name}" for name in m._TEXT_LAYER]
        names += ["visual.patch_embed.proj.bias", "visual.pos_embed.weight"]
        for i in range(m.NUM_VISION_BLOCKS):
            names += [f"visual.blocks.{i}.{name}" for name in m._VISION_BLOCK]
        names += [f"visual.merger.{name}" for name in m._MERGER]
        for i in range(m.NUM_DEEPSTACK_MERGERS):
            names += [f"visual.deepstack_merger_list.{i}.{name}" for name in m._MERGER]
        return [WeightTarget(to_pattern=name, from_pattern=[f"model.{name}"]) for name in names] + [
            WeightTarget(
                to_pattern="visual.patch_embed.proj.weight",
                from_pattern=["model.visual.patch_embed.proj.weight"],
                transform=WeightTransforms.transpose_conv3d_weight,
            )
        ]

    @staticmethod
    def get_vae_mapping() -> List[WeightTarget]:
        # The FLUX.2 VAE: FLUX.2 klein's weights, stored in float32
        return Flux2WeightMapping.get_vae_mapping()
