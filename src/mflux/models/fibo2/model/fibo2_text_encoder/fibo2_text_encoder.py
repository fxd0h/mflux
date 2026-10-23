import mlx.core as mx
import numpy as np
from mlx import nn

from mflux.models.qwen21.model.qwen21_text_encoder.text_encoder import QwenImage21TextEncoder

# briaai/Fibo-2.0-dev text_encoder/config.json (Qwen3-VL-4B-Instruct), in the layout QwenImage21TextEncoder reads. Fibo
# 2.0 reads the hidden states up to the output of layer 35, so the 36th layer is not built.
FIBO2_TEXT_ENCODER_CONFIG = {
    "image_token_id": 151655,
    "text_config": {
        "vocab_size": 151936,
        "hidden_size": 2560,
        "intermediate_size": 9728,
        "num_hidden_layers": 35,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 262144,
        "rms_norm_eps": 1e-6,
        "rope_theta": 5000000,
        "rope_scaling": {"mrope_section": [24, 20, 20]},
        "attention_bias": False,
    },
    "vision_config": {
        "depth": 24,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_heads": 16,
        "in_channels": 3,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2560,
        "deepstack_visual_indexes": [5, 11, 17],
    },
}


class Fibo2TextEncoder(nn.Module):
    # Qwen3-VL, read from inside: Fibo 2.0 conditions on hidden states of intermediate layers, so neither the
    # generation head nor the last layer is loaded.
    def __init__(self, config: dict = FIBO2_TEXT_ENCODER_CONFIG):
        super().__init__()
        # Qwen-Image-2.1's encoder builds the same Qwen3-VL, with the activations the reference uses
        encoder = QwenImage21TextEncoder(config)
        self.language_model = encoder.language_model
        self.visual = encoder.visual
        self.image_token_id = config["image_token_id"]
        self.merge_size = config["vision_config"]["spatial_merge_size"]

    def __call__(
        self,
        input_ids: mx.array,
        layers: tuple[int, ...],
        pixel_values: mx.array | None = None,
        image_grid_thw: mx.array | None = None,
    ) -> dict[int, mx.array]:
        # Returns the hidden states numbered as in transformers: 0 is the input embeddings, k the output of layer k
        model = self.language_model
        hidden = model.embed_tokens(input_ids)
        image_indices = mx.array(np.flatnonzero(np.asarray(input_ids[0]) == self.image_token_id), dtype=mx.int32)
        deepstack = None
        if pixel_values is not None:
            features, deepstack = self.visual(pixel_values.astype(hidden.dtype), image_grid_thw, return_deepstack=True)
            if features.shape[0] != image_indices.size:
                raise ValueError("Vision feature count does not match the image placeholders.")
            hidden[:, image_indices] = features.astype(hidden.dtype)[None]
        positions = QwenImage21TextEncoder.mrope_positions(
            input_ids, image_grid_thw, self.image_token_id, self.merge_size
        )
        rope = model.rotary_emb(hidden, positions)
        index = mx.arange(hidden.shape[1])
        mask = (index[:, None] >= index[None, :])[None, None]
        states = {0: hidden} if 0 in layers else {}
        for number, layer in enumerate(model.layers[: max(layers)], start=1):
            hidden, _ = layer(hidden, attention_mask=mask, position_embeddings=rope)
            if number in layers:
                states[number] = hidden
            # The vision features of the first layers are added to the image tokens after the layer's output is read
            if deepstack is not None and number <= len(deepstack):
                hidden = hidden.at[:, image_indices].add(deepstack[number - 1].astype(hidden.dtype)[None])
        return states
