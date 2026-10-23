import mlx.core as mx
import numpy as np
import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLModel

from mflux.models.common.weights.mapping.weight_transforms import WeightTransforms
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_prompt_encoder import Fibo2PromptEncoder
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_encoder import Fibo2TextEncoder

pytestmark = pytest.mark.fast

# A tiny Qwen3-VL from transformers on the CPU, against the same weights in Fibo2TextEncoder: Fibo 2.0 reads the hidden
# states of intermediate layers, numbered as transformers numbers them (0 is the input embeddings), with the vision
# features in the first layers (deepstack) and the image positions of the multimodal RoPE.
CONFIG = dict(
    image_token_id=99,
    vision_start_token_id=98,
    vision_end_token_id=97,
    text_config=dict(
        vocab_size=128,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        intermediate_size=64,
        head_dim=16,
        max_position_embeddings=2048,
        rope_theta=5000000.0,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_scaling=dict(rope_type="default", mrope_section=[4, 2, 2], mrope_interleaved=True),
    ),
    vision_config=dict(
        depth=3,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        patch_size=2,
        temporal_patch_size=2,
        in_channels=3,
        spatial_merge_size=2,
        out_hidden_size=32,
        num_position_embeddings=16,
        deepstack_visual_indexes=[0, 1, 2],
        hidden_act="gelu_pytorch_tanh",
    ),
)


class TinyQwen3VL:
    @staticmethod
    def pair() -> tuple[Qwen3VLModel, Fibo2TextEncoder]:
        torch.manual_seed(0)
        reference = Qwen3VLModel(Qwen3VLConfig(**CONFIG)).to(device="cpu", dtype=torch.float32).eval()
        model = Fibo2TextEncoder(CONFIG)
        weights = []
        for key, value in reference.state_dict().items():
            weight = mx.array(value.detach().float().numpy())
            if key.endswith("patch_embed.proj.weight"):
                weight = WeightTransforms.transpose_conv3d_weight(weight)
            weights.append((key, weight))
        model.load_weights(weights, strict=False)
        return reference, model


@pytest.mark.parametrize(
    ("ids", "grids"),
    [
        ([1, 2, 3, 4, 5], None),
        ([1, 98, 99, 99, 99, 99, 97, 5], [[1, 4, 4]]),
        ([1, 98, 99, 99, 99, 99, 97, 98, 99, 99, 97, 5], [[1, 4, 4], [1, 2, 4]]),
    ],
)
def test_the_hidden_states_match_transformers_layer_by_layer(ids, grids):
    reference, model = TinyQwen3VL.pair()
    input_ids = torch.tensor([ids])
    kwargs, mlx_kwargs = {}, {}
    if grids is not None:
        grid = torch.tensor(grids)
        pixels = torch.randn(int(grid.prod(-1).sum()), 24)
        kwargs = dict(pixel_values=pixels, image_grid_thw=grid, mm_token_type_ids=(input_ids == 99).int())
        mlx_kwargs = dict(pixel_values=mx.array(pixels.numpy()), image_grid_thw=mx.array(grid.numpy()))
    with torch.no_grad():
        expected = reference(input_ids=input_ids, use_cache=False, output_hidden_states=True, **kwargs).hidden_states
    device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        # The last layer's output is not compared: transformers replaces it with the final norm, which Fibo 2.0
        # never reads
        actual = model(mx.array(input_ids.numpy()), layers=(0, 1, 2, 3), **mlx_kwargs)
    finally:
        mx.set_default_device(device)
    for layer in (0, 1, 2, 3):
        np.testing.assert_allclose(np.array(actual[layer][0]), expected[layer][0].numpy(), atol=2e-5, rtol=1e-4)


def test_unit_rms_is_taken_in_float32_and_kept_in_the_input_dtype():
    states = mx.array(np.random.default_rng(0).normal(scale=40.0, size=(1, 3, 64)).astype(np.float32)).astype(mx.bfloat16)  # fmt: off
    normalized = Fibo2PromptEncoder._unit_rms(states)
    values = np.array(states.astype(mx.float32), dtype=np.float64)
    expected = values / np.sqrt((values**2).mean(axis=-1, keepdims=True) + 1e-6)
    assert normalized.dtype == mx.bfloat16
    np.testing.assert_allclose(np.array(normalized.astype(mx.float32)), expected, rtol=1e-2)
