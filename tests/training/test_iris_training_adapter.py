from types import SimpleNamespace

import mlx.core as mx
import pytest
from mlx import nn
from mlx.utils import tree_flatten
from PIL import Image

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.training.lora.target_injector import inject_lora_targets
from mflux.models.common.training.state.training_spec import BlockRange, LoraLayersSpec, LoraTargetSpec
from mflux.models.common.training.trainer import TrainingTrainer
from mflux.models.iris.model.iris_scheduler.iris_training_schedule import IrisTrainingSchedule
from mflux.models.iris.model.iris_transformer.iris_transformer import IrisTransformer
from mflux.models.iris.training_adapter import iris_training_adapter as adapter_module
from mflux.models.iris.weights.iris_lora_mapping import IrisLoRAMapping

pytestmark = pytest.mark.fast

IrisTrainingAdapter = adapter_module.IrisTrainingAdapter


def _tiny_transformer(seed: int = 0) -> IrisTransformer:
    mx.random.seed(seed)
    return IrisTransformer(
        hidden_size=128, depth=3, dual_depth=1, num_heads=2, num_kv_heads=1, patch_size=4, in_channels=3, text_len=16,
        text_dim=128, text_num_layers=3, text_layer_num_heads=2, pixel_depth=2, pixel_hidden_size=16,
        pixel_attn_hidden_size=128, pixel_num_heads=2,
    )  # fmt: off


class _FakeIris(nn.Module):
    # Stands in for Iris: a tiny real transformer, and a generate_image that records its arguments
    def __init__(self, *, quantize=None, model_config=None, model_path=None, bake_lora=False):  # noqa: ARG002
        super().__init__()
        self.transformer = _tiny_transformer()
        self.text_encoder = nn.Linear(2, 2)
        self.prompt_cache = {}
        self.generated = None

    def generate_image(self, **kwargs):
        self.generated = kwargs
        self.prompt_cache["preview"] = (mx.zeros((1,)), mx.zeros((1,)))
        return SimpleNamespace(image="preview-image")


def _adapter(monkeypatch) -> IrisTrainingAdapter:
    monkeypatch.setattr(adapter_module, "Iris", _FakeIris)
    return IrisTrainingAdapter(model_config=ModelConfig.iris_3b_base(), quantize=None)


def _inputs():
    image = mx.random.normal((1, 3, 16, 16), key=mx.random.key(0))
    text = mx.random.normal((1, 16, 3, 128), key=mx.random.key(1))
    return image, (text, mx.ones((1, 16)))


TARGETS = [
    LoraTargetSpec(module_path="blocks.0.attn.q_proj_x", rank=4),
    LoraTargetSpec(module_path="blocks.{block}.mlp.w2", blocks=BlockRange(start=1, end=3), rank=4),
    LoraTargetSpec(module_path="t_embedder.linear_1", rank=4, alpha=8.0),
    LoraTargetSpec(module_path="pixel_blocks.{block}.mlp.fc1", blocks=BlockRange(start=0, end=2), rank=4),
]


def _with_lora(transformer: IrisTransformer) -> IrisTransformer:
    inject_lora_targets(transformer, TARGETS)
    for _, module in transformer.named_modules():
        if hasattr(module, "lora_B"):
            module.lora_B = mx.random.normal(module.lora_B.shape, key=mx.random.key(7)) * 0.05
    return transformer


def test_the_schedule_is_the_reference_shift_for_the_resolution():
    assert IrisTrainingSchedule.shift(512, 512) == pytest.approx(3.0)
    assert IrisTrainingSchedule.shift(1024, 1024) == pytest.approx(4.0)
    config = SimpleNamespace(num_inference_steps=4, height=512, width=512)
    # raw 1, 0.75, 0.5, 0.25 shifted by 3: 3s / (1 + 2s)
    assert IrisTrainingSchedule(config).sigmas.tolist() == pytest.approx([1.0, 0.9, 0.75, 0.5, 0.0])


def test_predict_noise_feeds_one_thousand_times_sigma_and_returns_the_velocity(monkeypatch):
    adapter = _adapter(monkeypatch)
    seen = {}

    def transformer(x, t, y, mask):
        seen.update(t=t, y=y, mask=mask)
        return (x * 2).astype(mx.bfloat16)

    adapter._iris.transformer = transformer
    image, cond = _inputs()
    sigmas = mx.array([1.0, 0.75, 0.0])
    out = adapter.predict_noise(t=1, latents_t=image, sigmas=sigmas, cond=cond, config=None)
    assert seen["t"].tolist() == [750.0] and seen["y"] is cond[0] and seen["mask"] is cond[1]
    assert out.dtype == image.dtype and mx.allclose(out, image * 2, rtol=1e-2, atol=1e-2).item()


def test_the_config_uses_the_training_schedule_and_the_preview_its_default_guidance(monkeypatch):
    adapter = _adapter(monkeypatch)
    spec = SimpleNamespace(steps=30, guidance=0.0, lora_layers=LoraLayersSpec(targets=[]))
    config = adapter.create_config(spec, width=512, height=512)
    assert isinstance(config.scheduler, IrisTrainingSchedule) and config.scheduler.sigmas.shape == (31,)
    assert adapter.generate_preview_image(seed=1, prompt="a fox", width=64, height=64, steps=4) == "preview-image"
    assert adapter._iris.generated["guidance"] is None
    # The cached encodings live on the module, where training would treat them as parameters
    assert adapter._iris.prompt_cache == {}


def test_turbo_is_refused(monkeypatch):
    monkeypatch.setattr(adapter_module, "Iris", _FakeIris)
    with pytest.raises(ValueError, match="iris-3b-base"):
        IrisTrainingAdapter(model_config=ModelConfig.iris_3b(), quantize=None)


def test_encode_data_gives_the_image_in_minus_one_to_one_and_the_caption_encoding(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch)
    encoded = (mx.zeros((1, 300, 12, 8)), mx.ones((1, 300)))
    monkeypatch.setattr(adapter_module.IrisPromptEncoder, "encode_prompt", staticmethod(lambda **kw: encoded))
    adapter._iris.tokenizers = {"qwen3_vl": None}
    path = tmp_path / "white.png"
    Image.new("RGB", (40, 24), "white").save(path)
    clean, cond = adapter.encode_data(data_id=0, image_path=path, prompt="a fox", width=32, height=16)
    assert clean.shape == (1, 3, 16, 32) and clean.dtype == mx.float32
    assert mx.allclose(clean, mx.ones_like(clean)).item()
    assert cond is encoded or cond == encoded


def test_a_saved_lora_has_reference_keys_and_loads_back_at_its_strength(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch)
    _with_lora(adapter.transformer())
    image, (text, mask) = _inputs()
    expected = adapter.transformer()(image, mx.array([500.0]), text, mask)
    path = tmp_path / "adapter.safetensors"
    adapter.save_lora_adapter(path=path, training_spec=SimpleNamespace(lora_layers=LoraLayersSpec(targets=TARGETS), model="iris-3b-base"))  # fmt: off

    weights = mx.load(str(path))
    # The reference names under diffusion_model., as ai-toolkit saves: the timestep MLP is mlp.0 there
    assert "diffusion_model.t_embedder.mlp.0.lora_A.weight" in weights
    assert "diffusion_model.t_embedder.mlp.0.alpha" in weights
    assert weights["diffusion_model.blocks.2.mlp.w2.lora_B.weight"].shape[1] == 4
    assert not any(".alpha" in key and "t_embedder" not in key for key in weights)

    fresh = _tiny_transformer()
    LoRALoader.load_and_apply_lora(IrisLoRAMapping.get_mapping(), fresh, [str(path)], [1.0], bake_lora=False)
    assert mx.allclose(fresh(image, mx.array([500.0]), text, mask), expected, atol=1e-4).item()


def test_gradient_checkpointing_keeps_the_loss_and_lora_gradients():
    image, (text, mask) = _inputs()
    transformer = _with_lora(_tiny_transformer())
    transformer.freeze()
    transformer.unfreeze(keys=["lora_A", "lora_B"], strict=False)

    def loss_fn(model):
        return mx.mean(model(image, mx.array([500.0]), text, mask).astype(mx.float32) ** 2)

    results = []
    for checkpointing in (False, True):
        transformer.gradient_checkpointing = checkpointing
        loss, grads = nn.value_and_grad(transformer, loss_fn)(transformer)
        if checkpointing:
            TrainingTrainer._evaluate_backward_in_block_order(loss, grads)
        results.append((loss, dict(tree_flatten(grads))))
    (plain_loss, plain), (ckpt_loss, ckpt) = results
    assert len(plain) == 12 and plain.keys() == ckpt.keys()
    assert mx.allclose(plain_loss, ckpt_loss).item()
    for key in plain:
        assert mx.allclose(plain[key], ckpt[key], rtol=1e-4, atol=1e-6).item(), key


def test_reference_names_cover_the_renamed_modules():
    assert IrisLoRAMapping.reference_name("t_embedder.linear_2") == "t_embedder.mlp.2"
    assert IrisLoRAMapping.reference_name("y_embedder.layer_blocks.1.mlp.fc1") == "y_embedder.layer_blocks.1.mlp.0"
    assert IrisLoRAMapping.reference_name("blocks.3.q_proj") == "blocks.3.q_proj"
