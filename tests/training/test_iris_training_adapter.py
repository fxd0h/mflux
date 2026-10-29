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
from mflux.models.common.training.utils import TrainingUtil
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

    def _encode_prompt(self, prompt):
        if prompt not in self.prompt_cache:
            if self.text_encoder is None:
                raise AssertionError(f"no text encoder to encode {prompt!r}")
            self.prompt_cache[prompt] = (mx.full((1, 2), len(prompt)), mx.ones((1, 2)))
        return self.prompt_cache[prompt]

    def generate_image(self, **kwargs):
        self.generated = kwargs
        self._encode_prompt(kwargs["prompt"])
        self._encode_prompt("")
        return SimpleNamespace(image="preview-image")


def _adapter(monkeypatch) -> IrisTrainingAdapter:
    monkeypatch.setattr(adapter_module, "Iris", _FakeIris)
    return IrisTrainingAdapter(model_config=ModelConfig.iris_3b_base(), quantize=None)


def _inputs():
    image = mx.random.normal((1, 3, 16, 16), key=mx.random.key(0))
    text = mx.random.normal((1, 16, 3, 128), key=mx.random.key(1))
    return image, {"embeddings": text, "mask": mx.ones((1, 16))}


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
    # Small images, which the loader accepts, get no shift instead of a zero one that divides 0 by 0
    assert IrisTrainingSchedule.shift(64, 64) == IrisTrainingSchedule.shift(32, 32) == 1.0
    small = IrisTrainingSchedule(SimpleNamespace(num_inference_steps=4, height=64, width=64)).sigmas.tolist()
    assert small == pytest.approx([1.0, 0.75, 0.5, 0.25, 0.0])
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
    assert seen["t"].tolist() == [750.0] and seen["y"] is cond["embeddings"] and seen["mask"] is cond["mask"]
    assert out.dtype == image.dtype and mx.allclose(out, image * 2, rtol=1e-2, atol=1e-2).item()


def test_the_config_uses_the_training_schedule_and_the_preview_its_default_guidance(monkeypatch):
    adapter = _adapter(monkeypatch)
    # The training grid has the reference trainer's 1000 levels whatever the preview step count, so at 4 preview steps
    # training still reaches the low noise levels
    spec = SimpleNamespace(steps=4, guidance=0.0, lora_layers=LoraLayersSpec(targets=[]))
    config = adapter.create_config(spec, width=512, height=512)
    assert isinstance(config.scheduler, IrisTrainingSchedule) and config.scheduler.sigmas.shape == (1001,)
    assert config.num_inference_steps == 1000 and float(config.scheduler.sigmas[-2]) < 0.01
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
    assert cond["embeddings"] is encoded[0] and cond["mask"] is encoded[1]


def test_a_saved_lora_has_reference_keys_and_loads_back_at_its_strength(monkeypatch, tmp_path):
    adapter = _adapter(monkeypatch)
    _with_lora(adapter.transformer())
    image, cond = _inputs()
    text, mask = cond["embeddings"], cond["mask"]
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
    image, cond = _inputs()
    text, mask = cond["embeddings"], cond["mask"]
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


def test_each_item_trains_on_the_shift_of_its_own_size(monkeypatch):
    # The preview is 1024 (shift 4) and the item 512 (shift 3): the item must not reuse the preview's schedule
    import random

    from mflux.models.common.training.dataset.batch import DataItem

    adapter = _adapter(monkeypatch)
    spec = SimpleNamespace(steps=4, guidance=None, training_loop=SimpleNamespace(timestep_low=0, timestep_high=None, timestep_type=None))  # fmt: off
    base_config = adapter.create_config(spec, width=1024, height=1024)
    _ = base_config.scheduler
    seen = {}

    def predict_noise(*, t, latents_t, sigmas, cond, config):  # noqa: ARG001
        seen["sigmas"] = sigmas
        return mx.zeros_like(latents_t)

    adapter.predict_noise = predict_noise
    item = DataItem(data_id=0, prompt="a fox", image_path="x.png", clean_latents=mx.zeros((1, 3, 512, 512)), cond={}, width=512, height=512)  # fmt: off
    TrainingTrainer._single_example_loss(adapter, spec, base_config, item, random.Random(0))
    expected = IrisTrainingSchedule(SimpleNamespace(num_inference_steps=1000, height=512, width=512)).sigmas
    assert mx.array_equal(seen["sigmas"], expected).item()


def test_the_conditioning_goes_through_the_low_ram_data_cache(monkeypatch, tmp_path):
    from mflux.models.common.training.dataset.data_cache import TrainingDataCache

    adapter = _adapter(monkeypatch)
    image, cond = _inputs()
    paths = TrainingDataCache.wipe_and_init(data_root=tmp_path)
    TrainingDataCache.save_item(paths=paths, data_id=0, prompt="a fox", image_path=tmp_path / "x.png", width=16, height=16, clean_latents=image, cond=cond)  # fmt: off
    clean, loaded, _, _ = TrainingDataCache.load_tensors(paths=paths, data_id=0)
    adapter._iris.transformer = _tiny_transformer()
    sigmas = mx.array([1.0, 0.5, 0.0])
    expected = adapter.predict_noise(t=1, latents_t=image, sigmas=sigmas, cond=cond, config=None)
    assert mx.allclose(
        adapter.predict_noise(t=1, latents_t=clean, sigmas=sigmas, cond=loaded, config=None), expected
    ).item()


def test_a_resumed_lora_trains_at_its_saved_scale(monkeypatch, tmp_path):
    # alpha 8 at rank 4 trains at scale 2. Resuming must give the layer that scale and B back as it was, so the
    # restored optimizer state still matches the weights it moves
    adapter = _adapter(monkeypatch)
    _with_lora(adapter.transformer())
    trained = TrainingUtil.get_train_lora(adapter.transformer(), "t_embedder.linear_1")
    path = tmp_path / "adapter.safetensors"
    adapter.save_lora_adapter(path=path, training_spec=SimpleNamespace(lora_layers=LoraLayersSpec(targets=TARGETS), model="iris-3b-base"))  # fmt: off

    resumed = _adapter(monkeypatch)
    resumed.load_lora_adapter(path=path)
    loaded = TrainingUtil.get_train_lora(resumed.transformer(), "t_embedder.linear_1")
    assert trained.scale == loaded.scale == pytest.approx(2.0)
    assert mx.array_equal(loaded.lora_B, trained.lora_B).item() and mx.array_equal(loaded.lora_A, trained.lora_A).item()


def test_releasing_the_text_encoder_keeps_the_previews_working(monkeypatch):
    adapter = _adapter(monkeypatch)
    spec = SimpleNamespace(monitoring=SimpleNamespace(preview_prompts=["a fox", "a bluxyz lighthouse"]))
    assert adapter.release_text_encoder(spec)
    assert adapter._iris.text_encoder is None and adapter._iris.prompt_cache == {}
    adapter.freeze_base()
    assert (
        adapter.generate_preview_image(seed=1, prompt="a bluxyz lighthouse", width=64, height=64, steps=4)
        == "preview-image"
    )
    # The encodings stay on the adapter, never in the model's cache, where training would treat them as parameters
    assert adapter._iris.prompt_cache == {}
    assert sorted(adapter._preview_encodings) == ["", "a bluxyz lighthouse", "a fox"]
