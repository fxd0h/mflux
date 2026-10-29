from contextlib import nullcontext
from types import SimpleNamespace

import mlx.core as mx
import pytest

from mflux.models.anima.model.anima_scheduler.anima_scheduler import AnimaScheduler
from mflux.models.anima.model.anima_text_encoder.anima_prompt_weights import AnimaPromptWeights
from mflux.models.anima.model.anima_transformer.anima_transformer import AnimaTransformer
from mflux.models.anima.training_adapter import anima_training_adapter as anima_adapter_module
from mflux.models.anima.weights.anima_lora_mapping import AnimaLoRAMapping
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.training.lora.target_injector import inject_lora_targets
from mflux.models.common.training.state.training_spec import BlockRange, LoraLayersSpec, LoraTargetSpec
from mflux.models.common.training.utils import TrainingUtil

AnimaTrainingAdapter = anima_adapter_module.AnimaTrainingAdapter


class _FakeAnima:
    def __init__(self, *, quantize, model_config, model_path=None, bake_lora=True):  # noqa: ARG002
        self.recorded = []
        self.last_generate_kwargs = None
        self.vae = "vae"
        self.tiling_config = None

        def transformer(latents, sigma, context):
            self.recorded.append(SimpleNamespace(latents=latents, sigma=sigma, context=context))
            return (latents * 2).astype(mx.bfloat16)

        self.transformer = transformer

    def generate_image(self, **kwargs):
        self.last_generate_kwargs = kwargs
        return "preview-image"


@pytest.mark.fast
def test_predict_noise_feeds_sigma_and_returns_the_velocity_in_the_latent_dtype(monkeypatch):
    # The trainer's loss is (clean + prediction - noise)^2, so the prediction is the flow velocity noise - clean,
    # which is what Anima's transformer returns for sigma (not an index)
    monkeypatch.setattr(anima_adapter_module, "Anima", _FakeAnima)
    adapter = AnimaTrainingAdapter(model_config=ModelConfig.anima_base(), quantize=None)
    latents_t = mx.ones((1, 16, 8, 8), dtype=mx.float32)

    velocity = adapter.predict_noise(t=1, latents_t=latents_t, sigmas=mx.array([0.9, 0.4, 0.0]), cond=mx.zeros((1, 512, 8)), config=None)  # fmt: off

    call = adapter._anima.recorded[-1]
    assert call.sigma.shape == (1,) and float(call.sigma[0]) == pytest.approx(0.4)
    assert velocity.dtype == mx.float32 and mx.allclose(velocity, latents_t * 2).item()


@pytest.mark.fast
def test_the_schedule_is_anima_and_the_preview_uses_the_training_guidance(monkeypatch):
    monkeypatch.setattr(anima_adapter_module, "Anima", _FakeAnima)
    adapter = AnimaTrainingAdapter(model_config=ModelConfig.anima_base(), quantize=None)
    monkeypatch.setattr(adapter, "_assistant_disabled", lambda: nullcontext())

    config = adapter.create_config(
        SimpleNamespace(steps=30, guidance=4.0, lora_layers=LoraLayersSpec(targets=[])), width=512, height=512
    )
    adapter.generate_preview_image(seed=7, prompt="bluxyz, a guitar", width=512, height=512, steps=30)

    assert mx.array_equal(config.scheduler.sigmas, AnimaScheduler.compute_sigmas(30)).item()
    assert adapter._anima.last_generate_kwargs["guidance"] == 4.0


@pytest.mark.fast
def test_a_saved_lora_has_comfyui_keys_and_loads_back_through_the_anima_mapping(monkeypatch, tmp_path):
    # ComfyUI and ai-toolkit name Anima LoRA layers diffusion_model.blocks.N...; mflux reads the same keys
    transformer = AnimaTransformer(dim=128, num_blocks=2, num_heads=2, context_dim=64, adaln_lora_dim=32)
    targets = [
        LoraTargetSpec(module_path="blocks.{block}.self_attn.q_proj", blocks=BlockRange(start=0, end=2), rank=4),
        LoraTargetSpec(module_path="blocks.{block}.adaln_modulation_mlp.1", blocks=BlockRange(start=0, end=2), rank=4),
    ]
    inject_lora_targets(transformer, targets)
    adapter = object.__new__(AnimaTrainingAdapter)
    adapter._anima = SimpleNamespace(transformer=transformer)
    path = tmp_path / "anima_lora.safetensors"

    adapter.save_lora_adapter(path=path, training_spec=SimpleNamespace(lora_layers=LoraLayersSpec(targets=targets), model="anima-base"))  # fmt: off

    keys = set(mx.load(str(path)))
    assert "diffusion_model.blocks.1.self_attn.q_proj.lora_A.weight" in keys
    assert "diffusion_model.blocks.0.adaln_modulation_mlp.1.lora_B.weight" in keys
    fresh = AnimaTransformer(dim=128, num_blocks=2, num_heads=2, context_dim=64, adaln_lora_dim=32)
    paths, _ = LoRALoader.load_and_apply_lora(AnimaLoRAMapping.get_mapping(), fresh, [str(path)], [1.0], bake_lora=False)  # fmt: off
    assert paths == [str(path)]


@pytest.mark.fast
def test_gradient_checkpointing_keeps_the_output():
    transformer = AnimaTransformer(dim=128, num_blocks=2, num_heads=2, context_dim=64, adaln_lora_dim=32)
    latents = mx.random.normal((1, 16, 8, 8), key=mx.random.key(0))
    context = mx.random.normal((1, 8, 64), key=mx.random.key(1))
    plain = transformer(latents, mx.array([0.5]), context)
    transformer.gradient_checkpointing = True
    assert mx.allclose(transformer(latents, mx.array([0.5]), context), plain).item()


@pytest.mark.fast
def test_turbo_is_refused_for_training(monkeypatch):
    monkeypatch.setattr(anima_adapter_module, "Anima", _FakeAnima)
    with pytest.raises(ValueError, match="anima-base"):
        AnimaTrainingAdapter(model_config=ModelConfig.from_name("anima"), quantize=None)


@pytest.mark.fast
def test_an_llm_adapter_target_is_refused(monkeypatch):
    # The adapter output is computed once per caption before training, so its LoRA would never get a gradient
    monkeypatch.setattr(anima_adapter_module, "Anima", _FakeAnima)
    adapter = AnimaTrainingAdapter(model_config=ModelConfig.anima_base(), quantize=None)
    spec = SimpleNamespace(steps=30, guidance=4.0, lora_layers=LoraLayersSpec(targets=[LoraTargetSpec(module_path="llm_adapter.out_proj", rank=4)]))  # fmt: off
    with pytest.raises(ValueError, match="llm_adapter.out_proj"):
        adapter.create_config(spec, width=512, height=512)


@pytest.mark.fast
def test_an_alpha_other_than_the_rank_is_saved_and_loads_at_the_same_strength(tmp_path):
    transformer = AnimaTransformer(dim=128, num_blocks=1, num_heads=2, context_dim=64, adaln_lora_dim=32)
    targets = [LoraTargetSpec(module_path="blocks.{block}.mlp.layer1", blocks=BlockRange(start=0, end=1), rank=8, alpha=4.0)]  # fmt: off
    inject_lora_targets(transformer, targets)
    trained = TrainingUtil.get_train_lora(transformer, "blocks.0.mlp.layer1")
    trained.lora_B = mx.random.normal(trained.lora_B.shape, key=mx.random.key(2)) * 0.1
    x = mx.random.normal((2, 128), key=mx.random.key(3))
    expected = trained(x)
    adapter = object.__new__(AnimaTrainingAdapter)
    adapter._anima = SimpleNamespace(transformer=transformer)
    path = tmp_path / "alpha.safetensors"

    adapter.save_lora_adapter(path=path, training_spec=SimpleNamespace(lora_layers=LoraLayersSpec(targets=targets), model="anima-base"))  # fmt: off

    assert float(mx.load(str(path))["diffusion_model.blocks.0.mlp.layer1.alpha"]) == pytest.approx(4.0)
    fresh = AnimaTransformer(dim=128, num_blocks=1, num_heads=2, context_dim=64, adaln_lora_dim=32)
    fresh.blocks[0].mlp.layer1.weight = trained.linear.weight
    LoRALoader.load_and_apply_lora(AnimaLoRAMapping.get_mapping(), fresh, [str(path)], [1.0], bake_lora=False)
    assert mx.allclose(fresh.blocks[0].mlp.layer1(x), expected, atol=1e-4).item()


@pytest.mark.fast
def test_captions_are_encoded_as_plain_text(monkeypatch):
    # Danbooru tags with a series in parentheses are text, not ComfyUI weights
    from mflux.models.anima.model.anima_text_encoder import anima_prompt_encoder as encoder_module

    seen = []

    class _Tokenizer:
        def tokenize(self, text, keep_empty=False):  # noqa: ARG002
            seen.append(text)
            return SimpleNamespace(input_ids=mx.array([[1, 2]]))

    def text_encoder(ids, dtype):  # noqa: ARG001
        return mx.zeros((1, ids.shape[1], 4)), None

    transformer = SimpleNamespace(encode_text=lambda hidden, t5_ids, t5_weights: t5_weights)
    caption = "1girl, hatsune miku (vocaloid), smile"
    weights = encoder_module.AnimaPromptEncoder.encode(caption, _Tokenizer(), _Tokenizer(), text_encoder, transformer, weighted=False)  # fmt: off
    assert weights is None and seen == [caption, caption]
    # A caption escaped for A1111 or ComfyUI trains on the text the same escaped prompt gives at inference
    seen.clear()
    escaped = "1girl, hatsune miku \\(vocaloid\\), smile"
    encoder_module.AnimaPromptEncoder.encode(escaped, _Tokenizer(), _Tokenizer(), text_encoder, transformer, weighted=False)  # fmt: off
    assert seen == [caption, caption] == [AnimaPromptWeights.parse(escaped)[0][0]] * 2


@pytest.mark.fast
def test_the_preview_leaves_no_cached_prompt_and_defaults_its_guidance(monkeypatch):
    # A cached context on the module would be a trainable parameter; guidance 0.0 from the config means the default
    monkeypatch.setattr(anima_adapter_module, "Anima", _FakeAnima)
    adapter = AnimaTrainingAdapter(model_config=ModelConfig.anima_base(), quantize=None)
    monkeypatch.setattr(adapter, "_assistant_disabled", lambda: nullcontext())
    adapter.create_config(SimpleNamespace(steps=30, guidance=0.0, lora_layers=LoraLayersSpec(targets=[])), width=512, height=512)  # fmt: off
    adapter._anima.prompt_cache = {"bluxyz, a lighthouse": mx.zeros((2,))}
    adapter.generate_preview_image(seed=1, prompt="bluxyz, a lighthouse", width=512, height=512, steps=2)
    assert adapter._anima.last_generate_kwargs["guidance"] is None
    assert adapter._anima.prompt_cache == {}


@pytest.mark.fast
def test_gradient_checkpointing_keeps_the_lora_gradients():
    import mlx.nn as nn
    from mlx.utils import tree_flatten

    from mflux.models.common.training.trainer import TrainingTrainer

    transformer = AnimaTransformer(dim=128, num_blocks=3, num_heads=2, context_dim=64, adaln_lora_dim=32)
    targets = [
        LoraTargetSpec(module_path=f"blocks.{{block}}.{name}", blocks=BlockRange(start=0, end=3), rank=4)
        for name in ("self_attn.q_proj", "self_attn.output_proj", "mlp.layer1", "mlp.layer2")
    ]
    inject_lora_targets(transformer, targets)
    for _, module in transformer.named_modules():
        if hasattr(module, "lora_B"):
            module.lora_B = mx.random.normal(module.lora_B.shape, key=mx.random.key(7)) * 0.01
    transformer.freeze()
    transformer.unfreeze(keys=["lora_A", "lora_B"], strict=False)
    latents = mx.random.normal((1, 16, 8, 8), key=mx.random.key(0))
    context = mx.random.normal((1, 8, 64), key=mx.random.key(1))

    def loss_fn(model):
        return mx.mean(model(latents, mx.array([0.5]), context).astype(mx.float32) ** 2)

    results = []
    for checkpointing in (False, True):
        transformer.gradient_checkpointing = checkpointing
        loss, grads = nn.value_and_grad(transformer, loss_fn)(transformer)
        if checkpointing:
            TrainingTrainer._evaluate_backward_in_block_order(loss, grads)
        results.append((loss, dict(tree_flatten(grads))))
    (plain_loss, plain), (ckpt_loss, ckpt) = results
    assert len(plain) == 24 and plain.keys() == ckpt.keys()
    assert mx.allclose(plain_loss, ckpt_loss).item()
    for key in plain:
        assert mx.allclose(plain[key], ckpt[key], rtol=1e-4, atol=1e-6).item(), key


@pytest.mark.fast
def test_training_images_are_encoded_with_the_low_ram_tiling(monkeypatch, tmp_path):
    # low_ram sets the model's tiling config; the encode of the training images must use it as the decode does
    from mflux.models.common.vae.tiling_config import TilingConfig

    monkeypatch.setattr(anima_adapter_module, "Anima", _FakeAnima)
    adapter = AnimaTrainingAdapter(model_config=ModelConfig.anima_base(), quantize=None)
    adapter._anima.tiling_config = TilingConfig()
    seen = {}

    def encode_image(**kwargs):
        seen.update(kwargs)
        raise StopIteration

    monkeypatch.setattr(anima_adapter_module.LatentCreator, "encode_image", staticmethod(encode_image))
    with pytest.raises(StopIteration):
        adapter.encode_data(data_id=0, image_path=tmp_path / "x.png", prompt="a fox", width=64, height=64)
    assert seen["tiling_config"] is adapter._anima.tiling_config
