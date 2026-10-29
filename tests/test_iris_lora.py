import json
import re

import mlx.core as mx
import pytest
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.iris.model.iris_scheduler.iris_few_step_scheduler import IrisFewStepScheduler
from mflux.models.iris.model.iris_transformer.iris_transformer import IrisTransformer
from mflux.models.iris.weights.iris_lora_mapping import IrisLoRAMapping

pytestmark = pytest.mark.fast


def _tiny_transformer() -> IrisTransformer:
    # depth 3 with one dual-stream block, so both block kinds are there
    return IrisTransformer(
        hidden_size=128, depth=3, dual_depth=1, num_heads=2, num_kv_heads=1, patch_size=4, in_channels=3, text_len=16,
        text_dim=128, text_num_layers=3, text_layer_num_heads=2, pixel_depth=2, pixel_hidden_size=16,
        pixel_attn_hidden_size=128, pixel_num_heads=2,
    )  # fmt: off


def _reference_name(path: str) -> str:
    # The reference checkpoint's name for an MLX module path
    for mlx_name, reference in IrisLoRAMapping.REFERENCE_NAMES.items():
        pattern = re.escape(mlx_name).replace(re.escape("{block}"), r"(\d+)")
        match = re.fullmatch(pattern, path)
        if match:
            return reference.format(block=match.group(1)) if match.groups() else reference
    return path


def _linear_paths(transformer: nn.Module) -> list[str]:
    return [name for name, module in transformer.named_modules() if isinstance(module, nn.Linear)]


def _forward(transformer: IrisTransformer) -> mx.array:
    x = mx.random.normal((1, 3, 16, 16), key=mx.random.key(0))
    text = mx.random.normal((1, 16, 3, 128), key=mx.random.key(1))
    return transformer(x, mx.array([500.0]), text, mx.ones((1, 16)))


def test_every_linear_layer_has_a_target():
    # The full model's 311 linear layers, under the names its checkpoint uses
    paths = _linear_paths(IrisTransformer())
    assert len(paths) == 311
    keys = [f"{_reference_name(path)}.lora_down" for path in paths]
    reached = {
        target.model_path.format(block=block)
        for key in keys
        for target in IrisLoRAMapping.get_mapping()
        for pattern in target.possible_down_patterns
        if (block := LoRALoader._match_pattern(key, pattern)) is not None
    }
    assert reached == set(paths)


@pytest.mark.parametrize(
    ("down", "up"),
    [("{}.lora_down", "{}.lora_up"), ("diffusion_model.{}.lora_A.weight", "diffusion_model.{}.lora_B.weight")],
    ids=["reference-kohya", "ai-toolkit"],
)
def test_a_lora_gives_every_layer_its_delta(tmp_path, down, up):
    # The reference repo's names with bare kohya matrices (as pentacoxian-dev/iris-3b-turbo-lora saves), and
    # ai-toolkit's PEFT names under diffusion_model., on every linear layer of a tiny model
    transformer = _tiny_transformer()
    paths = _linear_paths(transformer)
    rank = 2
    weights, deltas = {}, {}
    for i, path in enumerate(paths):
        linear = dict(transformer.named_modules())[path]
        out_dim, in_dim = linear.weight.shape
        a = mx.random.normal((rank, in_dim), key=mx.random.key(2 * i)) * 0.01
        b = mx.random.normal((out_dim, rank), key=mx.random.key(2 * i + 1)) * 0.01
        name = _reference_name(path)
        weights[down.format(name)], weights[up.format(name)] = a, b
        deltas[path] = (b @ a, linear.weight)
    file = tmp_path / "iris_lora.safetensors"
    mx.save_safetensors(str(file), weights)

    LoRALoader.load_and_apply_lora(IrisLoRAMapping.get_mapping(), transformer, [str(file)], [1.0], bake_lora=True)

    modules = dict(transformer.named_modules())
    for path, (delta, before) in deltas.items():
        assert mx.allclose(modules[path].weight - before, delta, atol=1e-5).item(), path


def test_a_runtime_lora_matches_the_baked_one(tmp_path):
    rank, weights = 2, {}
    for i, path in enumerate(["blocks.0.attn.q_proj_x", "blocks.2.mlp.w2", "t_embedder.linear_1", "pixel_blocks.1.mlp.fc1"]):  # fmt: off
        linear = dict(_tiny_transformer().named_modules())[path]
        out_dim, in_dim = linear.weight.shape
        weights[f"{_reference_name(path)}.lora_down"] = mx.random.normal((rank, in_dim), key=mx.random.key(i)) * 0.1
        weights[f"{_reference_name(path)}.lora_up"] = mx.random.normal((out_dim, rank), key=mx.random.key(9 + i)) * 0.1
    file = tmp_path / "iris_lora.safetensors"
    mx.save_safetensors(str(file), weights)
    outputs = []
    for bake in (True, False):
        mx.random.seed(0)
        transformer = _tiny_transformer()
        LoRALoader.load_and_apply_lora(IrisLoRAMapping.get_mapping(), transformer, [str(file)], [0.7], bake_lora=bake)
        outputs.append(_forward(transformer))
    mx.random.seed(0)
    plain = _forward(_tiny_transformer())
    # Baking rounds the merged weight (about 1e-3 here); the LoRA itself moves the output by about 0.1
    assert mx.allclose(outputs[0], outputs[1], atol=5e-3).item()
    assert not mx.allclose(outputs[0], plain, atol=5e-2).item()


class TestIrisFewStepScheduler:
    def test_the_times_are_the_reference_turbo_times(self):
        # pentaCoxian/iris-3b-turbo turbo_times(4, 4.0): an even grid from 1, shifted
        assert IrisFewStepScheduler(4, 4.0, seed=0).sigmas == pytest.approx([1.0, 12 / 13, 0.8, 4 / 7])

    def test_each_step_renoises_the_clean_estimate(self):
        noise = mx.random.normal((1, 3, 4, 4), key=mx.random.key(0))
        clean = mx.random.normal((1, 3, 4, 4), key=mx.random.key(1))
        scheduler = IrisFewStepScheduler(4, 4.0, seed=3)
        latents = noise
        for step in range(4):
            sigma = scheduler.sigmas[step]
            latents, estimate = scheduler.step(latents, (latents - clean) / sigma, step)
            assert mx.allclose(estimate, clean, atol=1e-5).item()
        assert mx.allclose(latents, clean, atol=1e-5).item()

    def test_the_noise_follows_the_seed(self):
        latents, velocity = mx.zeros((1, 3, 4, 4)), mx.zeros((1, 3, 4, 4))
        first = IrisFewStepScheduler(4, 4.0, seed=5).step(latents, velocity, 0)[0]
        again = IrisFewStepScheduler(4, 4.0, seed=5).step(latents, velocity, 0)[0]
        other = IrisFewStepScheduler(4, 4.0, seed=6).step(latents, velocity, 0)[0]
        assert mx.array_equal(first, again).item() and not mx.array_equal(first, other).item()

    def test_a_few_step_lora_is_found_from_its_metadata(self, tmp_path):
        turbo = tmp_path / "student.safetensors"
        mx.save_safetensors(str(turbo), {"a": mx.zeros((1,))}, metadata={"rank": "64", "extra": json.dumps({"turbo_steps": 4, "shift": 4.0})})  # fmt: off
        style = tmp_path / "style.safetensors"
        mx.save_safetensors(str(style), {"a": mx.zeros((1,))}, metadata={"format": "pt"})
        assert IrisFewStepScheduler.from_lora_files([str(style)]) is None
        assert IrisFewStepScheduler.from_lora_files([str(style), str(turbo)]) == (4, 4.0)
        assert IrisFewStepScheduler.from_lora_files(None) is None
        # At scale 0 the LoRA changes no weight, so it does not change the sampler
        assert IrisFewStepScheduler.from_lora_files([str(turbo)], [0.0]) is None

    @pytest.mark.parametrize("extra", ["null", "[4]", '"turbo_steps"', "not json"])
    def test_extra_metadata_that_is_not_an_object_records_no_sampler(self, tmp_path, extra):
        path = tmp_path / "other.safetensors"
        mx.save_safetensors(str(path), {"a": mx.zeros((1,))}, metadata={"extra": extra})
        assert IrisFewStepScheduler.from_lora_files([str(path)]) is None

    @pytest.mark.parametrize("extra", [{"turbo_steps": 0}, {"turbo_steps": -2}, {"turbo_steps": 4, "shift": 0}, {"turbo_steps": 10**9}, {"turbo_steps": "four"}])  # fmt: off
    def test_invalid_few_step_metadata_is_refused(self, tmp_path, extra):
        path = tmp_path / "bad.safetensors"
        mx.save_safetensors(str(path), {"a": mx.zeros((1,))}, metadata={"extra": json.dumps(extra)})
        with pytest.raises(ValueError, match="needs 1 to 100 steps"):
            IrisFewStepScheduler.from_lora_files([str(path)])


def test_the_initializer_records_the_loras_on_the_model(tmp_path, capsys):
    # On the real nn.Module: MLX stores list attributes in the module's dict, which a class-level default breaks
    from types import SimpleNamespace

    from mflux.models.iris.iris_initializer import IrisInitializer
    from mflux.models.iris.variants.txt2img.iris import Iris

    turbo = tmp_path / "student.safetensors"
    mx.save_safetensors(str(turbo), {"t_embedder.mlp.0.lora_down": mx.zeros((2, 256)), "t_embedder.mlp.0.lora_up": mx.zeros((128, 2))}, metadata={"extra": json.dumps({"turbo_steps": 4, "shift": 4.0})})  # fmt: off
    model = Iris.__new__(Iris)
    nn.Module.__init__(model)
    model.transformer = _tiny_transformer()
    model.model_config = ModelConfig.iris_3b_base()
    IrisInitializer._apply_loras(model, SimpleNamespace(dense_component=lambda name: None), [str(turbo)], [1.0], True)  # noqa: ARG005
    assert (model.lora_paths, model.lora_scales, model.few_step) == ([str(turbo)], [1.0], (4, 4.0))
    assert "the base model runs 4 steps without CFG" in capsys.readouterr().out

    plain = Iris.__new__(Iris)
    nn.Module.__init__(plain)
    plain.transformer = _tiny_transformer()
    plain.model_config = ModelConfig.iris_3b_base()
    IrisInitializer._apply_loras(plain, SimpleNamespace(dense_component=lambda name: None), None, None, True)  # noqa: ARG005
    assert (plain.lora_paths, plain.few_step) == ([], None)
    assert "Few-step LoRA" not in capsys.readouterr().out


def test_turbo_with_a_few_step_lora_keeps_its_own_sampler_and_can_be_saved(tmp_path, capsys, monkeypatch):
    from types import SimpleNamespace

    from mflux.models.common.weights.saving.model_saver import ModelSaver
    from mflux.models.iris.iris_initializer import IrisInitializer
    from mflux.models.iris.variants.txt2img.iris import Iris

    turbo = tmp_path / "student.safetensors"
    mx.save_safetensors(str(turbo), {"t_embedder.mlp.0.lora_down": mx.zeros((2, 256)), "t_embedder.mlp.0.lora_up": mx.zeros((128, 2))}, metadata={"extra": json.dumps({"turbo_steps": 4, "shift": 4.0})})  # fmt: off
    model = Iris.__new__(Iris)
    nn.Module.__init__(model)
    model.transformer = _tiny_transformer()
    model.model_config = ModelConfig.iris_3b()
    IrisInitializer._apply_loras(model, SimpleNamespace(dense_component=lambda name: None), [str(turbo)], [1.0], True)  # noqa: ARG005
    assert model.few_step is None and "Few-step LoRA" not in capsys.readouterr().out
    model.bits = None
    saved = []
    monkeypatch.setattr(ModelSaver, "save_model", staticmethod(lambda **kwargs: saved.append(kwargs)))
    model.save_model(str(tmp_path / "saved"))
    assert len(saved) == 1


def test_a_model_with_a_few_step_lora_cannot_be_saved(tmp_path):
    # The saved weights would keep the LoRA, but nothing would record its sampler
    from mflux.models.iris.variants.txt2img.iris import Iris

    model = Iris.__new__(Iris)
    nn.Module.__init__(model)
    model.few_step = (4, 4.0)
    with pytest.raises(ValueError, match="cannot record"):
        model.save_model(str(tmp_path))


def test_a_runtime_lora_on_every_linear_layer_runs(tmp_path):
    # Without baking every target stays a LoRALinear, which has no .weight of its own (pixel_embedder.proj read it)
    transformer = _tiny_transformer()
    weights = {}
    for i, path in enumerate(_linear_paths(transformer)):
        out_dim, in_dim = dict(transformer.named_modules())[path].weight.shape
        weights[f"{_reference_name(path)}.lora_down"] = mx.random.normal((2, in_dim), key=mx.random.key(2 * i)) * 0.01
        weights[f"{_reference_name(path)}.lora_up"] = mx.random.normal((out_dim, 2), key=mx.random.key(2 * i + 1)) * 0.01  # fmt: off
    file = tmp_path / "iris_lora.safetensors"
    mx.save_safetensors(str(file), weights)
    LoRALoader.load_and_apply_lora(IrisLoRAMapping.get_mapping(), transformer, [str(file)], [1.0], bake_lora=False)
    assert mx.isfinite(_forward(transformer)).all().item()


def test_the_rank_and_alpha_of_the_file_metadata_set_the_scale(tmp_path):
    # The few-step fork records one rank and alpha for the file and scales the update by alpha / rank
    path, rank = "blocks.2.mlp.w2", 2
    linear = dict(_tiny_transformer().named_modules())[path]
    out_dim, in_dim = linear.weight.shape
    down = mx.random.normal((rank, in_dim), key=mx.random.key(0)) * 0.01
    up = mx.random.normal((out_dim, rank), key=mx.random.key(1)) * 0.01
    deltas = []
    for metadata in ({"rank": "2", "alpha": "2.0"}, {"rank": "2", "alpha": "4.0"}):
        file = tmp_path / f"fork_{metadata['alpha']}.safetensors"
        mx.save_safetensors(str(file), {f"{_reference_name(path)}.lora_down": down, f"{_reference_name(path)}.lora_up": up}, metadata=metadata)  # fmt: off
        transformer = _tiny_transformer()
        before = dict(transformer.named_modules())[path].weight
        LoRALoader.load_and_apply_lora(IrisLoRAMapping.get_mapping(), transformer, [str(file)], [1.0], bake_lora=True)
        deltas.append(dict(transformer.named_modules())[path].weight - before)
    assert mx.allclose(deltas[0], up @ down, atol=1e-5).item()
    assert mx.allclose(deltas[1], 2 * up @ down, atol=1e-5).item()


@pytest.mark.parametrize(
    ("weights", "metadata"),
    [({"a": mx.zeros((1,))}, {"format": "iris-turbo-svd-v1"}), ({"dense.final_layer.linear.bias": mx.zeros((1,))}, {})],
    ids=["format", "dense-keys"],
)
def test_a_turbo_svd_file_is_refused(tmp_path, weights, metadata):
    # Aikimi/iris-3b-turbo-svd-lora also replaces dense tensors, which a LoRA load would drop
    file = tmp_path / "turbo_svd.safetensors"
    mx.save_safetensors(str(file), weights, metadata=metadata)
    with pytest.raises(ValueError, match="iris-turbo-svd-v1"):
        IrisFewStepScheduler.refuse_turbo_svd([str(file)])
    plain = tmp_path / "plain.safetensors"
    mx.save_safetensors(str(plain), {"a": mx.zeros((1,))})
    IrisFewStepScheduler.refuse_turbo_svd([str(plain)])


def test_the_save_check_needs_only_the_lora_header(tmp_path):
    from mflux.models.iris.variants.txt2img.iris import Iris

    turbo = tmp_path / "student.safetensors"
    mx.save_safetensors(str(turbo), {"a": mx.zeros((1,))}, metadata={"extra": json.dumps({"turbo_steps": 4})})
    with pytest.raises(ValueError, match="cannot record"):
        Iris.check_save(ModelConfig.iris_3b_base(), [str(turbo)], [1.0])
    # Turbo keeps its own sampler, and a LoRA at scale 0 changes nothing
    Iris.check_save(ModelConfig.iris_3b(), [str(turbo)], [1.0])
    Iris.check_save(ModelConfig.iris_3b_base(), [str(turbo)], [0.0])
