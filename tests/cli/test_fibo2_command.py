import json
import sys

import pytest
from PIL import Image

from mflux.models.common.config.model_config import AVAILABLE_MODELS
from mflux.models.fibo2.cli import (
    fibo2_edit_generate as edit_cli,
    fibo2_generate as cli,
)
from mflux.models.fibo2.variants import Fibo2
from tests.cli.helpers.command_fakes import FakeModel, reset_cli_globals

CAPTION = {
    "short_description": "A red bicycle against a white wall.",
    "background_setting": "a white brick wall",
    "lighting": {"conditions": "morning sun", "direction": "from the right"},
    "aesthetics": {"composition": "centered", "color_scheme": "red on white", "mood_atmosphere": "calm"},
    "context": "a street photograph",
}


class FakeFibo2(FakeModel):
    real = Fibo2


class FakeFiboVLM:
    calls: list[str] = []

    def __init__(self, quantize=None):
        self.quantize = quantize

    def generate(self, prompt: str, seed: int | None = None) -> str:
        FakeFiboVLM.calls.append(prompt)
        return json.dumps({**CAPTION, "short_description": prompt})


@pytest.fixture(autouse=True)
def isolated_cli(monkeypatch):
    FakeFibo2.instances.clear()
    FakeFiboVLM.calls.clear()
    monkeypatch.setattr(cli, "Fibo2", FakeFibo2)
    monkeypatch.setattr(edit_cli, "Fibo2", FakeFibo2)
    monkeypatch.setattr(cli, "FiboVLM", FakeFiboVLM)
    cli.Fibo2Command.expand_prompt.cache_clear()
    reset_cli_globals(monkeypatch)
    yield
    FakeFibo2.instances.clear()


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "source.png"
    Image.new("RGB", (96, 64), "white").save(path)
    return path


def run(monkeypatch, module, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-fibo2", *argv])
    module.main()
    assert len(FakeFibo2.instances) == 1
    return FakeFibo2.instances[0]


def args_for(monkeypatch, module, argv):
    monkeypatch.setattr(sys, "argv", ["mflux-generate-fibo2", *argv])
    return module.build_parser().parse_args()


@pytest.mark.fast
class TestGenerateCommand:
    def test_the_defaults_are_the_model_card_settings(self, monkeypatch, tmp_path):
        model = run(monkeypatch, cli, ["--prompt", "a red bicycle", "--seed", "3", "--output", str(tmp_path / "out.png")])  # fmt: off
        assert model.init_kwargs == {
            "model_config": AVAILABLE_MODELS["fibo-2.0-dev"],
            "quantize": None,
            "model_path": None,
            "lora_paths": None,
            "lora_scales": None,
            "bake_lora": True,
        }
        assert model.generate_calls == [
            {
                "seed": 3,
                "prompt": "a red bicycle",
                "num_inference_steps": 50,
                "height": 1024,
                "width": 1024,
                "guidance": 5.0,
                "negative_prompt": "",
                "step_cache_ratio": None,
            }
        ]
        assert model.images[0].saves == [(str(tmp_path / "out.png"), False)]

    def test_every_flag_reaches_the_model(self, monkeypatch, tmp_path):
        argv = ["--prompt", "x", "--seed", "1", "2", "--steps", "28", "--width", "1424", "--height", "800", "--guidance", "3.5", "--negative-prompt", "blurry", "--step-cache-ratio", "0.25", "-q", "8", "--output", str(tmp_path / "out.png"), "--make-conf"]  # fmt: off
        model = run(monkeypatch, cli, argv)
        assert model.init_kwargs["quantize"] == 8
        assert [call["seed"] for call in model.generate_calls] == [1, 2]
        call = model.generate_calls[0]
        assert (call["num_inference_steps"], call["width"], call["height"], call["guidance"]) == (28, 1424, 800, 3.5)
        assert (call["negative_prompt"], call["step_cache_ratio"]) == ("blurry", 0.25)
        assert [image.saves for image in model.images] == [
            [(str(tmp_path / "out_seed_1.png"), True)],
            [(str(tmp_path / "out_seed_2.png"), True)],
        ]

    def test_edit_turbo_is_left_to_the_edit_command(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["mflux-generate-fibo2", "--prompt", "x", "--model", "fibo-2.0-edit-turbo"])
        with pytest.raises(SystemExit):
            cli.main()
        assert FakeFibo2.instances == []

    def test_a_broken_json_caption_stops_before_the_model_loads(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["mflux-generate-fibo2", "--prompt", '{"short_description": "x",}'])
        with pytest.raises(SystemExit):
            cli.main()
        assert "not valid JSON" in capsys.readouterr().err
        assert FakeFibo2.instances == []

    def test_a_caption_file_with_a_wrong_type_stops_before_the_model_loads(self, monkeypatch, tmp_path, capsys):
        prompt_file = tmp_path / "caption.json"
        prompt_file.write_text(json.dumps({**CAPTION, "lighting": "soft"}))
        monkeypatch.setattr(sys, "argv", ["mflux-generate-fibo2", "--prompt-file", str(prompt_file)])
        with pytest.raises(SystemExit):
            cli.main()
        assert "lighting has to be a JSON object" in capsys.readouterr().err
        assert FakeFibo2.instances == []

    def test_a_size_that_is_not_a_multiple_of_16_stops_before_the_model_loads(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["mflux-generate-fibo2", "--prompt", "x", "--width", "1000"])
        with pytest.raises(SystemExit):
            cli.main()
        assert FakeFibo2.instances == []

    def test_expand_prompt_writes_the_caption_once_for_all_seeds(self, monkeypatch, tmp_path):
        argv = ["--prompt", "a red bicycle", "--expand-prompt", "--seed", "1", "2", "--output", str(tmp_path / "o.png")]
        model = run(monkeypatch, cli, argv)
        assert FakeFiboVLM.calls == ["a red bicycle"]
        assert [json.loads(call["prompt"])["short_description"] for call in model.generate_calls] == ["a red bicycle"] * 2  # fmt: off

    def test_expand_prompt_keeps_a_json_caption_as_it_is(self, monkeypatch, tmp_path):
        model = run(monkeypatch, cli, ["--prompt", json.dumps(CAPTION), "--expand-prompt", "--output", str(tmp_path / "o.png")])  # fmt: off
        assert FakeFiboVLM.calls == []
        assert model.generate_calls[0]["prompt"] == json.dumps(CAPTION)

    def test_the_scheduler_flag_is_ignored_with_a_warning(self, monkeypatch, tmp_path):
        with pytest.warns(UserWarning, match="--scheduler is ignored"):
            run(monkeypatch, cli, ["--prompt", "x", "--scheduler", "flow_match_euler_discrete", "--output", str(tmp_path / "o.png")])  # fmt: off


@pytest.mark.fast
class TestEditCommand:
    def test_edit_turbo_is_the_default_with_4_steps_and_no_guidance(self, monkeypatch, tmp_path, source):
        model = run(monkeypatch, edit_cli, ["--prompt", "Make it blue.", "--image-paths", str(source), "--seed", "3", "--output", str(tmp_path / "o.png")])  # fmt: off
        assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["fibo-2.0-edit-turbo"]
        assert model.generate_calls == [
            {
                "seed": 3,
                "prompt": "Make it blue.",
                "num_inference_steps": 4,
                "height": None,
                "width": None,
                "guidance": 1.0,
                "negative_prompt": "",
                "image_paths": [source],
                "mask_path": None,
                "step_cache_ratio": None,
            }
        ]

    def test_dev_edits_with_50_steps_and_guidance_5(self, monkeypatch, tmp_path, source):
        model = run(monkeypatch, edit_cli, ["--model", "fibo-2.0-dev", "--prompt", "x", "--image-paths", str(source), "--output", str(tmp_path / "o.png")])  # fmt: off
        call = model.generate_calls[0]
        assert model.init_kwargs["model_config"] is AVAILABLE_MODELS["fibo-2.0-dev"]
        assert (call["num_inference_steps"], call["guidance"]) == (50, 5.0)

    @pytest.mark.parametrize(
        ("checkpoint", "key"),
        [("/models/fibo-2-0-dev-mflux-q8", "fibo-2.0-dev"), ("/models/fibo-2-0-edit-turbo-mflux-q4", "fibo-2.0-edit-turbo"), ("/models/Fibo-2.0-dev", "fibo-2.0-dev"), ("/models/my-save", "fibo-2.0-edit-turbo")],
    )  # fmt: skip
    def test_a_saved_model_is_recognized_by_its_name(self, monkeypatch, source, checkpoint, key):
        args = args_for(monkeypatch, edit_cli, ["--model", checkpoint, "--prompt", "x", "--image-paths", str(source)])
        assert edit_cli.Fibo2EditCommand.validate(args) is AVAILABLE_MODELS[key]

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            (["--guidance", "3.0"], "distilled to run without guidance"),
            (["--width", "1024"], "both --width and --height"),
            (["--width", "1000", "--height", "1024"], "multiples of 16"),
            (["--prompt", json.dumps(CAPTION)], "edit_instruction"),
        ],
    )
    def test_a_bad_flag_or_caption_stops_before_the_model_loads(self, monkeypatch, capsys, source, argv, message):
        prompt = [] if "--prompt" in argv else ["--prompt", "Make it blue."]
        monkeypatch.setattr(sys, "argv", ["mflux-generate-fibo2-edit", *prompt, "--image-paths", str(source), *argv])
        with pytest.raises(SystemExit):
            edit_cli.main()
        assert message in capsys.readouterr().err
        assert FakeFibo2.instances == []

    def test_a_mask_needs_exactly_one_image(self, monkeypatch, capsys, source):
        argv = ["--prompt", "x", "--image-paths", str(source), str(source), "--mask-image", str(source)]
        monkeypatch.setattr(sys, "argv", ["mflux-generate-fibo2-edit", *argv])
        with pytest.raises(SystemExit):
            edit_cli.main()
        assert "exactly one image" in capsys.readouterr().err

    def test_a_sidecar_replays_the_images_and_the_mask(self, monkeypatch, tmp_path, source):
        sidecar = tmp_path / "run.metadata.json"
        sidecar.write_text(json.dumps({"prompt": "Make it blue.", "seed": 5, "image_paths": [str(source)], "masked_image_path": str(source)}))  # fmt: off
        model = run(monkeypatch, edit_cli, ["--config-from-conf", str(sidecar), "--output", str(tmp_path / "o.png")])
        call = model.generate_calls[0]
        assert (call["seed"], call["image_paths"], str(call["mask_path"])) == (5, [source], str(source))
