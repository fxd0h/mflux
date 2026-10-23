import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
from PIL import Image

from mflux.models.fibo2.latent_creator.fibo2_latent_creator import Fibo2LatentCreator
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_caption import (
    NULL_CAPTION,
    TEXT_TO_IMAGE_NEGATIVE,
    VISION_MARKER,
    Fibo2Caption,
)
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_caption_check import Fibo2CaptionCheck
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_prompt_encoder import Fibo2PromptEncoder
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_images import Fibo2TextImages
from mflux.models.fibo2.variants.fibo2 import Fibo2
from mflux.models.fibo2.variants.fibo2_edit_images import Fibo2EditImages
from mflux.models.fibo2.weights.fibo2_weight_definition import Fibo2WeightDefinition

pytestmark = pytest.mark.fast

# The expected strings are what diffusers' BriaFibo2Pipeline gives for the same prompts (diffusers main at 48cdbae)
CAPTION = {
    "short_description": "A red bicycle against a white wall.",
    "background_setting": "a white brick wall",
    "lighting": {"conditions": "morning sun", "direction": "from the right"},
    "aesthetics": {"composition": "centered", "color_scheme": "red on white", "mood_atmosphere": "calm"},
    "context": "a street photograph",
}


class TestCaption:
    @pytest.mark.parametrize(
        ("prompt", "caption", "instruction"),
        [
            ("a red bicycle against a white wall", "a red bicycle against a white wall", None),
            # Empty values and FIBO 1's scores go, at every level
            (
                '{"short_description":"x","objects":[],"lighting":{"conditions":"","direction":"left"},"aesthetics":{"aesthetic_score":"high","preference_score":"high","mood_atmosphere":"calm"},"context":null}',
                {"short_description": "x", "lighting": {"direction": "left"}, "aesthetics": {"mood_atmosphere": "calm"}},
                None,
            ),
            ('{"short_description":"x","aesthetics":{"aesthetic_score":"high"}}', {"short_description": "x"}, None),
            # A short description and a text_render list keep the list even when it is empty
            ('{"short_description":"x","text_render":[]}', {"short_description": "x", "text_render": []}, None),
            (
                '{"short_description":"x","objects":[{"description":"d","number_of_objects":NaN}]}',
                {"short_description": "x", "objects": [{"description": "d"}]},
                None,
            ),
            ('{"structured_caption":"a short description","edit_instruction":"Make it red."}', "a short description", "Make it red."),  # fmt: off
            ('{"edit_instruction":"Make it red.","short_description":"y"}', {"short_description": "y"}, "Make it red."),
            ("[1, 2]", "[1, 2]", None),
        ],
    )  # fmt: skip
    def test_parse_cleans_the_caption_as_training_did(self, prompt, caption, instruction):
        assert Fibo2Caption.parse(prompt) == (caption, instruction)

    def test_a_generation_prompt_is_serialized_compactly(self):
        prompt = '{"short_description": "x",\n  "text_render": [{"text": "OPEN", "location": "top"}, {"text": ""}, {"location": "no text"}]}'  # fmt: off
        assert Fibo2Caption.generation_prompt(prompt) == '{"short_description":"x","text_render":[{"text":"OPEN","location":"top"},{"location":"no text"}]}'  # fmt: off
        assert Fibo2Caption.texts(prompt) == ["OPEN"]
        assert Fibo2Caption.generation_prompt("plain text") == "plain text"

    def test_an_instruction_alone_becomes_the_edit_prompt(self):
        expected = '{"image":"<|vision_start|><|image_pad|><|vision_end|>","edit_instruction":"a red bicycle against a white wall"}'  # fmt: off
        assert Fibo2Caption.edit_prompt("a red bicycle against a white wall", 1) == expected

    def test_an_edit_caption_marks_each_image_and_each_text(self):
        prompt = '{"short_description":"A dog next to a sign that reads \\"GOOD BOY\\".","text_render":[{"text":"GOOD BOY","location":"right"},{"text":"","location":"x"}],"edit_instruction":"Add a sign."}'  # fmt: off
        marker = VISION_MARKER
        expected = (
            f'{{"image_1":"{marker}","image_2":"{marker}","structured_caption":{{"short_description":"A dog next to a sign that reads \\"GOOD BOY\\".",'  # fmt: off
            f'"text_render":[{{"text":"GOOD BOY","location":"right","image_tokens":"{marker}"}},{{"location":"x"}}]}},"edit_instruction":"Add a sign."}}'  # fmt: off
        )
        assert Fibo2Caption.edit_prompt(prompt, 2, text_images=True) == expected

    def test_the_null_caption_is_the_reference_negative(self):
        assert json.loads(NULL_CAPTION) == {
            "short_description": "",
            "objects": [],
            "background_setting": "",
            "lighting": {},
            "aesthetics": {},
            "photographic_characteristics": {},
            "style_medium": "",
            "context": "",
            "artistic_style": "",
        }


class TestCaptionCheck:
    def test_a_complete_caption_and_plain_text_pass_without_warnings(self):
        assert Fibo2CaptionCheck.check(json.dumps(CAPTION), edit=False) == []
        assert Fibo2CaptionCheck.check("a red bicycle", edit=False) == []
        assert Fibo2CaptionCheck.check("[cinematic] a red bicycle", edit=False) == []

    def test_a_prompt_that_starts_like_json_but_does_not_parse_is_an_error(self):
        with pytest.raises(ValueError, match=r"not valid JSON: .* at line 1, column 26"):
            Fibo2CaptionCheck.check('{"short_description": "x",}', edit=False)

    def test_every_field_of_the_wrong_type_is_named(self):
        caption = {**CAPTION, "short_description": 5, "objects": {"description": "d"}, "lighting": "soft"}
        with pytest.raises(ValueError) as error:
            Fibo2CaptionCheck.check(json.dumps(caption), edit=False)
        assert str(error.value).splitlines()[1:] == [
            "  short_description has to be a string, not the number 5.",
            "  lighting has to be a JSON object, not the string 'soft'.",
            "  objects has to be a list of objects, not a JSON object.",
        ]

    def test_a_list_entry_of_the_wrong_type_has_its_index(self):
        caption = {
            **CAPTION,
            "text_render": [{"text": "OPEN", "location": "top", "size": "m", "color": "w", "font": 3}],
        }
        with pytest.raises(ValueError, match=r"text_render\[0\]\.font has to be a string, not the number 3"):
            Fibo2CaptionCheck.check(json.dumps(caption), edit=False)

    @pytest.mark.parametrize(("value", "valid"), [(3, True), ("3", True), (3.0, True), (True, True), (3.5, False), ("three", False)])  # fmt: off
    def test_a_count_takes_what_thales_vgl_takes_as_an_integer(self, value, valid):
        # pydantic's lax mode in ThalesVGL (bria-vgl 0.10.0) accepts the first four and rejects the last two
        caption = {**CAPTION, "objects": [{"description": "d", "location": "l", "relationship": "r", "number_of_objects": value}]}  # fmt: off
        if valid:
            assert Fibo2CaptionCheck.check(json.dumps(caption), edit=False) == []
        else:
            with pytest.raises(ValueError, match="number_of_objects has to be an integer"):
                Fibo2CaptionCheck.check(json.dumps(caption), edit=False)

    def test_unknown_and_missing_fields_are_warnings(self):
        caption = {key: value for key, value in CAPTION.items() if key != "context"} | {"short_desciption": "typo"}
        warnings = Fibo2CaptionCheck.check(json.dumps(caption), edit=False)
        assert warnings == [
            "Missing: context. Fibo 2.0's training captions have these fields.",
            "Unknown field 'short_desciption': the model was not trained on it. Did you mean 'short_description'?",
        ]

    def test_fibo1_scores_and_empty_values_are_not_reported(self):
        caption = {**CAPTION, "aesthetics": {**CAPTION["aesthetics"], "aesthetic_score": "very high"}, "objects": []}
        assert Fibo2CaptionCheck.check(json.dumps(caption), edit=False) == []

    def test_an_edit_caption_needs_its_instruction(self):
        with pytest.raises(ValueError, match="edit_instruction"):
            Fibo2CaptionCheck.check(json.dumps(CAPTION), edit=True)
        assert Fibo2CaptionCheck.check(json.dumps({**CAPTION, "edit_instruction": "Make it blue."}), edit=True) == []
        assert Fibo2CaptionCheck.check("Make it blue.", edit=True) == []

    def test_a_structured_caption_edit_holds_only_the_caption_and_the_instruction(self):
        prompt = json.dumps({"structured_caption": CAPTION, "edit_instruction": "Make it blue.", "extra": 1})
        with pytest.raises(ValueError, match="nothing else but"):
            Fibo2CaptionCheck.check(prompt, edit=True)

    def test_a_generation_says_that_it_ignores_an_instruction(self):
        warnings = Fibo2CaptionCheck.check(json.dumps({**CAPTION, "edit_instruction": "Make it blue."}), edit=False)
        assert warnings == ["edit_instruction is read only when editing images; this generation ignores it."]


class TestEditImages:
    @pytest.mark.parametrize(
        ("aspect_ratio", "size"),
        [(1.0, (1024, 1024)), (1.5, (1248, 832)), (16 / 9, (1424, 800)), (9 / 16, (800, 1424)), (4 / 3, (1152, 864)), (0.624, (848, 1360)), (2.5, (1600, 640)), (0.2, (464, 2320))],
    )  # fmt: skip
    def test_the_trained_size_matches_the_reference(self, aspect_ratio, size):
        assert Fibo2EditImages.trained_size(aspect_ratio) == size

    def test_one_image_is_cropped_to_the_result_and_several_to_their_own_sizes(self):
        wide, tall = Image.new("RGB", (900, 600), "red"), Image.new("RGB", (300, 600), "blue")
        assert [crop.size for crop in Fibo2EditImages.crops([wide], width=512, height=512)] == [(512, 512)]
        assert [crop.size for crop in Fibo2EditImages.crops([wide, tall], width=512, height=512)] == [(1248, 832), (720, 1440)]  # fmt: off

    def test_the_text_encoder_sees_at_most_768_pixels_a_side(self):
        assert Fibo2EditImages.text_encoder_copy(Image.new("RGB", (1248, 832))).size == (768, 512)
        assert Fibo2EditImages.text_encoder_copy(Image.new("RGB", (512, 352))).size == (512, 352)

    def test_the_mask_greys_out_the_region_through_a_binary_threshold(self):
        image = Image.new("RGB", (4, 1), (10, 20, 30))
        mask = Image.fromarray(np.array([[0, 127, 128, 255]], dtype=np.uint8), mode="L")
        greyed = np.asarray(Fibo2EditImages.grey_out(image, mask))[0].tolist()
        assert greyed == [[10, 20, 30], [10, 20, 30], [128, 128, 128], [128, 128, 128]]

    def test_loading_rejects_too_many_images_and_a_mask_of_another_size(self, tmp_path):
        paths = []
        for k in range(6):
            Image.new("RGB", (64, 64)).save(tmp_path / f"{k}.png")
            paths.append(tmp_path / f"{k}.png")
        Image.new("L", (32, 32)).save(tmp_path / "mask.png")
        with pytest.raises(ValueError, match="1 to 5 images"):
            Fibo2EditImages.load(paths, None)
        with pytest.raises(ValueError, match="exactly one image"):
            Fibo2EditImages.load(paths[:2], tmp_path / "mask.png")
        with pytest.raises(ValueError, match="The mask is 32x32 but the image is 64x64"):
            Fibo2EditImages.load(paths[:1], tmp_path / "mask.png")


class TestTextImages:
    def test_a_text_is_drawn_black_on_white_in_one_line(self):
        image = Fibo2TextImages.draw("OPEN  \n now", fonts_dir=None)
        pixels = np.asarray(image)
        assert image.mode == "RGB" and pixels.max() == 255 and pixels.min() == 0
        assert image.height < 40 < image.width

    def test_a_long_text_is_split_into_lines_of_256_characters(self):
        one_line = Fibo2TextImages.draw("a" * 256, fonts_dir=None)
        two_lines = Fibo2TextImages.draw("a" * 257, fonts_dir=None)
        assert two_lines.height > one_line.height


class TestPipelineConstants:
    def test_the_text_encoder_layers_are_the_reference_bundles(self):
        assert Fibo2PromptEncoder.OUT_LAYERS == ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))  # fmt: off
        assert Fibo2PromptEncoder.TEMPLATE == "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"

    def test_the_last_qwen3_vl_layer_is_not_loaded(self):
        assert Fibo2WeightDefinition.text_encoder_key("model.language_model.layers.35.mlp.up_proj.weight") is None
        assert Fibo2WeightDefinition.text_encoder_key("model.language_model.layers.34.mlp.up_proj.weight") is not None
        assert max(layer for group in Fibo2PromptEncoder.OUT_LAYERS for layer in group) == 35

    @pytest.mark.parametrize(("sigma", "shows"), [(1.0, False), (0.9001, False), (0.9, True), (0.5, True)])
    def test_a_generation_shows_the_text_images_from_noise_level_0_9(self, sigma, shows):
        assert Fibo2._shows_text_images(mx.array(sigma, dtype=mx.float32)) is shows

    def test_the_negative_prompt_defaults_to_the_model_cards(self, monkeypatch):
        # Text to image steers away from the model card's caption of soft, featureless images; an edit uses the null
        # caption. A negative prompt that is given replaces either.
        encoded = []
        monkeypatch.setattr(Fibo2PromptEncoder, "encode", lambda prompt, *_: encoded.append(prompt) or prompt)
        model = SimpleNamespace(prompt_cache={}, processor=None, text_encoder=None)
        for negative_prompt, edit in [(None, False), ("", True), (" ", False), ("blurry", True)]:
            Fibo2._encode_negative(model, negative_prompt, edit=edit)
        assert encoded == [TEXT_TO_IMAGE_NEGATIVE, NULL_CAPTION, "blurry"]
        assert json.loads(TEXT_TO_IMAGE_NEGATIVE)["objects"][0]["description"] == "the main subject"


class TestLatentCreator:
    def test_the_noise_is_float32_in_the_packed_layout(self):
        noise = Fibo2LatentCreator.create_noise(seed=3, height=256, width=512)
        assert noise.shape == (1, 128, 16, 32) and noise.dtype == mx.float32

    @pytest.mark.parametrize(("width", "height"), [(1000, 1024), (1024, 0), (-16, 64)])
    def test_a_size_that_is_not_a_positive_multiple_of_16_is_rejected(self, width, height):
        with pytest.raises(ValueError, match="positive multiples of 16"):
            Fibo2LatentCreator.check_size(height=height, width=width)


def test_the_command_step_defaults_are_the_model_defaults():
    from mflux.cli.defaults.defaults import MODEL_INFERENCE_STEPS

    assert (MODEL_INFERENCE_STEPS["fibo-2.0-dev"], MODEL_INFERENCE_STEPS["fibo-2.0-edit-turbo"]) == (Fibo2.DEV_STEPS, Fibo2.TURBO_STEPS)  # fmt: off
