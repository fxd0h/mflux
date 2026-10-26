import pytest
from huggingface_hub import hf_hub_download

from mflux.models.common.config.model_config import ModelConfig
from tests.image_generation.helpers.image_generation_moss_image_test_helper import ImageGeneratorMOSSImageTestHelper

BICYCLE_PROMPT = (
    "A vintage red bicycle leaning against a weathered cobblestone wall covered in green ivy, "
    "narrow European alley, warm golden afternoon sunlight, shallow depth of field, photorealistic."
)


class TestImageGeneratorMOSSImage:
    @pytest.mark.slow
    def test_image_generation_moss_image_sft(self):
        ImageGeneratorMOSSImageTestHelper.assert_matches_reference_image(
            reference_image_path="reference_moss_image_sft.png",
            output_image_path="output_moss_image_sft.png",
            prompt=BICYCLE_PROMPT,
            steps=20,
            seed=7,
            height=368,
            width=640,
            quantize=8,
        )

    @pytest.mark.slow
    def test_image_generation_moss_image_base(self):
        ImageGeneratorMOSSImageTestHelper.assert_matches_reference_image(
            reference_image_path="reference_moss_image_base.png",
            output_image_path="output_moss_image_base.png",
            prompt=BICYCLE_PROMPT,
            steps=20,
            seed=7,
            height=368,
            width=640,
            quantize=8,
            model_config=ModelConfig.moss_image_base(),
        )

    @pytest.mark.slow
    def test_image_generation_moss_image_sft_8_step_lora(self):
        # The step-distilled LoRA as a runtime layer, the MOSS default. A merge into the weights keeps 44% of
        # its update. The file is fetched by revision: the community 4-step LoRA has the same file name.
        lora = hf_hub_download(
            repo_id="lodestones/MOSS-Image-SFT-8-steps-dmd",
            filename="lora.safetensors",
            revision="f95df83b5e90ae91ea412434b3f9b7f1c8c6314f",
        )
        ImageGeneratorMOSSImageTestHelper.assert_matches_reference_image(
            reference_image_path="reference_moss_image_sft_8_step_lora.png",
            output_image_path="output_moss_image_sft_8_step_lora.png",
            prompt=BICYCLE_PROMPT,
            steps=8,
            guidance=1.0,
            seed=7,
            height=368,
            width=640,
            quantize=8,
            lora_paths=[lora],
        )
