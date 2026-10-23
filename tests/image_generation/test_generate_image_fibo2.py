import json

import pytest

from mflux.models.common.config import ModelConfig
from tests.image_generation.helpers.image_generation_fibo2_test_helper import ImageGeneratorFibo2TestHelper

# A structured caption with a text_render entry, so the generation also draws and encodes the text image
BICYCLE_CAPTION = json.dumps(
    {
        "short_description": 'A red city bicycle with a wicker basket leans against a white brick wall on a quiet street on a sunny morning. A small wooden sign above it reads "OPEN".',
        "objects": [
            {
                "description": "A red city bicycle with a step-through frame, black tyres and a wicker basket on the handlebars.",
                "location": "center",
                "relationship": "Leaning against the wall, below the sign.",
                "relative_size": "large within frame",
                "shape_and_color": "Classic upright bicycle frame, glossy cherry red.",
                "texture": "glossy painted metal and woven wicker",
                "orientation": "side-on, facing left",
            },
        ],
        "background_setting": "A white-painted brick wall with a green wooden door on the right and a cobblestone pavement below.",
        "lighting": {
            "conditions": "bright morning sun",
            "direction": "side-lit from the right",
            "shadows": "crisp shadow of the bicycle cast on the wall to the left",
        },
        "aesthetics": {
            "composition": "centered, eye-level, with the bicycle filling the lower two thirds",
            "color_scheme": "red and green accents on white",
            "mood_atmosphere": "calm, cheerful",
        },
        "photographic_characteristics": {
            "depth_of_field": "deep",
            "focus": "sharp focus on the bicycle",
            "camera_angle": "eye-level",
            "lens_focal_length": "standard lens (e.g., 35mm-50mm)",
        },
        "style_medium": "photograph",
        "text_render": [
            {
                "text": "OPEN",
                "location": "top center, on a sign above the bicycle",
                "size": "medium",
                "color": "white",
                "font": "hand-painted serif",
                "appearance_details": "painted on a small dark wooden board",
            },
        ],
        "context": "A lifestyle street photograph for a café's social media post.",
        "artistic_style": "realistic",
    }
)


class TestImageGeneratorFibo2:
    @pytest.mark.slow
    def test_fibo2_dev_bicycle(self):
        ImageGeneratorFibo2TestHelper.assert_matches_reference_image(
            reference_image_path="reference_fibo2_dev_bicycle.png",
            output_image_path="output_fibo2_dev_bicycle.png",
            model_config=ModelConfig.fibo2_dev(),
            prompt=BICYCLE_CAPTION,
            steps=20,
            seed=0,
            height=512,
            width=512,
            quantize=8,
        )

    @pytest.mark.slow
    def test_fibo2_edit_turbo_remove_tulip(self):
        ImageGeneratorFibo2TestHelper.assert_matches_reference_image(
            reference_image_path="reference_fibo2_edit_turbo_dog.png",
            output_image_path="output_fibo2_edit_turbo_dog.png",
            model_config=ModelConfig.fibo2_edit_turbo(),
            prompt="Remove the tulip from the dog's mouth.",
            steps=4,
            seed=0,
            height=816,
            width=512,
            quantize=8,
            image_paths=["unsplash_dog.jpg"],
        )
