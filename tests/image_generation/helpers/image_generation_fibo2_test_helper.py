import os
from pathlib import Path

from mflux.models.common.config import ModelConfig
from mflux.models.fibo2.variants.fibo2 import Fibo2
from mflux.utils.image_compare import ImageCompare


class ImageGeneratorFibo2TestHelper:
    @staticmethod
    def assert_matches_reference_image(
        reference_image_path: str,
        output_image_path: str,
        model_config: ModelConfig,
        prompt: str,
        steps: int,
        seed: int,
        height: int,
        width: int,
        quantize: int | None = None,
        image_paths: list[str] | None = None,
        mismatch_threshold: float | None = None,
    ):
        reference_image_path = ImageGeneratorFibo2TestHelper.resolve_path(reference_image_path)
        output_image_path = ImageGeneratorFibo2TestHelper.resolve_path(output_image_path)
        if image_paths:
            image_paths = [ImageGeneratorFibo2TestHelper.resolve_path(path) for path in image_paths]

        try:
            model = Fibo2(quantize=quantize, model_config=model_config)

            result = model.generate_image(
                seed=seed,
                prompt=prompt,
                num_inference_steps=steps,
                height=height,
                width=width,
                image_paths=image_paths,
            )

            result.image.save(output_image_path)

            ImageCompare.check_images_close_enough(
                output_image_path,
                reference_image_path,
                "Generated image doesn't match reference image.",
                mismatch_threshold=mismatch_threshold,
            )
        finally:
            if os.path.exists(output_image_path) and "MFLUX_PRESERVE_TEST_OUTPUT" not in os.environ:
                os.remove(output_image_path)

    @staticmethod
    def resolve_path(path) -> Path | None:
        if path is None:
            return None
        return Path(__file__).parent.parent.parent / "resources" / path
