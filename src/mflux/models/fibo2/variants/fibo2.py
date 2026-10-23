import hashlib
import logging
import shutil
from pathlib import Path

import mlx.core as mx
from mlx import nn
from PIL import Image

from mflux.models.common.config.config import Config
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.step_cache.step_cache import StepCache
from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions
from mflux.models.common.vae.vae_util import VAEUtil
from mflux.models.common.weights.saving.model_saver import ModelSaver
from mflux.models.fibo2.fibo2_initializer import Fibo2Initializer
from mflux.models.fibo2.latent_creator.fibo2_latent_creator import Fibo2LatentCreator
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_caption import NULL_CAPTION, TEXT_TO_IMAGE_NEGATIVE
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_caption_check import Fibo2CaptionCheck
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_prompt_encoder import Fibo2PromptEmbeds, Fibo2PromptEncoder
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_encoder import Fibo2TextEncoder
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2TextConditioning, Fibo2Transformer
from mflux.models.fibo2.variants.fibo2_edit_images import Fibo2EditImages
from mflux.models.fibo2.weights.fibo2_weight_definition import Fibo2WeightDefinition
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.utils.exceptions import StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.image_util import ImageUtil

logger = logging.getLogger(__name__)


class Fibo2(nn.Module):
    vae: Flux2VAE
    transformer: Fibo2Transformer
    text_encoder: Fibo2TextEncoder

    DEFAULT_SIZE = 1024
    # The model cards' settings: Dev runs 50 steps with CFG 5, Edit Turbo 4 steps without CFG
    DEV_STEPS, DEV_GUIDANCE = 50, 5.0
    TURBO_STEPS, TURBO_GUIDANCE = 4, 1.0
    # A generation shows Qwen3-VL the images of its caption's text once the noise level is at or below this, as
    # training did; an edit shows them at every step
    TEXT_IMAGES_MAX_NOISE_LEVEL = 0.9

    def __init__(
        self,
        quantize: int | None = None,
        model_path: str | None = None,
        model_config: ModelConfig | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ):
        super().__init__()
        Fibo2Initializer.init(
            model=self,
            model_config=model_config or ModelConfig.fibo2_dev(),
            quantize=quantize,
            model_path=model_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str,
        num_inference_steps: int | None = None,
        height: int | None = None,
        width: int | None = None,
        guidance: float | None = None,
        negative_prompt: str | None = None,
        image_paths: list[str | Path] | None = None,
        mask_path: str | Path | None = None,
        step_cache_ratio: float | None = None,
    ) -> GeneratedImage:
        # Without image_paths, a text-to-image generation. With them, an edit of 1 to 5 images, and a mask confines
        # the edit of a single image to the region it marks. Steps and guidance default to the model card's.
        distilled = not self.model_config.supports_guidance
        if num_inference_steps is None:
            num_inference_steps = Fibo2.TURBO_STEPS if distilled else Fibo2.DEV_STEPS
        if guidance is None:
            guidance = Fibo2.TURBO_GUIDANCE if distilled else Fibo2.DEV_GUIDANCE
        for message in Fibo2CaptionCheck.check(prompt, edit=bool(image_paths)):
            logger.warning(message)
        images = Fibo2EditImages.load(list(image_paths), mask_path) if image_paths else []
        if images and width is None and height is None:
            width, height = Fibo2EditImages.trained_size(images[0].width / images[0].height)
        width, height = width or Fibo2.DEFAULT_SIZE, height or Fibo2.DEFAULT_SIZE
        Fibo2LatentCreator.check_size(height=height, width=width)
        config = Config(
            model_config=self.model_config,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            guidance=guidance,
            scheduler="linear",
        )
        crops = Fibo2EditImages.crops(images, width=width, height=height) if images else []

        # 1. The prompt, with the images to edit, and the negative prompt, without them, when guidance is on
        positive = self._encode_prompt(prompt, crops)
        negative = self._encode_negative(negative_prompt, edit=bool(crops)) if config.guidance > 1.0 else None
        context = [Fibo2LatentCreator.encode_image(self.vae, crop, self.tiling_config) for crop in crops] or None
        text = self.transformer.encode_text(positive.embeds)
        base_text = self.transformer.encode_text(positive.base) if positive.base is not None else None
        negative_text = self.transformer.encode_text(negative) if negative is not None else None
        latents = Fibo2LatentCreator.create_noise(seed=seed, height=height, width=width)
        step_cache = StepCache.for_run(config, ratio=step_cache_ratio)

        # 2. Denoise
        ctx = self.callbacks.start(seed=seed, prompt=prompt, config=config)
        ctx.before_loop(latents)
        for t in config.time_steps:
            try:
                noise = step_cache.reuse(t)
                if noise is None:
                    sigma = config.scheduler.sigmas[t]
                    conditioning = text if base_text is None or Fibo2._shows_text_images(sigma) else base_text
                    noise = self._predict(latents, sigma, conditioning, negative_text, context, config.guidance)
                    step_cache.store(noise)
                latents = config.scheduler.step(noise=noise, timestep=t, latents=latents)
                ctx.in_loop(t, latents)
                mx.eval(latents)
            except KeyboardInterrupt:  # noqa: PERF203
                ctx.interruption(t, latents)
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{config.num_inference_steps}")  # fmt: off
        del text, base_text, negative_text, context
        ctx.after_loop(latents)

        # 3. Decode
        decoded = VAEUtil.decode(
            vae=self.vae,
            latent=self.vae.unpack_packed_latents(latents).astype(ModelConfig.precision),
            tiling_config=self.tiling_config,
        )
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            image_paths=[str(path) for path in image_paths] if image_paths else None,
            masked_image_path=mask_path,
            negative_prompt=negative_prompt if negative is not None else None,
            generation_time=config.time_steps.format_dict["elapsed"],
            generation_parameters={
                **StepCache.generation_parameters(step_cache_ratio),
                **VAEDecodeOptions.generation_parameters(self),
            },
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(model=self, bits=self.bits, base_path=base_path, weight_definition=Fibo2WeightDefinition)
        destination = Path(base_path)
        # The image processor's config and the fonts the images of a caption's text are drawn with
        self.processor.save_pretrained(str(destination / "processor"))
        if self.fonts_dir is not None and self.fonts_dir.resolve() != (destination / "fonts").resolve():
            (destination / "fonts").mkdir(parents=True, exist_ok=True)
            for font in sorted(self.fonts_dir.iterdir()):
                shutil.copy2(font, destination / "fonts" / font.name)

    def _predict(
        self,
        latents: mx.array,
        sigma: mx.array,
        text: Fibo2TextConditioning,
        negative: Fibo2TextConditioning | None,
        context: list[mx.array] | None,
        guidance: float,
    ) -> mx.array:
        # The negative prompt runs as a second pass, as in the reference: the two prompts have different lengths and
        # gist counts. The images to edit condition both passes.
        sigma = sigma.reshape(1)
        cond = self.transformer(latents, sigma, text, context).astype(mx.float32)
        if negative is None:
            return cond
        uncond = self.transformer(latents, sigma, negative, context).astype(mx.float32)
        return uncond + guidance * (cond - uncond)

    def _encode_prompt(self, prompt: str, crops: list[Image.Image]) -> Fibo2PromptEmbeds:
        # An edit's embeddings depend on its images too, so they are cached under the images' hashes as well
        key = (prompt, *(hashlib.sha256(crop.tobytes()).hexdigest() for crop in crops)) if crops else prompt
        if key not in self.prompt_cache:
            if crops:
                images = [Fibo2EditImages.text_encoder_copy(crop) for crop in crops]
                embeds = Fibo2PromptEncoder.encode_edit(prompt, images, self.processor, self.text_encoder, self.fonts_dir)  # fmt: off
            else:
                embeds = Fibo2PromptEncoder.encode_generation(prompt, self.processor, self.text_encoder, self.fonts_dir)  # fmt: off
            self.prompt_cache[key] = embeds
        return self.prompt_cache[key]

    def _encode_negative(self, negative_prompt: str | None, edit: bool) -> mx.array:
        # The negative prompt is encoded as given, without images, like the reference. It defaults to the model card's:
        # for an edit the null caption Fibo 2.0 was trained with, for text to image a caption of what to steer away from
        default = NULL_CAPTION if edit else TEXT_TO_IMAGE_NEGATIVE
        negative = negative_prompt if negative_prompt and negative_prompt.strip() else default
        key = ("negative", negative)
        if key not in self.prompt_cache:
            self.prompt_cache[key] = Fibo2PromptEncoder.encode(negative, [], self.processor, self.text_encoder)
        return self.prompt_cache[key]

    @staticmethod
    def _shows_text_images(sigma: mx.array) -> bool:
        # The reference compares the scheduler timestep, sigma * 1000 in float32, divided by 1000
        return float((sigma * 1000).astype(mx.float32)) / 1000 <= Fibo2.TEXT_IMAGES_MAX_NOISE_LEVEL
