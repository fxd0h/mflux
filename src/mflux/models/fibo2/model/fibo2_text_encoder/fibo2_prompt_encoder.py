from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from PIL import Image

from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_caption import VISION_MARKER, Fibo2Caption
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_encoder import Fibo2TextEncoder
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_images import Fibo2TextImages
from mflux.models.qwen21.model.qwen21_text_encoder.processor import QwenImage21Processor


@dataclass
class Fibo2PromptEmbeds:
    # The text encoder features of one prompt, (1, 6, length, 7680). A generation whose caption asks for text also
    # has `base`: the same caption without the images of its text, which the first steps read.
    embeds: mx.array
    base: mx.array | None = None


class Fibo2PromptEncoder:
    # The chat turn each prompt is wrapped in for Qwen3-VL
    TEMPLATE = "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n"
    MAX_LENGTH = 4096
    # The Qwen3-VL layers behind the transformer's six text inputs: the Perceiver's, then one per injection block
    OUT_LAYERS = ((9, 20, 31), (5, 16, 27), (7, 18, 29), (9, 20, 31), (11, 22, 33), (13, 24, 35))

    @staticmethod
    def encode_generation(
        prompt: str,
        processor: QwenImage21Processor,
        text_encoder: Fibo2TextEncoder,
        fonts_dir: Path | None,
    ) -> Fibo2PromptEmbeds:
        prompt = Fibo2Caption.generation_prompt(prompt)
        text_images = [Fibo2TextImages.draw(text, fonts_dir) for text in Fibo2Caption.texts(prompt)]
        if not text_images:
            return Fibo2PromptEmbeds(Fibo2PromptEncoder.encode(prompt, [], processor, text_encoder))
        marked = Fibo2Caption.dumps(Fibo2Caption.mark_text_images(Fibo2Caption.parse(prompt)[0], VISION_MARKER))
        unmarked = Fibo2Caption.dumps(Fibo2Caption.mark_text_images(Fibo2Caption.parse(prompt)[0], ""))
        return Fibo2PromptEmbeds(
            embeds=Fibo2PromptEncoder.encode(marked, text_images, processor, text_encoder),
            base=Fibo2PromptEncoder.encode(unmarked, [], processor, text_encoder),
        )

    @staticmethod
    def encode_edit(
        prompt: str,
        images: list[Image.Image],
        processor: QwenImage21Processor,
        text_encoder: Fibo2TextEncoder,
        fonts_dir: Path | None,
    ) -> Fibo2PromptEmbeds:
        # The images to edit come first, then the images of the caption's text, which an edit shows at every step
        text_images = [Fibo2TextImages.draw(text, fonts_dir) for text in Fibo2Caption.texts(prompt)]
        edit_prompt = Fibo2Caption.edit_prompt(prompt, len(images), text_images=bool(text_images))
        return Fibo2PromptEmbeds(Fibo2PromptEncoder.encode(edit_prompt, images + text_images, processor, text_encoder))

    @staticmethod
    def encode(
        prompt: str,
        images: list[Image.Image],
        processor: QwenImage21Processor,
        text_encoder: Fibo2TextEncoder,
    ) -> mx.array:
        inputs = processor(
            text=[Fibo2PromptEncoder.TEMPLATE.format(prompt)],
            images=images or None,
            padding="longest",
            max_length=Fibo2PromptEncoder.MAX_LENGTH,
            truncation=True,
            return_tensors="np",
        )
        layers = sorted({layer for group in Fibo2PromptEncoder.OUT_LAYERS for layer in group})
        states = text_encoder(
            mx.array(inputs["input_ids"]),
            layers=tuple(layers),
            pixel_values=mx.array(inputs["pixel_values"]) if images else None,
            image_grid_thw=mx.array(inputs["image_grid_thw"]) if images else None,
        )
        # Each layer scaled to unit RMS per token in float32, then each bundle concatenates its three layers
        normalized = {layer: Fibo2PromptEncoder._unit_rms(states[layer]) for layer in layers}
        bundles = [mx.concatenate([normalized[layer] for layer in group], axis=-1) for group in Fibo2PromptEncoder.OUT_LAYERS]  # fmt: off
        embeds = mx.stack(bundles, axis=1)
        mx.eval(embeds)
        return embeds

    @staticmethod
    def _unit_rms(states: mx.array) -> mx.array:
        values = states.astype(mx.float32)
        return (values * mx.rsqrt(mx.mean(values * values, axis=-1, keepdims=True) + 1e-6)).astype(states.dtype)
