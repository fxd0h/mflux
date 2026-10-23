import functools
import gc
import warnings
from argparse import Namespace

import mlx.core as mx

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions
from mflux.models.fibo2.latent_creator.fibo2_latent_creator import Fibo2LatentCreator
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_caption import Fibo2Caption
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_caption_check import Fibo2CaptionCheck
from mflux.models.fibo2.variants.fibo2 import Fibo2
from mflux.models.fibo_vlm.model.fibo_vlm import FiboVLM
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs. Its model card calls Fibo 2.0 Edit Turbo an editing model and sends text to image to Dev.
DEFAULT_MODEL = "fibo-2.0-dev"
DEFAULT_GUIDANCE = Fibo2.DEV_GUIDANCE

IGNORED_OPTIONS = {
    **VAEDecodeOptions.ignored_options(),
    "--scheduler": "Fibo 2.0 samples on the size-shifted Euler schedule it was trained with.",
    "--lora-style": "Named LoRA styles are only supported by the Flux in-context CLI; use --lora.",
}
CONDITIONAL_OPTIONS = {
    "--negative-prompt": {
        "condition": "guidance greater than 1",
        "reason": "Guidance 1 runs only the positive branch.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Generate an image with Fibo 2.0 Dev from a structured JSON caption or plain text.")  # fmt: off
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_argument("--expand-prompt", action="store_true", help="Write a structured JSON caption from a plain-text prompt with FIBO-vlm (briaai/FIBO-vlm, about 9 GB on first use), and generate from that caption. A JSON caption is used as it is.")  # fmt: off
    parser.add_step_cache_arguments()
    parser.add_output_arguments()
    parser.add_vae_full_precision_arguments()
    parser.set_defaults(guidance=DEFAULT_GUIDANCE)
    return parser


class Fibo2Command:
    latent_creator = Fibo2LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: resolves --model against the in-memory registry only, and checks the size before the
        # 25 GB checkpoint loads
        model_config = ConfigResolution.resolve_restricted(
            args.model, DEFAULT_MODEL, model_path=args.model_path, base_model=args.base_model
        )
        Fibo2LatentCreator.check_size(height=args.height, width=args.width)
        return model_config

    @staticmethod
    def load(args: Namespace) -> Fibo2:
        model = Fibo2(
            model_config=Fibo2Command.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            **lora_init_kwargs_from_args(args),
        )
        VAEDecodeOptions.apply(args, model)
        return model

    @staticmethod
    def generate(model: Fibo2, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        if getattr(args, "expand_prompt", False):
            prompt = Fibo2Command.expand_prompt(prompt, args.quantize)
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance=DEFAULT_GUIDANCE if args.guidance is None else args.guidance,
            negative_prompt=args.negative_prompt,
            step_cache_ratio=args.step_cache_ratio,
        )

    @staticmethod
    @functools.lru_cache(maxsize=8)
    def expand_prompt(prompt: str, quantize: int | None) -> str:
        # FIBO-vlm writes the caption once per prompt, with the seed FIBO's own command uses, and is freed before
        # Fibo 2.0 runs. Its captions keep FIBO 1's scores, which Fibo 2.0 drops when it reads them.
        if isinstance(Fibo2Caption.parse(prompt)[0], dict):
            return prompt
        vlm = FiboVLM(quantize=quantize)
        caption = vlm.generate(prompt=prompt, seed=42)
        del vlm
        gc.collect()
        mx.clear_cache()
        print(f"FIBO-vlm caption: {caption}")
        return caption


def main():
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    if CommandLineParser._option_was_provided("--negative-prompt") and args.guidance is not None and args.guidance <= 1.0:  # fmt: off
        warnings.warn(f"--negative-prompt has no effect: {CONDITIONAL_OPTIONS['--negative-prompt']['reason']}", stacklevel=1)  # fmt: off
    try:
        Fibo2Command.validate(args)
        Fibo2CaptionCheck.check_args(args, edit=False)
    except ValueError as exc:
        parser.error(str(exc))

    model = Fibo2Command.load(args)
    memory_saver = CallbackManager.register_callbacks(
        args=args, model=model, latent_creator=Fibo2Command.latent_creator
    )
    try:
        for seed in args.seed:
            image = Fibo2Command.generate(model, args, seed, PromptUtil.read_prompt(args))
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
