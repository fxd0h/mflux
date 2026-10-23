import warnings
from argparse import Namespace
from pathlib import Path

from mflux.callbacks.callback_manager import CallbackManager
from mflux.cli.parser.parsers import CommandLineParser, lora_init_kwargs_from_args
from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.resolution.config_resolution import ConfigResolution
from mflux.models.common.vae.vae_decode_options import VAEDecodeOptions
from mflux.models.fibo2.latent_creator.fibo2_latent_creator import Fibo2LatentCreator
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_caption_check import Fibo2CaptionCheck
from mflux.models.fibo2.variants.fibo2 import Fibo2
from mflux.models.fibo2.variants.fibo2_edit_images import Fibo2EditImages
from mflux.utils.exceptions import PromptFileReadError, StopImageGenerationException
from mflux.utils.generated_image import GeneratedImage
from mflux.utils.prompt_util import PromptUtil

# The model this CLI runs when --model is omitted: the checkpoint distilled for editing. Fibo 2.0 Dev edits too.
DEFAULT_MODEL = "fibo-2.0-edit-turbo"
FAMILY_MODELS = ("fibo-2.0-dev",)
DEV_GUIDANCE, TURBO_GUIDANCE = Fibo2.DEV_GUIDANCE, Fibo2.TURBO_GUIDANCE

IGNORED_OPTIONS = {
    "--scheduler": "Fibo 2.0 samples on the size-shifted Euler schedule it was trained with.",
    "--lora-style": "Named LoRA styles are only supported by the Flux in-context CLI; use --lora.",
}
CONDITIONAL_OPTIONS = {
    **VAEDecodeOptions.conditional_options(),
    "--guidance": {
        "condition": "fibo-2.0-dev",
        "reason": "Fibo 2.0 Edit Turbo is distilled to run without guidance; any value other than 1.0 exits with an error.",  # fmt: off
    },
    "--negative-prompt": {
        "condition": "fibo-2.0-dev with guidance greater than 1",
        "reason": "Guidance 1 runs only the positive branch.",
    },
}


def build_parser() -> CommandLineParser:
    parser = CommandLineParser(description="Edit one to five images with Fibo 2.0 (Edit Turbo by default, or Dev).")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False, default_model=DEFAULT_MODEL)
    parser.add_lora_arguments()
    parser.add_image_paths_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.set_defaults(width=None, height=None)
    for flag in ("--width", "--height"):
        parser._option_string_actions[flag].help = (
            "Output size, a multiple of 16. If omitted, the size Fibo 2.0 was trained at for the aspect ratio of the "
            "first image, about one megapixel."
        )
    # dest mask_path: a --config-from-conf sidecar restores the mask with it
    parser.add_argument("--mask-image", dest="mask_path", type=Path, default=None, help="Edit only this region of the single image: white marks the region to change. It must have the size of the image.")  # fmt: off
    parser.add_step_cache_arguments()
    parser.add_output_arguments()
    parser.add_vae_full_precision_arguments()
    return parser


class Fibo2EditCommand:
    latent_creator = Fibo2LatentCreator

    @staticmethod
    def validate(args: Namespace) -> ModelConfig:
        # Weight-free: resolves --model and checks the flags before the 25 GB checkpoint loads
        model_config = ConfigResolution.resolve_restricted(
            args.model, DEFAULT_MODEL, model_path=args.model_path, extra_keys=FAMILY_MODELS, base_model=args.base_model
        )
        if not 1 <= len(args.image_paths or []) <= Fibo2EditImages.MAX_IMAGES:
            raise ValueError(f"--image-paths takes 1 to {Fibo2EditImages.MAX_IMAGES} images.")
        if args.mask_path is not None and len(args.image_paths) != 1:
            raise ValueError("--mask-image needs exactly one image in --image-paths.")
        if (args.width is None) != (args.height is None):
            raise ValueError("Give both --width and --height, or neither.")
        if args.width is not None:
            Fibo2LatentCreator.check_size(height=args.height, width=args.width)
        if not model_config.supports_guidance and args.guidance not in (None, 1.0):
            raise ValueError(f"--guidance {args.guidance}: {CONDITIONAL_OPTIONS['--guidance']['reason']}")
        return model_config

    @staticmethod
    def load(args: Namespace) -> Fibo2:
        model = Fibo2(
            model_config=Fibo2EditCommand.validate(args),
            quantize=args.quantize,
            model_path=args.model_path,
            **lora_init_kwargs_from_args(args),
        )
        VAEDecodeOptions.apply(args, model)
        return model

    @staticmethod
    def generate(model: Fibo2, args: Namespace, seed: int, prompt: str) -> GeneratedImage:
        return model.generate_image(
            seed=seed,
            prompt=prompt,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance=Fibo2EditCommand.guidance(model.model_config, args),
            negative_prompt=args.negative_prompt,
            image_paths=args.image_paths,
            mask_path=args.mask_path,
            step_cache_ratio=args.step_cache_ratio,
        )

    @staticmethod
    def guidance(model_config: ModelConfig, args: Namespace) -> float:
        if args.guidance is not None:
            return args.guidance
        return DEV_GUIDANCE if model_config.supports_guidance else TURBO_GUIDANCE


def main():
    parser = build_parser()
    args = parser.parse_args()
    CommandLineParser.warn_ignored_options(IGNORED_OPTIONS)
    VAEDecodeOptions.warn_encode_only_tiling()
    try:
        model_config = Fibo2EditCommand.validate(args)
        Fibo2CaptionCheck.check_args(args, edit=True)
    except ValueError as exc:
        parser.error(str(exc))
    if CommandLineParser._option_was_provided("--negative-prompt") and Fibo2EditCommand.guidance(model_config, args) <= 1.0:  # fmt: off
        warnings.warn(f"--negative-prompt has no effect: {CONDITIONAL_OPTIONS['--negative-prompt']['reason']}", stacklevel=1)  # fmt: off

    model = Fibo2EditCommand.load(args)
    memory_saver = CallbackManager.register_callbacks(args=args, model=model, latent_creator=Fibo2EditCommand.latent_creator)  # fmt: off
    try:
        for seed in args.seed:
            image = Fibo2EditCommand.generate(model, args, seed, PromptUtil.read_prompt(args))
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except (StopImageGenerationException, PromptFileReadError) as exc:
        print(exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
