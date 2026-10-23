from pathlib import Path

import mlx.core as mx

from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.models.common.config import ModelConfig
from mflux.models.common.lora.mapping.lora_loader import LoRALoader
from mflux.models.common.resolution.path_resolution import PathResolution
from mflux.models.common.tokenizer import TokenizerLoader
from mflux.models.common.vae.block_decoder import VAEBlockDecoder
from mflux.models.common.weights.loading.weight_applier import WeightApplier
from mflux.models.common.weights.loading.weight_loader import WeightLoader
from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_encoder import Fibo2TextEncoder
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2Transformer
from mflux.models.fibo2.weights.fibo2_lora_mapping import Fibo2LoRAMapping
from mflux.models.fibo2.weights.fibo2_weight_definition import Fibo2WeightDefinition
from mflux.models.flux2.model.flux2_vae.vae import Flux2VAE
from mflux.models.qwen21.model.qwen21_text_encoder.processor import QwenImage21Processor


class Fibo2Initializer:
    @staticmethod
    def init(
        model,
        model_config: ModelConfig,
        quantize: int | None,
        model_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        bake_lora: bool = True,
    ) -> None:
        root = PathResolution.resolve(
            model_path or model_config.model_name,
            Fibo2WeightDefinition.get_download_patterns(),
            saved_subdirs=[c.hf_subdir for c in Fibo2WeightDefinition.get_components()],
        )
        Fibo2Initializer._init_config(model, model_config, root)
        weights = WeightLoader.load(weight_definition=Fibo2WeightDefinition, model_path=str(root))
        model.tokenizers = TokenizerLoader.load_all(Fibo2WeightDefinition.get_tokenizers(), str(root))
        model.processor = QwenImage21Processor(root / "processor", model.tokenizers["fibo2"].tokenizer)
        model.vae = Flux2VAE()
        model.vae.decode_mode = VAEBlockDecoder.LOW_MEMORY
        model.transformer = Fibo2Transformer()
        model.text_encoder = Fibo2TextEncoder()
        model.bits = WeightApplier.apply_and_quantize(
            weights=weights,
            quantize_arg=quantize,
            weight_definition=Fibo2WeightDefinition,
            models={"vae": model.vae, "transformer": model.transformer, "text_encoder": model.text_encoder},
        )
        model.lora_paths, model.lora_scales = LoRALoader.load_and_apply_lora(
            lora_mapping=Fibo2LoRAMapping.get_mapping(),
            transformer=model.transformer,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
            bake_lora=bake_lora,
            dense_weights=weights.dense_component("transformer"),
        )
        del weights
        mx.clear_cache()

    @staticmethod
    def _init_config(model, model_config: ModelConfig, root: Path) -> None:
        model.model_config = model_config
        model.callbacks = CallbackRegistry()
        model.prompt_cache = {}
        model.tiling_config = None
        model.checkpoint_path = root
        # The fonts the images of a caption's text are drawn with, when the checkpoint has them
        model.fonts_dir = root / "fonts" if (root / "fonts").is_dir() else None
