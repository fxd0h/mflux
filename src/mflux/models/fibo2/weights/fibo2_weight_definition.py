from typing import List

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.common.tokenizer import LanguageTokenizer
from mflux.models.common.weights.loading.weight_definition import ComponentDefinition, TokenizerDefinition
from mflux.models.fibo2.weights.fibo2_weight_mapping import Fibo2WeightMapping


class Fibo2WeightDefinition:
    # Loads briaai/Fibo-2.0-dev or briaai/Fibo-2.0-edit-turbo in their published layout, or a checkpoint written
    # by mflux-save. The two share the text encoder, the VAE and the processor; only the transformer differs.
    @staticmethod
    def get_components() -> List[ComponentDefinition]:
        return [
            ComponentDefinition(
                name="vae",
                hf_subdir="vae",
                precision=ModelConfig.precision,
                mapping_getter=Fibo2WeightMapping.get_vae_mapping,
            ),
            ComponentDefinition(
                name="transformer",
                hf_subdir="transformer",
                precision=ModelConfig.precision,
                mapping_getter=Fibo2WeightMapping.get_transformer_mapping,
            ),
            ComponentDefinition(
                name="text_encoder",
                hf_subdir="text_encoder",
                precision=ModelConfig.precision,
                mapping_getter=Fibo2WeightMapping.get_text_encoder_mapping,
                key_transform=Fibo2WeightDefinition.text_encoder_key,
            ),
        ]

    @staticmethod
    def get_tokenizers() -> List[TokenizerDefinition]:
        return [
            TokenizerDefinition(
                name="fibo2",
                hf_subdir="processor",
                encoder_class=LanguageTokenizer,
                padding="longest",
                download_patterns=["processor/**"],
            )
        ]

    @staticmethod
    def get_download_patterns() -> List[str]:
        return [
            "vae/*.safetensors",
            "vae/*.json",
            "transformer/*.safetensors",
            "transformer/*.json",
            "text_encoder/*.safetensors",
            "text_encoder/*.json",
            "processor/**",
            # The fonts training drew the images of a caption's text with
            "fonts/*.otf",
            "fonts/*.ttf",
        ]

    @staticmethod
    def text_encoder_key(key: str) -> str | None:
        # Fibo 2.0 reads Qwen3-VL up to the output of layer 35 of 36: the last layer is not loaded
        if key.startswith(f"model.language_model.layers.{Fibo2WeightMapping.NUM_TEXT_LAYERS}."):
            return None
        return key

    @staticmethod
    def quantization_predicate(path: str, module) -> bool:
        return hasattr(module, "to_quantized")
