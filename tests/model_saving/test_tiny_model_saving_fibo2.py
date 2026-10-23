import pytest

from mflux.models.fibo2.model.fibo2_text_encoder.fibo2_text_encoder import Fibo2TextEncoder
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer import Fibo2Transformer
from mflux.models.fibo2.weights.fibo2_weight_definition import Fibo2WeightDefinition
from tests.model_saving.tiny_checkpoint_helper import TinyCheckpointRoundtrip, TinyVAEStandIn


class TestTinyFibo2ModelSaving:
    @pytest.mark.fast
    def test_tiny_quantized_checkpoint_roundtrips_exactly(self, tmp_path):
        # Fibo 2.0 has no slow save test. This runs the ModelSaver -> WeightLoader -> WeightApplier seam with the real
        # Fibo 2.0 component classes at tiny dimensions: the Perceiver, an injection block and the Qwen3-VL vision tower.
        TinyCheckpointRoundtrip.save_and_reload_expecting_identical_weights(
            weight_definition=Fibo2WeightDefinition,
            make_components=TestTinyFibo2ModelSaving._tiny_components,
            base_path=tmp_path / "fibo2_tiny_q8",
            bits=8,
            # Force shard boundaries so index.json/weight_map multi-shard paths are
            # exercised — the size-based split never shards test-sized tensors.
            tensors_per_shard=8,
        )

    @staticmethod
    def _tiny_components():
        # Every input dimension is a multiple of 64 (MLX's quantization group size). The transformer's RoPE axes add
        # up to its head size (128 / 2 heads), and its image plane (max gist count + 1) fits in the first axis.
        return {
            "vae": TinyVAEStandIn(),
            "transformer": Fibo2Transformer(
                in_channels=16,
                dim=128,
                n_layers=2,
                n_refiner_layers=1,
                n_heads=2,
                cap_feat_dim=128,
                injection_layer_ids=(1,),
                perceiver_num_layers=1,
                min_num_gist_tokens=4,
                max_num_gist_tokens=8,
                gist_step=2,
                gist_min_text_len=4,
                gist_max_text_len=12,
                axes_dims=(16, 24, 24),
                axes_lens=(16, 8, 8),
            ),
            "text_encoder": Fibo2TextEncoder(
                {
                    "image_token_id": 99,
                    "text_config": {
                        "vocab_size": 128,
                        "hidden_size": 128,
                        "intermediate_size": 128,
                        "num_hidden_layers": 2,
                        "num_attention_heads": 2,
                        "num_key_value_heads": 1,
                        "head_dim": 64,
                        "max_position_embeddings": 2048,
                        "rms_norm_eps": 1e-6,
                        "rope_theta": 5000000,
                        "rope_scaling": {"mrope_section": [16, 8, 8]},
                        "attention_bias": False,
                    },
                    "vision_config": {
                        "depth": 1,
                        "hidden_size": 64,
                        "intermediate_size": 128,
                        "num_heads": 1,
                        "in_channels": 3,
                        "patch_size": 2,
                        "spatial_merge_size": 2,
                        "temporal_patch_size": 2,
                        "num_position_embeddings": 16,
                        "out_hidden_size": 128,
                        "deepstack_visual_indexes": [0],
                    },
                }
            ),
        }
