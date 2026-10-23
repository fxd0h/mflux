from mflux.models.common.lora.mapping.lora_mapping import LoRAMapping, LoRATarget


class Fibo2LoRAMapping(LoRAMapping):
    # LoRAs for the Fibo 2.0 transformer. diffusers' trainer adapts every linear layer and saves PEFT keys under
    # "transformer."; a PEFT model saves them under "base_model.model.", ComfyUI under "diffusion_model.". The module
    # paths are diffusers' except the two layers Z-Image's modules name instead of indexing a Sequential.
    PREFIXES = ("", "transformer.", "base_model.model.", "diffusion_model.")
    ATTENTION = ("to_q", "to_k", "to_v", "to_out.0")
    FEED_FORWARD = ("w1", "w2", "w3")

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        m = Fibo2LoRAMapping
        targets = [
            m._target("x_embedder"),
            m._target("t_embedder.linear1", source="t_embedder.mlp.0"),
            m._target("t_embedder.linear2", source="t_embedder.mlp.2"),
            m._target("final_layer.linear"),
            m._target("final_layer.adaLN_modulation.0", source="final_layer.adaLN_modulation.1"),
        ]
        for block in ("noise_refiner.{block}", "layers.{block}"):
            targets += [m._target(f"{block}.attention.{name}") for name in m.ATTENTION]
            targets += [m._target(f"{block}.feed_forward.{name}") for name in m.FEED_FORWARD]
            targets.append(m._target(f"{block}.adaLN_modulation.0"))
        # Only the injection blocks have a cross-attention; the pattern matches the indices the LoRA names
        targets += [m._target(f"layers.{{block}}.cross_attention.{name}") for name in m.ATTENTION]
        for attention in ("cross_attention", "self_attention"):
            targets += [m._target(f"perceiver.layers.{{block}}.{attention}.{name}") for name in m.ATTENTION]
        targets += [m._target(f"perceiver.layers.{{block}}.ffn.{name}") for name in m.FEED_FORWARD]
        return targets

    @staticmethod
    def _target(model_path: str, source: str | None = None) -> LoRATarget:
        prefixed = [f"{prefix}{source or model_path}" for prefix in Fibo2LoRAMapping.PREFIXES]
        up = ("lora_B.weight", "lora_B.default.weight", "lora_up.weight")
        down = ("lora_A.weight", "lora_A.default.weight", "lora_down.weight")
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=[f"{path}.{name}" for path in prefixed for name in up],
            possible_down_patterns=[f"{path}.{name}" for path in prefixed for name in down],
            possible_alpha_patterns=[f"{path}.alpha" for path in prefixed],
        )
