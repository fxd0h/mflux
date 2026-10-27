from mflux.models.common.lora.mapping.lora_mapping import LoRATarget


class Fibo2LoRAMapping:
    # Every linear layer, as diffusers' trainer adapts them; only the timestep MLP and final modulation are renamed
    PREFIXES = ("", "transformer.", "diffusion_model.", "base_model.model.")
    ATTENTION = ("to_q", "to_k", "to_v", "to_out.0")
    BLOCK = (
        *(f"attention.{name}" for name in ATTENTION),
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
        "adaLN_modulation.0",
    )
    INJECTION = tuple(f"cross_attention.{name}" for name in ATTENTION)
    PERCEIVER_LAYER = (
        *(f"cross_attention.{name}" for name in ATTENTION),
        *(f"self_attention.{name}" for name in ATTENTION),
        "ffn.w1",
        "ffn.w2",
        "ffn.w3",
    )

    @staticmethod
    def get_mapping() -> list[LoRATarget]:
        targets = [
            Fibo2LoRAMapping._target("x_embedder"),
            Fibo2LoRAMapping._target("t_embedder.linear1", "t_embedder.mlp.0"),
            Fibo2LoRAMapping._target("t_embedder.linear2", "t_embedder.mlp.2"),
            Fibo2LoRAMapping._target("final_layer.linear"),
            Fibo2LoRAMapping._target("final_layer.adaLN_modulation.0", "final_layer.adaLN_modulation.1"),
        ]
        targets += [Fibo2LoRAMapping._target(f"noise_refiner.{{block}}.{name}") for name in Fibo2LoRAMapping.BLOCK]
        targets += [
            Fibo2LoRAMapping._target(f"layers.{{block}}.{name}")
            for name in Fibo2LoRAMapping.BLOCK + Fibo2LoRAMapping.INJECTION
        ]
        targets += [
            Fibo2LoRAMapping._target(f"perceiver.layers.{{block}}.{name}") for name in Fibo2LoRAMapping.PERCEIVER_LAYER
        ]
        targets += Fibo2LoRAMapping._comfy_targets()
        return targets

    @staticmethod
    def _comfy_targets() -> list[LoRATarget]:
        # ComfyUI's layout (Comfy-Org/Fibo-2.0), which ai-toolkit also saves, fuses self-attention q, k and v into
        # attention.qkv, cross-attention k and v into to_kv, and the feed-forward gate and up into w13 (gate first),
        # and drops the .0 of to_out. A LoRA on a fused layer splits along its output: every part keeps the shared
        # down weight and takes its slice of the up weight.
        split = Fibo2LoRAMapping._split_targets
        targets = []
        for stack in ("noise_refiner.{block}", "layers.{block}"):
            targets += split(f"{stack}.attention.qkv", [f"{stack}.attention.to_{x}" for x in "qkv"])
            targets += split(f"{stack}.feed_forward.w13", [f"{stack}.feed_forward.w1", f"{stack}.feed_forward.w3"])
            targets.append(Fibo2LoRAMapping._target(f"{stack}.attention.to_out.0", f"{stack}.attention.out"))
        for attention in ("layers.{block}.cross_attention", "perceiver.layers.{block}.cross_attention", "perceiver.layers.{block}.self_attention"):  # fmt: off
            targets += split(f"{attention}.to_kv", [f"{attention}.to_k", f"{attention}.to_v"])
            targets.append(Fibo2LoRAMapping._target(f"{attention}.to_out.0", f"{attention}.to_out"))
        targets += split("perceiver.layers.{block}.ffn.w13", ["perceiver.layers.{block}.ffn.w1", "perceiver.layers.{block}.ffn.w3"])  # fmt: off
        return targets

    @staticmethod
    def _split_targets(source_path: str, model_paths: list[str]) -> list[LoRATarget]:
        count = len(model_paths)
        return [
            Fibo2LoRAMapping._target(
                path, source_path, up_transform=lambda up, i=i: Fibo2LoRAMapping._part(up, i, count)
            )  # fmt: off
            for i, path in enumerate(model_paths)
        ]

    @staticmethod
    def _part(up, index: int, count: int):
        if up.shape[0] % count != 0:
            raise ValueError(f"A fused Fibo 2 LoRA up weight {tuple(up.shape)} does not split into {count} equal parts")
        rows = up.shape[0] // count
        return up[index * rows : (index + 1) * rows]

    @staticmethod
    def _target(model_path: str, source_path: str | None = None, up_transform=None) -> LoRATarget:
        source = source_path or model_path
        prefixes = Fibo2LoRAMapping.PREFIXES
        defaults = ("", ".default")
        return LoRATarget(
            model_path=model_path,
            possible_up_patterns=[f"{p}{source}.lora_B{d}.weight" for p in prefixes for d in defaults],
            possible_down_patterns=[f"{p}{source}.lora_A{d}.weight" for p in prefixes for d in defaults],
            possible_alpha_patterns=[f"{p}{source}.alpha" for p in prefixes],
            up_transform=up_transform,
        )
