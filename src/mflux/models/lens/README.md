# Lens

This directory contains MFLUX's MLX implementation of **Microsoft Lens (Turbo)**.

Lens Turbo pairs a 3.8B dual-stream MMDiT (48 blocks) with a GPT-OSS 20B multi-layer text
encoder and reuses the FLUX.2 VAE already in-tree. It is a 4-step distillation with CFG
internalized, so `--guidance`, `--negative-prompt` and `--scheduler` are accepted but
ignored (the CLI warns when you pass them). Weights download from
[`Comfy-Org/Lens`](https://huggingface.co/Comfy-Org/Lens) on first run.

## Example

```sh
mflux-generate-lens \
  --prompt "A cozy bookshop cafe at dusk, warm light through the window, a cat sleeping on a stack of books" \
  --width 512 \
  --height 512 \
  --seed 7 \
  --steps 4 \
  -q 8
```

Quantization (`-q 3|4|5|6|8`) is supported and recommended: the GPT-OSS encoder is the
bulk of the memory footprint. LoRA flags are not available for this model.

<details>
<summary>Python API</summary>

```python
from mflux.models.common.config import ModelConfig
from mflux.models.lens.variants.txt2img.lens_image import LensImage

model = LensImage(
    model_config=ModelConfig.lens_turbo(),
    quantize=8,
)
image = model.generate_image(
    seed=7,
    prompt="A cozy bookshop cafe at dusk, warm light through the window, a cat sleeping on a stack of books",
    width=512,
    height=512,
    num_inference_steps=4,
)
image.save(path="lens.png")
```

</details>
