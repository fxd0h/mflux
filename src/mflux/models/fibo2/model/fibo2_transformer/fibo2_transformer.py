from dataclasses import dataclass

import mlx.core as mx
from mlx import nn

from mflux.models.common.config.model_config import ModelConfig
from mflux.models.fibo2.model.fibo2_transformer.fibo2_perceiver import Fibo2Perceiver
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer_block import Fibo2TransformerBlock
from mflux.models.z_image.model.z_image_transformer.final_layer import FinalLayer
from mflux.models.z_image.model.z_image_transformer.rope_embedder import RopeEmbedder
from mflux.models.z_image.model.z_image_transformer.timestep_embedder import TimestepEmbedder


@dataclass
class Fibo2TextConditioning:
    # What the transformer reads from one prompt. It does not depend on the timestep, so it is computed once per
    # prompt: the Perceiver's gist tokens, and the keys and values of each injection block's cross-attention.
    gist: mx.array
    key_values: list[tuple[mx.array, mx.array]]


class Fibo2Transformer(nn.Module):
    # Fibo 2.0's single-stream DiT. The gist tokens, the image tokens and the tokens of the images to edit run in one
    # stream through Z-Image-style blocks, in that order. Five of the blocks also cross-attend to the text encoder.
    def __init__(
        self,
        in_channels: int = 32,
        patch_size: int = 2,
        dim: int = 3584,
        n_layers: int = 44,
        n_refiner_layers: int = 2,
        n_heads: int = 28,
        norm_eps: float = 1e-5,
        cap_feat_dim: int = 7680,
        injection_layer_ids: tuple[int, ...] = (6, 14, 22, 30, 38),
        perceiver_num_layers: int = 2,
        min_num_gist_tokens: int = 256,
        max_num_gist_tokens: int = 512,
        gist_step: int = 64,
        gist_min_text_len: int = 256,
        gist_max_text_len: int = 2048,
        rope_theta: float = 256.0,
        t_scale: float = 1000.0,
        axes_dims: tuple[int, ...] = (32, 48, 48),
        axes_lens: tuple[int, ...] = (2048, 512, 512),
    ):
        super().__init__()
        self.t_scale = t_scale
        self.injection_layer_ids = tuple(injection_layer_ids)
        # The image sits on RoPE plane max_num_gist_tokens + 1 whatever the gist count, context image k on the k-th
        # plane after it; the gist tokens sit on planes 1 to their count
        self.image_plane = max_num_gist_tokens + 1
        token_dim = patch_size * patch_size * in_channels
        self.x_embedder = nn.Linear(token_dim, dim, bias=True)
        self.t_embedder = TimestepEmbedder(out_size=min(dim, 256), mid_size=1024)
        self.rope_embedder = RopeEmbedder(theta=rope_theta, axes_dims=list(axes_dims), axes_lens=list(axes_lens))
        self.noise_refiner = [Fibo2TransformerBlock(dim, n_heads, norm_eps) for _ in range(n_refiner_layers)]
        self.perceiver = Fibo2Perceiver(
            dim=dim,
            n_heads=n_heads,
            cross_attention_dim=cap_feat_dim,
            num_layers=perceiver_num_layers,
            norm_eps=norm_eps,
            min_gist_tokens=min_num_gist_tokens,
            max_gist_tokens=max_num_gist_tokens,
            gist_step=gist_step,
            gist_min_text_len=gist_min_text_len,
            gist_max_text_len=gist_max_text_len,
        )
        self.layers = [
            Fibo2TransformerBlock(dim, n_heads, norm_eps, cap_feat_dim if i in self.injection_layer_ids else None)
            for i in range(n_layers)
        ]
        self.final_layer = FinalLayer(dim, token_dim)

    def encode_text(self, text: mx.array) -> Fibo2TextConditioning:
        # text: (1, 1 + len(injection_layer_ids), length, cap_feat_dim), the Perceiver's features first
        dtype = ModelConfig.precision
        injection = [layer for layer in self.layers if layer.cross_attention is not None]
        return Fibo2TextConditioning(
            gist=self.perceiver(text[:, 0].astype(dtype)),
            key_values=[
                layer.cross_attention.key_value(text[:, index + 1].astype(dtype))
                for index, layer in enumerate(injection)
            ],
        )

    def __call__(
        self,
        latents: mx.array,
        sigma: mx.array,
        text: Fibo2TextConditioning,
        context_latents: list[mx.array] | None = None,
    ) -> mx.array:
        # latents: (1, 128, h, w), the normalized and patchified FLUX.2 VAE latents of the image being generated, and
        # context_latents the same for each image to edit. Returns the velocity, with the shape of latents.
        dtype = ModelConfig.precision
        _, channels, height, width = latents.shape
        num_image_tokens = height * width
        t_noisy = self.t_embedder(sigma.astype(mx.float32) * self.t_scale).astype(dtype)
        t_clean = self.t_embedder(mx.zeros(sigma.shape, dtype=mx.float32)).astype(dtype)

        # 1. The image and the images to edit share the patch embedding, each image on its own RoPE plane
        tokens = [Fibo2Transformer._tokens(latents)]
        ids = [Fibo2Transformer._grid((self.image_plane, 0, 0), height, width)]
        for k, context in enumerate(context_latents or []):
            tokens.append(Fibo2Transformer._tokens(context))
            ids.append(Fibo2Transformer._grid((self.image_plane + k + 1, 0, 0), context.shape[2], context.shape[3]))
        x = self.x_embedder(mx.concatenate(tokens, axis=1).astype(dtype))
        image_freqs = self.rope_embedder(mx.concatenate(ids, axis=0))

        # 2. Noise refiner. With images to edit, only the image tokens are noisy
        refiner_mask = None
        if context_latents:
            refiner_mask = Fibo2Transformer._noise_mask(0, num_image_tokens, x.shape[1])
        for block in self.noise_refiner:
            x = block(x, image_freqs, t_noisy, t_clean, refiner_mask)

        # 3. One stream: the gist tokens first, then the image tokens, then the context tokens
        num_gist = text.gist.shape[1]
        gist_ids = mx.stack([mx.arange(1, num_gist + 1), mx.zeros(num_gist), mx.zeros(num_gist)], axis=-1)
        freqs = mx.concatenate([self.rope_embedder(gist_ids.astype(mx.int32)), image_freqs], axis=0)
        x = mx.concatenate([text.gist, x], axis=1)
        noise_mask = Fibo2Transformer._noise_mask(num_gist, num_image_tokens, x.shape[1])
        key_values = iter(text.key_values)
        for layer in self.layers:
            key_value = next(key_values) if layer.cross_attention is not None else None
            x = layer(x, freqs, t_noisy, t_clean, noise_mask, key_value)

        # 4. Final layer on the image tokens only, back to the latent layout
        out = self.final_layer(x[:, num_gist : num_gist + num_image_tokens], t_noisy)
        return out.reshape(1, height, width, channels).transpose(0, 3, 1, 2)

    @staticmethod
    def _tokens(latents: mx.array) -> mx.array:
        # (1, 128, h, w) -> (1, h * w, 128): the patchified channels are already ordered (channel, dy, dx)
        return latents.reshape(latents.shape[0], latents.shape[1], -1).transpose(0, 2, 1)

    @staticmethod
    def _grid(start: tuple[int, int, int], height: int, width: int) -> mx.array:
        rows, cols = mx.meshgrid(mx.arange(height), mx.arange(width), indexing="ij")
        plane = mx.full((height * width,), start[0])
        return mx.stack([plane, rows.flatten() + start[1], cols.flatten() + start[2]], axis=-1).astype(mx.int32)

    @staticmethod
    def _noise_mask(start: int, length: int, total: int) -> mx.array:
        positions = mx.arange(total)
        return ((positions >= start) & (positions < start + length))[None, :, None]
