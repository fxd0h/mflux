import math

import mlx.core as mx
from mlx import nn

from mflux.models.fibo2.model.fibo2_transformer.fibo2_cross_attention import Fibo2CrossAttention
from mflux.models.fibo2.model.fibo2_transformer.fibo2_transformer_block import Fibo2TransformerBlock
from mflux.models.z_image.model.z_image_transformer.attention import ZImageAttention
from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward


class Fibo2PerceiverLayer(nn.Module):
    # The gist tokens read the text encoder (cross-attention), then each other (self-attention, no RoPE), then an FFN.
    def __init__(self, dim: int, n_heads: int, cross_attention_dim: int, norm_eps: float):
        super().__init__()
        self.cross_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.cross_attention = Fibo2CrossAttention(dim, n_heads, cross_attention_dim, eps=norm_eps)
        self.self_attn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.self_attention = ZImageAttention(dim=dim, n_heads=n_heads, qk_norm=True, eps=norm_eps)
        self.ffn_norm = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn = FeedForward(dim=dim, hidden_dim=Fibo2TransformerBlock.ffn_hidden_dim(dim))

    def __call__(self, latents: mx.array, text: mx.array) -> mx.array:
        key_value = self.cross_attention.key_value(text)
        latents = latents + self.cross_attention(self.cross_attn_norm(latents), key_value)
        latents = latents + self.self_attention(self.self_attn_norm(latents))
        return latents + self.ffn(self.ffn_norm(latents))


class Fibo2Perceiver(nn.Module):
    # Squeezes the text encoder's features into 256 to 512 gist tokens, more for longer prompts.
    def __init__(
        self,
        dim: int,
        n_heads: int,
        cross_attention_dim: int,
        num_layers: int,
        norm_eps: float,
        min_gist_tokens: int,
        max_gist_tokens: int,
        gist_step: int,
        gist_min_text_len: int,
        gist_max_text_len: int,
    ):
        super().__init__()
        self.latent_queries = mx.zeros((max_gist_tokens, dim))
        self.layers = [Fibo2PerceiverLayer(dim, n_heads, cross_attention_dim, norm_eps) for _ in range(num_layers)]
        self.norm_out = nn.RMSNorm(dim, eps=norm_eps)
        self._gist_range = (min_gist_tokens, max_gist_tokens, gist_step, gist_min_text_len, gist_max_text_len)

    def __call__(self, text: mx.array) -> mx.array:
        latents = self.latent_queries[None, : self.gist_count(text.shape[1])]
        for layer in self.layers:
            latents = layer(latents, text)
        return self.norm_out(latents)

    def gist_count(self, text_length: int) -> int:
        # Lengths between the minimum and maximum text lengths map linearly onto the gist range, rounded up to a
        # multiple of the step
        min_gist, max_gist, step, min_len, max_len = self._gist_range
        ratio = min(1.0, max(0.0, (float(text_length) - min_len) / (max_len - min_len)))
        count = int(math.ceil((min_gist + ratio * (max_gist - min_gist)) / step)) * step
        return max(min_gist, min(max_gist, count))
