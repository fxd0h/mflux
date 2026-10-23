import mlx.core as mx
from mlx import nn

from mflux.models.fibo2.model.fibo2_transformer.fibo2_cross_attention import Fibo2CrossAttention
from mflux.models.z_image.model.z_image_transformer.attention import ZImageAttention
from mflux.models.z_image.model.z_image_transformer.feed_forward import FeedForward


class Fibo2TransformerBlock(nn.Module):
    # Z-Image's block with a feed-forward width rounded down to a multiple of 64, and per-token modulation: the
    # noisy image tokens are modulated by the timestep, the clean ones (gist and context images) by timestep 0.
    # With cross_attention_dim it is an injection block, which also reads the text encoder through a gated
    # cross-attention between the attention and the feed-forward.
    def __init__(self, dim: int, n_heads: int, norm_eps: float, cross_attention_dim: int | None = None):
        super().__init__()
        self.attention = ZImageAttention(dim=dim, n_heads=n_heads, qk_norm=True, eps=norm_eps)
        self.feed_forward = FeedForward(dim=dim, hidden_dim=Fibo2TransformerBlock.ffn_hidden_dim(dim))
        self.attention_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.attention_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
        self.ffn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
        self.adaLN_modulation = [nn.Linear(min(dim, 256), 4 * dim, bias=True)]
        self.cross_attention = None
        if cross_attention_dim is not None:
            self.cross_attn_norm1 = nn.RMSNorm(dim, eps=norm_eps)
            self.cross_attention = Fibo2CrossAttention(dim, n_heads, cross_attention_dim, eps=norm_eps)
            self.cross_attn_norm2 = nn.RMSNorm(dim, eps=norm_eps)
            self.cross_attn_gate = mx.zeros((1,))

    def __call__(
        self,
        x: mx.array,
        freqs_cis: mx.array,
        t_noisy: mx.array,
        t_clean: mx.array,
        noise_mask: mx.array | None,
        text_key_value: tuple[mx.array, mx.array] | None = None,
    ) -> mx.array:
        scale_msa, gate_msa, scale_mlp, gate_mlp = self._modulation(t_noisy, t_clean, noise_mask)
        attn_out = self.attention(self.attention_norm1(x) * scale_msa, freqs_cis=freqs_cis)
        x = x + gate_msa * self.attention_norm2(attn_out)
        if self.cross_attention is not None:
            cross_out = self.cross_attention(self.cross_attn_norm1(x), text_key_value)
            x = x + mx.tanh(self.cross_attn_gate) * self.cross_attn_norm2(cross_out)
        ffn_out = self.feed_forward(self.ffn_norm1(x) * scale_mlp)
        return x + gate_mlp * self.ffn_norm2(ffn_out)

    @staticmethod
    def ffn_hidden_dim(dim: int) -> int:
        return (dim * 8 // 3) // 64 * 64

    def _modulation(self, t_noisy: mx.array, t_clean: mx.array, noise_mask: mx.array | None) -> list[mx.array]:
        noisy = Fibo2TransformerBlock._split(self.adaLN_modulation[0](t_noisy))
        if noise_mask is None:
            return noisy
        clean = Fibo2TransformerBlock._split(self.adaLN_modulation[0](t_clean))
        return [mx.where(noise_mask, n, c) for n, c in zip(noisy, clean)]

    @staticmethod
    def _split(modulation: mx.array) -> list[mx.array]:
        scale_msa, gate_msa, scale_mlp, gate_mlp = mx.split(modulation[:, None], 4, axis=2)
        return [1.0 + scale_msa, mx.tanh(gate_msa), 1.0 + scale_mlp, mx.tanh(gate_mlp)]
