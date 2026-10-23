import mlx.core as mx
from mlx import nn
from mlx.core.fast import scaled_dot_product_attention


class Fibo2CrossAttention(nn.Module):
    # Queries from the token stream, keys and values from the text encoder's features. No RoPE and no mask:
    # one prompt is encoded at a time, so nothing is padded.
    def __init__(self, dim: int, n_heads: int, kv_dim: int, eps: float = 1e-5):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(kv_dim, dim, bias=False)
        self.to_v = nn.Linear(kv_dim, dim, bias=False)
        self.to_out = [nn.Linear(dim, dim, bias=False)]
        self.norm_q = nn.RMSNorm(self.head_dim, eps=eps)
        self.norm_k = nn.RMSNorm(self.head_dim, eps=eps)

    def __call__(self, hidden_states: mx.array, key_value: tuple[mx.array, mx.array]) -> mx.array:
        batch, length, dim = hidden_states.shape
        query = self.norm_q(self.to_q(hidden_states).reshape(batch, length, self.n_heads, self.head_dim))
        key, value = key_value
        out = scaled_dot_product_attention(query.transpose(0, 2, 1, 3), key, value, scale=self.head_dim**-0.5)
        return self.to_out[0](out.transpose(0, 2, 1, 3).reshape(batch, length, dim))

    def key_value(self, encoder_hidden_states: mx.array) -> tuple[mx.array, mx.array]:
        # The keys and values depend on the text only, so a generation computes them once per prompt
        batch, length, _ = encoder_hidden_states.shape
        key = self.norm_k(self.to_k(encoder_hidden_states).reshape(batch, length, self.n_heads, self.head_dim))
        value = self.to_v(encoder_hidden_states).reshape(batch, length, self.n_heads, self.head_dim)
        return key.transpose(0, 2, 1, 3), value.transpose(0, 2, 1, 3)
