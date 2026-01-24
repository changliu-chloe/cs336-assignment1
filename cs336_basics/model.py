import torch
import math
from einops import einsum, rearrange

from cs336_basics.layers.linear import Linear
from cs336_basics.layers.rotary_embeddings import RotaryPositionalEmbedding
from cs336_basics.layers.layernorm import RMSNorm
from cs336_basics.layers.activations import SwiGLU
from cs336_basics.layers.embedding import Embedding


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute the softmax of the input tensor along the specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Dimension along which to compute the softmax. Default is -1.

    Returns:
        torch.Tensor: Softmax of the input tensor.
    """
    exp_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    sum_exp_x = torch.sum(exp_x, dim=dim, keepdim=True)
    return exp_x / sum_exp_x

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None):
    d_k = Q.shape[-1]
    attn_scores = einsum(Q, K, "... seq_q d, ... seq_k d -> ... seq_q seq_k")
    attn_scores = attn_scores / math.sqrt(d_k)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))

    attn_weights = softmax(attn_scores, dim=-1)
        
    output = einsum(attn_weights, V, "... seq_q seq_k, ... seq_k d -> ... seq_q d")

    return output

class CausalMultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int, device=None, dtype=None, **kwargs):
        super().__init__()

        self.wqkv = Linear(d_model, 3 * d_model, device, dtype)
        self.output_proj = Linear(d_model, d_model, device, dtype)

        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads

    def forward(
            self, 
            x: torch.Tensor, 
            rope: RotaryPositionalEmbedding | None = None, 
            token_positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        _, seq_len, _ = x.shape
        qkv = self.wqkv(x)

        q, k, v = qkv.split(self.d_model, dim=2)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = rope(q, token_positions)
            k = rope(k, token_positions)
        
        mask = ~torch.triu(torch.ones((seq_len, seq_len), device=x.device, dtype=torch.bool), diagonal=1)

        y = scaled_dot_product_attention(q, k, v, mask)
        y = rearrange(y, "b h s d -> b s (h d)")
        return self.output_proj(y)
    

class Block(torch.nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            rope: RotaryPositionalEmbedding | None = None,
            device=None,
            dtype=None,
            **kwargs,
    ):
        super().__init__()

        self.rope = rope

        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads, device, dtype)

        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

        ffn_type = kwargs.get("ffn_type", "swiglu")

        if ffn_type == "silu":
            # self.ffn = 
            pass
        elif ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device, dtype)
        else:
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")
    
    def forward(self, x: torch.Tensor):
        x += self.attn(self.ln1(x), self.rope)
        x += self.ffn(self.ln2(x))
        return x
    
class Transformer(torch.nn.Module):
    def __init__(
            self, 
            d_model: int,
            num_heads: int,
            d_ff: int,
            vocab_size: int,
            context_length: int,
            num_layers: int,
            rope_theta: float = 10000.0,
            device=None,
            dtype=None,
            **kwargs
    ):
        super().__init__()

        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device, dtype, **kwargs)

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        
        d_head = d_model // num_heads
        rope = RotaryPositionalEmbedding(rope_theta, d_head, context_length, device, dtype)
        
        self.layers = torch.nn.ModuleList(
            [Block(d_model, num_heads, d_ff, rope, device, dtype, **kwargs) for _ in range(num_layers)]
        )

        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device, dtype)

        if kwargs.get("weight_tying", False):
            self.lm_head.weight = self.token_embeddings.weight

    def forward(self, x: torch.Tensor):
        _, seq_len = x.shape

        if seq_len > self.context_length:
            raise ValueError(f"Input sequence length ({seq_len}) exceeds model context length ({self.context_length})")
        
        x = self.token_embeddings(x)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        x = self.lm_head(x)

        return x