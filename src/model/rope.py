from typing import Tuple
from torch import Tensor
from torch.nn import Module, Buffer
import torch

CosSin = Tuple[Tensor, Tensor]


def rotate_half(x: Tensor):
    """Rotates half the hidden dims of the input.
    
    Args:
        x (Tensor): Input tensor of shape [..., hidden_size]
    """
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)

class RotaryEmbedding(Module):
    """
    Based on https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/models/layers.py.
    """
    def __init__(self, dim: int, max_position_embeddings: int, base: float = 10000.0, device=None):
        """
        Docstring for __init__
        
        Args:
            dim (int): Dimension of the model.
            max_position_embeddings (int): Maximum number of position embeddings. We usually don't need to specify max length for sequences when implementing positional encodings, but we do this due to engineering reasons.
            base (float, optional): Base for the frequency calculation. Defaults to 10000.
            device: Device to use for the computation.
        """
        super().__init__()

        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = Buffer(emb.cos(), persistent=False)
        self.sin_cached = Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached