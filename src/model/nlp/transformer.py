from typing import Literal, Optional
import einops
import torch
from torch import Tensor
from torch.nn import Linear, Module, ModuleList, RMSNorm
import torch.nn.functional as F
from torch.nn.functional import scaled_dot_product_attention
from src.model.rope import apply_rotary_pos_emb, CosSin, RotaryEmbedding
from transformers import AutoTokenizer

class Attention(Module):
    def __init__(self, hidden_size: int, head_dim: int, num_heads: int, causal: bool = False):
        """ 
        Args:
            hidden_size (int): Hidden size of the model
            head_dim (int): Dimension of each head
            num_heads (int): Number of heads
            causal (bool): Whether the attention is causal
        
        With this notation we don't need to worry about num_heads dividing hidden_size. We can have any head_dim we want.
        """
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.causal = causal

        self.qkv_proj = Linear(self.hidden_size, 3*self.num_heads * self.head_dim, bias=False)
        self.o_proj = Linear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: Tensor, **kwargs) -> Tensor:
        """ 
        Args:
            cos_sin (CosSin): Cosine and sine of the rotary position embedding
            hidden_states (Tensor): Hidden states of the model of shape [bs, seq_len, hidden_size]
        """
        batch_size, seq_len, _ = hidden_states.shape

        # qkv: [bs, seq_len, 3*num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, 3 * self.num_heads, self.head_dim)
        query, key, value = qkv.chunk(3, dim=2) 

        # RoPE
        if cos_sin is not None:
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos, sin)

        # flash attn
        query, key, value = map(lambda t: einops.rearrange(t, 'B S H D -> B H S D'), (query, key, value)) # needed for scaled_dot_product_attention
        attn_output = scaled_dot_product_attention(query=query, key=key, value=value, is_causal=self.causal, **kwargs) # of shape [B, H, S, D] can pass attn_mask or dropout_p
        attn_output = einops.rearrange(attn_output, 'B H S D -> B S H D')
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size) # self.output_size = num_heads * head_dim
        return self.o_proj(attn_output)

class SwiGLU(Module):
    """ SwiGLU activation function. Better than GeLU in most cases.
    This implementation comes from https://github.com/SamsungSAILMontreal/TinyRecursiveModels/blob/main/models/layers.py
    
    Args:
        hidden_size (int): Hidden size of the model
        expansion (float): Expansion factor.
    """
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = self._find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        # chunk gate with data
        self.gate_data_proj = Linear(hidden_size, inter * 2, bias=False)
        self.down_proj = Linear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, data = self.gate_data_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * data)
    
    @staticmethod
    def _find_multiple(a, b):
        return (-(a // -b)) * b

class TransformerDecoderLayer(Module):
    def __init__(self, hidden_size: int, num_heads: int, expansion: float, pre_norm: bool = False) -> None:
        """ A single layer of the Transformer decoder. Here causal is always True since it's a decoder.
        Args:
            hidden_size (int): Hidden size of the model
            num_heads (int): Number of heads
            expansion (float): Expansion factor.
            pre_norm (bool): Whether to use pre-norm or post-norm
        """
        super().__init__()

        self.hidden_size: int = hidden_size
        self.num_heads: int = num_heads
        self.expansion: float = expansion
        self.pre_norm: bool = pre_norm
        self.self_attn = Attention(
            hidden_size=self.hidden_size,
            head_dim=self.hidden_size // self.num_heads,
            num_heads=self.num_heads,
            causal=True
        )
        self.mlp = SwiGLU(
            hidden_size=self.hidden_size,
            expansion=self.expansion,
        )
        self.rms_norm1 = RMSNorm(self.hidden_size)
        self.rms_norm2 = RMSNorm(self.hidden_size)

    def forward_postnorm(self, cos_sin: CosSin, hidden_states: Tensor) -> Tensor:
        """
        Args:
            cos_sin (CosSin): Cosine and sine of the rotary position embedding
            hidden_states (Tensor): Hidden states of the model of shape [bs, seq_len, hidden_size]
        """
        # self attention
        hidden_states = self.rms_norm1(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states))

        # mlp
        out = self.mlp(hidden_states)
        hidden_states = self.rms_norm2(hidden_states + out)
        return hidden_states
    
    def forward_prenorm(self,cos_sin: CosSin, hidden_states: Tensor) -> Tensor:
        """
        Args:
            cos_sin (CosSin): Cosine and sine of the rotary position embedding
            hidden_states (Tensor): Hidden states of the model of shape [bs, seq_len, hidden_size]
        """
        hidden_states = hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=self.rms_norm1(hidden_states))
        hidden_states = hidden_states + self.mlp(self.rms_norm2(hidden_states))
        return hidden_states
    
    def forward(self, cos_sin: CosSin, hidden_states: Tensor) -> Tensor:
        """
        Args:
            cos_sin (CosSin): Cosine and sine of the rotary position embedding
            hidden_states (Tensor): Hidden states of the model of shape [bs, seq_len, hidden_size]
        """
        if self.pre_norm:
            return self.forward_prenorm(cos_sin=cos_sin, hidden_states=hidden_states)
        else:
            return self.forward_postnorm(cos_sin=cos_sin, hidden_states=hidden_states)
        

class TransformerDecoderBlock(Module):
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, expansion: float, pre_norm: bool = False) -> None:
        super().__init__()
        self.layers = ModuleList([
                TransformerDecoderLayer(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    expansion=expansion,
                    pre_norm=pre_norm,
                ) for _ in range(num_layers)
            ])
    
    def forward(self, cos_sin: CosSin, hidden_states: Tensor) -> Tensor:
        """
        Args:
            cos_sin (CosSin): Cosine and sine of the rotary position embedding
            hidden_states (Tensor): Hidden states of the model of shape [bs, seq_len, hidden_size]
        Returns:
            hidden_states (Tensor): Hidden states after transformer layers of shape [bs, seq_len, hidden_size]
        """
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
        return hidden_states

class TransformerDecoderBlockInjected(TransformerDecoderBlock):
    """
    A general transformer block that receives two inputs: hidden_states and input_injection. Uses RoPE as default positional encoding and allows input injection either by summation or concatenation. #TODO other positional encodings."""
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, expansion: float, pre_norm: bool = False, injection_type: Literal["sum", "concat"] = "sum") -> None:
        super().__init__(num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads, expansion=expansion, pre_norm=pre_norm)
        self.injection_type = injection_type

    def forward(self, cos_sin: CosSin, hidden_states: Tensor, input_injection: Optional[Tensor] = None) -> Tensor:
        """
        Args:
            cos_sin (CosSin): Cosine and sine of the rotary position embedding
            hidden_states (Tensor): Hidden states of the model of shape [bs, context_len, hidden_size]
            input_injection (Tensor): Input injection of shape [bs, injection_len, hidden_size] if injection_type is "sum" or [bs, seq_len + input_injection_size, hidden_size] if injection_type is "concat"
        Returns:
            hidden_states (Tensor): Hidden states after transformer layers of shape [bs, seq_len, hidden_size] where seq_len is context_len if injection_type is "sum" or context_len + injection_len if injection_type is "concat".
        """
        if input_injection is not None:
            if self.injection_type == "sum":
                # In this case the sequence length could be different, so we need to pad before.
                if hidden_states.shape[1] == input_injection.shape[1]:
                    hidden_states = hidden_states + input_injection # bs, context_len, hidden_size
                else:
                    raise ValueError(f"Sequence length of hidden_states ({hidden_states.shape[1]}) and input_injection ({input_injection.shape[1]}) must be the same for sum injection. Maybe your padding is wrong?")
            elif self.injection_type == "concat":
                hidden_states = torch.cat([hidden_states, input_injection], dim=1) # bs, seq_len + injection_len, hidden_size
            else:
                raise ValueError(f"Unknown injection type: {self.injection_type}")
        for layer in self.layers:
            hidden_states = layer(cos_sin=cos_sin, hidden_states=hidden_states)
        return hidden_states

class ScalarEnergyHead(Module):
    def __init__(self, hidden_size: int, algorithm: Literal['hermitian', 'symmetric'] = 'symmetric', rank: int = 1) -> None:
        """ A projection head that computes a scalar energy from the hidden states.
        Args:
            hidden_size (int): Hidden size of the model
            algorithm (Literal['hermitian', 'symmetric']): Algorithm to use for the energy head. The rank is used to define trainable matrix B of shape [hidden_size, rank] such that B(x)^2 is the energy per timestep.
            rank (int): Rank of the energy head.
        """
        super().__init__()
        self.algorithm = algorithm
        self.rank = rank
        self.hidden_size = hidden_size
        if algorithm == 'hermitian':
            raise NotImplementedError # need complex numbers #TODO
        elif algorithm == 'symmetric':
            self.B = Linear(self.hidden_size, self.rank, bias=False)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def compute_latent_energies(self, hidden_states: Tensor) -> Tensor:
        """
        Compute latent energies based on the algorithm. Instead of being a scalar per timestep, latent energies are a vector of size rank per timestep.
        Args:
            hidden_states (Tensor): Hidden states of the model of shape [bs, seq_len, hidden_size]
        Returns:
            latent_energies (Tensor): Latent energies of shape [bs, seq_len, rank]
        """
        latent_energies = self.B(hidden_states) # [B, L, rank]
        return latent_energies
    
    def compute_energies(self, latent_energies: Tensor) -> Tensor:
        """
        Compute energies per timestep given latent energies. Consider B and x as the latent energies and hidden states respectively. Then E = x^T B^T B x = ||Bx||^2.
        Args:
            latent_energies (Tensor): Latent energies of shape [bs, seq_len, rank]
        Returns:
            energies_per_timestep (Tensor): Energies per timestep of shape [bs, seq_len]
        """
        energies_per_timestep = (latent_energies**2).sum(dim=-1) # [B, L]
        return energies_per_timestep
    
    def compute_energy(self, energies_per_timestep: Tensor) -> Tensor:
        """
        Compute total energy given energies per timestep.
        Args:
            energies_per_timestep (Tensor): Energies per timestep of shape [bs, seq_len]
        Returns:
            energy (Tensor): Energy of shape [bs]
        """
        energy = energies_per_timestep.sum(dim=-1) # [B]
        return energy
    
    def forward(self, hidden_states: Tensor) -> Tensor:
        """
        Compute total energy for the sequence.
        Variables latent_energies, energies_per_timestep are exposed for future use.
        Args:
            hidden_states (Tensor): Hidden states of the model of shape [bs, seq_len, hidden_size]
        Returns:
            energy (Tensor): Energy of shape [bs]
        """
        latent_energies = self.compute_latent_energies(hidden_states) # [B, L, rank]
        energies_per_timestep = self.compute_energies(latent_energies) # [B, L]
        energy = self.compute_energy(energies_per_timestep) # [B]
        return energy

class EnergyTransformerDecoder(TransformerDecoderBlockInjected):
    def __init__(self, num_layers: int, hidden_size: int, num_heads: int, expansion: float, pre_norm: bool = False, algorithm: Literal['hermitian', 'symmetric'] = 'symmetric', rank: int = 1, pos_encoding: str = "rope", max_seq_len: int = 2048, tokenizer: str = "EleutherAI/gpt-neox-20b"):
        """
        Energy Based Transformer receives a context (hidden_states) and a candidate (input_injection) and computes an energy score for the pair.
        Args:
        """
        super().__init__(num_layers=num_layers, hidden_size=hidden_size, num_heads=num_heads, expansion=expansion, pre_norm=pre_norm, injection_type="concat")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        if pos_encoding == "rope":
            self.positional_embedding = RotaryEmbedding(dim = hidden_size // num_heads, max_position_embeddings = max_seq_len+1) # AR means context has max_seq_len and candidate has 1 as sequence length.
        else:
            raise NotImplementedError(f"Positional encoding {pos_encoding} not implemented") #TODO LieRE or GraPE
        self.energy_head = ScalarEnergyHead(hidden_size = hidden_size, algorithm=algorithm, rank=rank)

    def forward(self, cos_sin: CosSin, context: Tensor, candidate: Tensor) -> Tensor:
        """
        Args:
            cos_sin (CosSin): Cosine and sine of the rotary position embedding
            context (Tensor): Context of shape [bs, context_len, hidden_size]
            candidate (Tensor): Candidate of shape [bs, 1, hidden_size]
        Returns:
            energy (Tensor): Energy of shape [bs]
        """
        hidden_states = super().forward(cos_sin=cos_sin, hidden_states=context, input_injection=candidate) # [B, context_len+1, hidden_size]
        energy = self.energy_head(hidden_states) # [B]
        return energy


def main():
    config_tiny = {
        "num_layers": 2,
        "hidden_size": 512,
        "num_heads": 8,
        "expansion": 4.0,
        "pre_norm": False,
        "pos_encoding": "rope",
        "algorithm": "symmetric",
        "rank": 1,
        "max_seq_len": 1024
    }
    model = EnergyTransformerDecoder(**config_tiny)
    bsz = 2
    seq_len = 1024
    context = torch.randn(bsz, seq_len, config_tiny["hidden_size"])
    candidate = torch.randn(bsz, 1, config_tiny["hidden_size"])
    cos_sin = model.positional_embedding()
    energy = model(cos_sin=cos_sin, context=context, candidate=candidate)
    print("Energy shape:", energy.shape)  # Should be [bsz]


if __name__ == "__main__":
    main()