import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mamba_ssm import Mamba


class FFN(nn.Module):
    """
    Feed Forward Network (FFN) with TD Encoding
    """

    def __init__(self, d_model, d_ff=None, dropout=0.1, activation="gelu"):
        super().__init__()
        d_ff = d_ff or d_model * 4
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = getattr(F, activation)

    def forward(self, x):
        # x: (B, L, D) or (B, N, D) for time series
        return self.fc2(self.dropout(self.activation(self.fc1(x))))


class SMambaBlock(nn.Module):
    """
    s-mamba block: Combination of bi-mamba and FFN with TD Encoding
    """

    def __init__(self, d_model, d_state, d_conv=2, expand=2, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Forward Mamba
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=True,
        )

        # Backward Mamba
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            use_fast_path=True,
        )

        # FFN with TD Encoding
        self.ffn = FFN(d_model, dropout=dropout)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, inference_params=None):
        # x: (B, L, D)
        B, L, D = x.shape

        # Bidirectional Mamba
        # Forward pass
        x_fwd = self.mamba_fwd(x)

        # Backward pass (reverse sequence, apply mamba, then reverse back)
        x_bwd = torch.flip(x, dims=[1])
        x_bwd = self.mamba_bwd(x_bwd)
        x_bwd = torch.flip(x_bwd, dims=[1])

        # Combine forward and backward
        x_mamba = x_fwd + x_bwd
        x = self.norm1(x + self.dropout(x_mamba))

        # FFN block
        x_ffn = self.ffn(x)
        x = self.norm2(x + self.dropout(x_ffn))

        return x

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        # Return caches for both forward and backward mamba
        return (
            self.mamba_fwd.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            ),
            self.mamba_bwd.allocate_inference_cache(
                batch_size, max_seqlen, dtype=dtype, **kwargs
            ),
        )


class AttentionFFNBlock(nn.Module):
    """
    Attention + FFN block: Multi-head attention followed by feed-forward network
    """

    def __init__(self, d_model, n_heads=8, d_ff=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Multi-head attention
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # FFN
        self.ffn = FFN(d_model, d_ff, dropout)

        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, inference_params=None):
        # x: (B, L, D)
        B, L, D = x.shape

        # Multi-head self-attention
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Apply causal mask
        causal_mask = torch.triu(
            torch.ones(L, L, dtype=torch.bool, device=x.device), diagonal=1
        )
        scores = scores.masked_fill(
            causal_mask.unsqueeze(0).unsqueeze(0), float("-inf")
        )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, D)
        attn_output = self.out_proj(attn_output)

        # Residual connection and norm
        x = self.norm1(x + self.dropout(attn_output))

        # FFN block
        x_ffn = self.ffn(x)
        x = self.norm2(x + self.dropout(x_ffn))

        return x

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        # Attention doesn't need a special cache for now
        return None


class LinearBlock(nn.Module):
    """
    Simple Linear block for ablation studies
    This block replaces the mamba layers with simple linear transformations
    """

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model

        # Simple linear transformation
        self.linear = nn.Linear(d_model, d_model)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, inference_params=None):
        # x: (B, L, D)
        # Simple linear transformation with residual connection
        x_linear = self.linear(x)
        x = self.norm(x + self.dropout(x_linear))
        return x

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        # Linear block doesn't need a special cache
        return None
