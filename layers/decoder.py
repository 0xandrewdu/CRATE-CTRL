import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init

class Attention_Decode(nn.Module):
    """
    Same as encoder attention layer
    """
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8, # number of subspaces
        dim_head: int = 8, # number of vectors in each basis U_k; higher dim means less redundancy in coverage by dif 
        dropout: float = 0.,
        scale: float = 0.
    ) -> None:
        super(Attention_Decode, self).__init__()
        self.num_heads = num_heads
        self.input_dim = dim
        self.dim_head = dim_head
        self.dropout = nn.Dropout(dropout)
        self.kp = num_heads * dim_head
        self.I = nn.Parameter(torch.eye(self.input_dim), requires_grad=False)
        self.UT = nn.Linear(dim, self.kp, bias=False) 
        self.attend = nn.Softmax(dim=-1)
        self.scale = scale if scale > 0 else dim_head ** -0.5 
    
    def forward(
            self,
            ZT: torch.Tensor,
        ) -> torch.Tensor:
        ZTU = rearrange(self.UT(ZT), 'b n (h d) -> b h n d', h=self.num_heads)
        UTZ = ZTU.transpose(-1, -2)
        UT = self.UT(self.I).T

        ZTUUTZ = torch.matmul(ZTU, UTZ) * self.scale
        attn = self.dropout(self.attend(ZTUUTZ))

        SSA_outs = torch.matmul(attn, ZTU)
        SSA_outs = rearrange(SSA_outs, 'b h n d -> b n (h d)')

        MSSA_out = torch.matmul(SSA_outs, UT)
        return MSSA_out
    
class MLP_Decode(nn.Module):
    """
    Weight tie with the corresponding encoder in forward.py
    """
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))

    def forward(self, x):
        return F.linear(x, self.weight, bias=None)
    
class CRATE_Transformer_Decode(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=8, dropout=0.):
        super().__init__()
        self.norm_attn, self.norm_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention_Decode(dim=dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout)
        self.mlp = MLP_Decode(dim=dim, dropout=dropout)
    
    def forward(self, x):
        z_half = self.norm_attn(self.mlp(self.norm_mlp(x)))
        z_out = z_half - self.attn(z_half)
        return z_out
