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
    Multiple subspace self attention layer for the forward pass.
    """
    def __init__(
        self, 
        dim: int,
        num_heads: int = 8, # number of subspaces
        dim_head: int = 8, # number of vectors in each basis U_k; higher dim means less redundancy in coverage by dif 
        dropout: float = 0.,
        scale: float = 0.
    ) -> None:
        super(MSSA, self).__init__()
        self.num_heads = num_heads
        self.input_dim = dim
        self.dim_head = dim_head
        self.dropout = nn.Dropout(dropout)
        self.kp = num_heads * dim_head
        self.I = nn.Parameter(torch.eye(self.input_dim), requires_grad=False)
        self.UT = nn.Linear(dim, self.kp, bias=False) # note that nn.Linear right multiplies by W^T
        self.attend = nn.Softmax(dim=-1)
        self.scale = scale if scale > 0 else dim_head ** -0.5 # reusing sqrt dim from aiayn, is equiv to p/(N * eps^2) for some eps anyway
    
    def forward(
            self,
            ZT: torch.Tensor,
        ) -> torch.Tensor:
        raise NotImplementedError
    
class MLP_Decode(nn.Module):
    """
    Weight tied with the corresponding encoder in forward.py
    """
    def __init__(self, dim, dropout=0.):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))

    def forward(self, x):
        return F.linear(x, self.weight, bias=None)
    
class CRATE_Transformer_Decode(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = Attention_Encode()
        self.norm1 = 
        self.mlp = 