import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init

class Attention_Encode(nn.Module):
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
        # multiplies by the subspace bases by recovering them from the nn.Linear layer as opposed to the
        # new trainable weights W done in the paper's implementation--check with Druv to see if this is ok
        ZTU = rearrange(self.UT(ZT), 'b n (h d) -> b h n d', h=self.num_heads)
        UTZ = ZTU.transpose(-1, -2)
        UT = self.UT(self.I).T

        ZTUUTZ = torch.matmul(ZTU, UTZ) * self.scale
        attn = self.dropout(self.attend(ZTUUTZ))

        SSA_outs = torch.matmul(attn, ZTU)
        SSA_outs = rearrange(SSA_outs, 'b h n d -> b n (h d)')

        MSSA_out = torch.matmul(SSA_outs, UT)
        return MSSA_out
    
class MLP_Encode(nn.Module):
    """
    Instead of using ISTA step, explicitly solves for non-negative LASSO solution. It turns out that for nonnegative
    LASSO, since D is orthogonal w.h.p. and stays so, the closed form LASSO solution is just ReLU of MLP on Z^{l + 1/2}
    summed with a fixed bias term lambda/2!
    TODO: try nonnegative vs unconstrained output
    TODO: instead of using lambd/2 as the bias term, maybe just include the bias in MLP lol
    """
    def __init__(self, dim, dropout=0., lambd=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.lambd: float = lambd

    def forward(self, x):
        return F.relu((self.lambd / 2.0) + F.linear(x, self.weight, bias=None))