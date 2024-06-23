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
        super(Attention_Encode, self).__init__()
        self.num_heads = num_heads
        self.input_dim = dim
        self.dim_head = dim_head
        self.dropout = nn.Dropout(dropout)
        self.I = nn.Parameter(torch.eye(self.input_dim), requires_grad=False)
        self.UT = nn.Linear(dim, num_heads * dim_head, bias=False) # note that nn.Linear right multiplies by W^T
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
    
class MLP_Encode_Old(nn.Module):
    """
    Instead of using ISTA step, explicitly solves for non-negative LASSO solution. It turns out that for nonnegative
    LASSO, since D is orthogonal w.h.p. and stays so, the closed form LASSO solution is just ReLU of MLP on Z^{l + 1/2}
    summed with a fixed bias term lambda/2
    TODO: instead of using lambd/2 as the bias term, maybe just include the bias in MLP lol
    """
    def __init__(self, dim, dropout=0., lambd=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim))
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.lambd: float = lambd

    def forward(self, Z_half):
        return F.relu((self.lambd / 2.0) + F.linear(Z_half, self.weight, bias=None))
    
class MLP_Encode(nn.Module):
    """
    Reverts to solution without assumption of orthogonality, but removes non-negativity from objective to make
    sampling from latent support negatives
    """
    def __init__(self, dim, dropout=0., step_size=0.5, lambd=0.5):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(dim, dim)) # self.D^T
        with torch.no_grad():
            init.kaiming_uniform_(self.weight)
        self.lambd: float = lambd
        self.step_size = step_size

    def forward(self, x, return_proj=False):
        Z_halfDT = F.linear(x, self.weight, bias=None)
        Z_halfDTD = F.linear(Z_halfDT, self.weight.T, bias=None)
        Z_halfD = F.linear(x, self.weight.T, bias=None)
        grad_step = x + self.step_size * (Z_halfD - Z_halfDTD)

        nonneg = True
        if nonneg:
            output = F.relu(torch.abs(grad_step) - self.step_size * self.lambd)
        else:
            output = F.relu(torch.abs(grad_step) - self.step_size * self.lambd) * torch.sign(grad_step)
        return (output, Z_halfDTD) if return_proj else output
    
class CRATE_Transformer_Encode(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head=8, dropout=0., mlp_step_size=0.5, lasso_lambd=0.5, step_size=1.):
        super().__init__()
        self.norm_attn, self.norm_mlp = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention_Encode(dim=dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout)
        self.mlp = MLP_Encode(dim=dim, dropout=dropout, step_size=mlp_step_size, lambd=lasso_lambd)
        self.step_size = step_size
    
    def forward(self, x, return_proj=False):
        """
        If return_proj is on (only for last layer in training), also returns the transformation of the projected
        tokens by calculating U^T D Z^{l+1} and returning as a second output
        """
        if return_proj:
            z = self.norm_attn(x)
            mssa_out = self.attn(z)
            z_half = (z + self.step_size * mssa_out) / (1 + self.step_size)
            z_half_norm = self.norm_mlp(z_half)
            z_out, z_halfddt = self.mlp(z_half_norm, return_proj=True)
            z_proj = rearrange(self.attn.UT(self.norm_attn(z_halfddt)), 'b n (h d) -> b h n d', h=self.attn.num_heads)
            return z_out, z_proj
        else:
            z = self.norm_attn(x)
            z_half = (z + self.step_size * self.attn(z)) / (1 + self.step_size)
            z_half_norm = self.norm_mlp(z_half)
            z_out = self.mlp(z_half_norm)
            return z_out