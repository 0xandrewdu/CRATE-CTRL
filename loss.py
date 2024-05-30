import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init

def coding_rate(
        ZT: torch.Tensor, # b x n x d | b x h x n x d
        eps: float = 0.01,
    ) -> torch.Tensor: # b | b x h
    n, d = ZT.shape[-2], ZT.shape[-1]
    sim = torch.matmul(ZT.transpose(-1, -2), ZT) if n > d else torch.matmul(ZT, ZT.transpose(-1, -2))
    return 0.5 * torch.logdet(torch.eye(min(d, n)) + sim * d / (n * eps ** 2))

def rate_reduction(
        ZT: torch.Tensor, # b x n x d
        ZTU: torch.Tensor, # b x h x n x d
        eps: float = 0.01,
    ) -> torch.Tensor:
    """
    Computes the rate reduction, defined as the coding rate of the tokens minus sum of the coding 
    rates of their projections onto each subspace (here we use the output of the transformer, which
    has the sparsification step after the attention application, so it calculates the RR for the
    subspaces rotated by the dictionary--should in theory be the same as just using the attention
    layer output, but matters when adding in a sparsity penalization term below)
    """
    return coding_rate(ZT, eps=eps) - torch.sum(coding_rate(ZTU, eps=eps), dim=1)

def sparse_rate_reduction(
        ZT: torch.Tensor, # b x n x d
        ZTU: torch.Tensor, # b x h x n x d
        eps: float = 0.01,
        lambd: float = 0.1
    ) -> torch.Tensor:
    """
    Adds a LASSO penalization term to the above rate reduction function
    """
    pass