import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init

def coding_rate(
        ZT: torch.Tensor, # b x n x d
        eps: float = 0.01,
    ) -> torch.Tensor:
    n, d = ZT.shape[-2], ZT.shape[-1]
    sim = torch.matmul(ZT.transpose(-1, -2), ZT) if n > d else torch.matmul(ZT, ZT.transpose(-1, -2))
    return 0.5 * torch.logdet(torch.eye(min(d, n)) + sim * d / (n * eps ** 2))

def rate_reduction(
        ZT: torch.Tensor, # b x n x d
        UT: torch.Te,
        eps: float = 0.01,
    ) -> torch.Tensor:
    pass

def sparse_rate_reduction(Z):
    pass