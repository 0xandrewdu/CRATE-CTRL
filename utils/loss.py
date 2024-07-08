import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init

def logdet(
        sim: torch.Tensor  # b x ((d x d) | (n x n)) || b x h x ((d x d) | (n x n))
    ) -> torch.Tensor:
    """
    A faster and more numerically stable logdet for our purposes, referencing the CTRL paper.
    """
    return 2 * torch.sum(torch.log(torch.diag(torch.linalg.cholesky(sim))))

def coding_rate(
        ZT: torch.Tensor, # b x n x d | b x h x n x d
        eps: float = 0.5,
        # logdet_eps: float = 10 ** -6, # to avoid negative eigenvals due to numerical precision issues in logdet
        debug: bool = False,
    ) -> torch.Tensor: # b | b x h
    """
    Calculates the coding rate (using a Gaussian codebook) of given set of tokens. When calculating the gramian,
    takes the transpose if the resulting matrix is smaller (for compactness and better stability when calculating
    determinant, since having zero valued eigenvalues can sometimes push them to be negative)
    """
    n, d = ZT.shape[-2], ZT.shape[-1]
    sim = torch.matmul(ZT.transpose(-1, -2), ZT) if n > d else torch.matmul(ZT, ZT.transpose(-1, -2))
    if debug: print("sim eigvals:", torch.linalg.eigvals(sim))
    # id = torch.eye(min(d, n)).to(sim.device)
    # sim = sim + id * logdet_eps
    output = 0.5 * logdet(id + sim * d / (n * (eps ** 2)))
    if debug:
        names = ["sim", "id", "output"]
        tensors = [sim, id, output]
        for name, tens in zip(names, tensors):
            print(f"{name} info:")
            print(tens)
            print("shape:", tens.shape)
            print("number nan:", torch.isnan(tens).sum().item())
            print("")
        print("sim eigvals:", torch.linalg.eigvals(sim))
    return output

def rate_reduction(
        ZT: torch.Tensor, # b x n x d
        ZTU: torch.Tensor, # b x h x n x d
        eps: float = 0.01,
        normalize: bool = False,
        debug: bool = False,
    ) -> torch.Tensor:
    """
    Computes the rate reduction, defined as the coding rate of the tokens minus sum of the coding 
    rates of their projections onto each subspace (here we use the output of the transformer, which
    has the sparsification step after the attention application, so it calculates the RR for the
    subspaces rotated by the dictionary--should in theory be the same as just using the attention
    layer output, but matters when adding in a sparsity penalization term below)
    """
    if normalize:
        ZT, ZTU = F.layer_norm(ZT, ZT.shape[-2:]), F.layer_norm(ZTU, ZTU.shape[-2:])
    if debug: print("coding rate info:")
    zt_cr = coding_rate(ZT, eps=eps, debug=debug)
    ztu_cr = torch.sum(coding_rate(ZTU, eps=eps), dim=1)
    if debug:
        names = ["zt_cr", "ztu_cr"]
        tensors = [zt_cr, ztu_cr]
        for name, tens in zip(names, tensors):
            print(f"{name} info:")
            print("shape:", tens.shape)
            print("number nan:", torch.isnan(tens).sum().item())
            print("")
    output = zt_cr - ztu_cr
    return output

def sparse_rate_reduction(
        ZT: torch.Tensor, # b x n x d
        ZTU: torch.Tensor, # b x h x n x d
        eps: float = 0.01,
        lambd: float = 0.1,
        normalize: bool = False,
        debug: bool = False,
    ) -> torch.Tensor:
    """
    Adds a LASSO penalization term to the above rate reduction function
    TODO: Check that there isn't weird stuff going on with the scale of ZT, ZTU and the l1 penalty. I
    feel like adding a layernorm with no learnable scale / shift could be useful here. There might be some
    weird stuff going on in the RR term since epsilon isn't scale invariant either?
    TODO: For labelled data, give an option to do (sparse) rate reduction classwise instead of by subspace
    projections. Could be interesting if the unlabelled autoencoder naturally learns to project different
    classes onto subspaces? Try training with num_heads = num_classes on CIFAR10.
    """
    if normalize:
        ZT, ZTU = F.layer_norm(ZT, ZT.shape[-2:]), F.layer_norm(ZTU, ZTU.shape[-2:])
    rr = rate_reduction(ZT, ZTU, eps=eps, debug=debug)
    lasso = ZT.abs().sum(dim=(-1, -2))
    if debug:
        names = ["rr", "lasso"]
        tensors = [rr, lasso]
        for name, tens in zip(names, tensors):
            print(f"{name} info:")
            print("shape:", tens.shape)
            print("number nan:", torch.isnan(tens).sum().item())
            print("")
    output = lambd * lasso - rr
    return output

def ctrl_objective(
        ZT: torch.Tensor, # b x n x d
        ZTU: torch.Tensor, # b x h x n x d
        ZT_hat: torch.Tensor, # b x n x d
        ZTU_hat: torch.Tensor, # b x h x n x d
        eps: float = 0.01,
        lambd_srr: float = 0.1,
        lambd_mse: float = 0.5,
        normalize: bool = False,
        debug: bool = False,
    ) -> torch.Tensor:
    """
    Initial formulation of the rate reduction closed loop transcription loss with a lasso
    term for sparsity. Dropped in favor of an alternating training scheme to allow for
    different learning rates of the encoding and decoding components of the network.
    """
    mse = torch.mean((ZT - ZT_hat) ** 2)
    if debug: print("ZT, ZTU srr:")
    srr = sparse_rate_reduction(ZT, ZTU, lambd=lambd_srr, eps=eps, normalize=normalize, debug=debug)
    if debug: print("ZT_hat, ZTU_hat srr:")
    srr_hat = sparse_rate_reduction(ZT_hat, ZTU_hat, lambd=lambd_srr, eps=eps, normalize=normalize, debug=debug)

    if debug:
        names = ["mse", "srr", "srr_hat"]
        tensors = [mse, srr, srr_hat]
        for name, tens in zip(names, tensors):
            print(f"{name} info:")
            print("shape:", tens.shape)
            print("number nan:", torch.isnan(tens).sum().item())
            print("")

    output = lambd_mse * mse - srr - srr_hat
    return output

def ctrl_objective_encoder(
        ZT: torch.Tensor, # b x n x d
        ZTU: torch.Tensor, # b x h x n x d
        ZT_hat: torch.Tensor, # b x n x d
        ZTU_hat: torch.Tensor, # b x h x n x d
        eps: float = 0.01,
        lambd_srr: float = 0.1,
        lambd_mse: float = 0.5,
        normalize: bool = False,
        debug: bool = False,
    ) -> torch.Tensor:
    """"""
    raise NotImplementedError

def ctrl_objective_decoder(
        ZT: torch.Tensor, # b x n x d
        ZTU: torch.Tensor, # b x h x n x d
        ZT_hat: torch.Tensor, # b x n x d
        ZTU_hat: torch.Tensor, # b x h x n x d
        eps: float = 0.01,
        lambd_srr: float = 0.1,
        lambd_mse: float = 0.5,
        normalize: bool = False,
        debug: bool = False,
    ) -> torch.Tensor:
    """"""
    raise NotImplementedError