import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init

DEBUG = False

def logdet(
        sim: torch.Tensor  # b x ((d x d) | (n x n)) || b x h x ((d x d) | (n x n))
    ) -> torch.Tensor:
    """
    A faster and more numerically stable logdet for our purposes, referencing the CTRL paper.
    """
    if DEBUG:
      return 2 * torch.sum(torch.log(torch.diagonal(torch.linalg.cholesky(sim), dim1=-1, dim2=-2)), axis=-1)
    else:
      for test in sim.view(-1, sim.shape[-2], sim.shape[-1]):
        try:
          torch.linalg.cholesky(test)
        except:
          print('not pd:')
          print(test)
          print(torch.linalg.eigvals(test).real)
      return 2 * torch.sum(torch.log(torch.diagonal(torch.linalg.cholesky(sim), dim1=-1, dim2=-2)), axis=-1)

def coding_rate(
        ZT: torch.Tensor, # b x n x d | b x h x n x d
        eps: float = 0.5,
    ) -> torch.Tensor: # b | b x h
    """
    Calculates the coding rate (using a Gaussian codebook) of given set of tokens. When calculating the gramian,
    takes the transpose if the resulting matrix is smaller (for compactness and better stability when calculating
    determinant, since having zero valued eigenvalues can sometimes push them to be negative)
    """
    n, d = ZT.shape[-2], ZT.shape[-1]
    sim = torch.matmul(ZT.transpose(-1, -2), ZT) if n > d else torch.matmul(ZT, ZT.transpose(-1, -2))
    id = torch.eye(min(d, n)).to(sim.device)
    if DEBUG:
        names = ["sim"]
        tensors = [sim]
        for name, tens in zip(names, tensors):
            print(f"{name} info:")
            print(tens)
            print("shape:", tens.shape)
            print("number nan:", torch.isnan(tens).sum().item())
            print("")
        print("sim eigvals:", sorted(torch.linalg.eigvals(sim).real.flatten()))
        print('getting logdet(sim)')
        print(d, n, eps)
        print(f'scale: {d / (n * (eps ** 2))}')
        dummy = logdet(sim)
        print('getting logdet(sim * d / (n * (eps ** 2)))')
        dummy = 0.5 * logdet(sim * d / (n * (eps ** 2)))
    output = 0.5 * logdet(id + sim * d / (n * (eps ** 2)))
    if DEBUG:
        names = ["output"]
        tensors = [output]
        for name, tens in zip(names, tensors):
            print(f"{name} info:")
            print(tens)
            print("shape:", tens.shape)
            print("number nan:", torch.isnan(tens).sum().item())
            print("")
    return output

def rate_reduction(
        ZT: torch.Tensor, # b x n x d
        ZTU: torch.Tensor, # b x h x n x d
        eps: float = 0.5,
        normalize: bool = False,
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
    if DEBUG: print("coding rate info:")
    zt_cr = coding_rate(ZT, eps=eps)
    ztu_cr = torch.sum(coding_rate(ZTU, eps=eps), dim=1)
    if DEBUG:
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
        eps: float = 0.5,
        lambd: float = 0.1,
        normalize: bool = False,
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
    rr = rate_reduction(ZT, ZTU, eps=eps)
    lasso = ZT.abs().sum(dim=(-1, -2))
    if DEBUG:
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
        eps: float = 0.5,
        lambd_srr: float = 0.1,
        lambd_mse: float = 500,
        normalize: bool = False,
    ) -> torch.Tensor:
    """
    Initial formulation of the rate reduction closed loop transcription loss with a lasso
    term for sparsity. Dropped in favor of an alternating training scheme to allow for
    different learning rates of the encoding and decoding components of the network.
    """
    mse = torch.mean((ZT - ZT_hat) ** 2)
    if DEBUG: print("ZT, ZTU srr:")
    srr = sparse_rate_reduction(ZT, ZTU, lambd=lambd_srr, eps=eps, normalize=normalize)
    if DEBUG: print("ZT_hat, ZTU_hat srr:")
    srr_hat = sparse_rate_reduction(ZT_hat, ZTU_hat, lambd=lambd_srr, eps=eps, normalize=normalize)

    if DEBUG:
        names = ["mse", "srr", "srr_hat"]
        tensors = [mse, srr, srr_hat]
        for name, tens in zip(names, tensors):
            print(f"{name} info:")
            print("shape:", tens.shape)
            print("number nan:", torch.isnan(tens).sum().item())
            print("")
    # print(lambd_mse * mse.item(), srr.median().item(), srr.mean().item(), srr_hat.median().item(), srr_hat.mean().item())
    output = lambd_mse * mse - torch.mean(srr) - torch.mean(srr_hat)
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
    ) -> torch.Tensor:
    """"""
    raise NotImplementedError