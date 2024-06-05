import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init
from layers.decoder import *
from layers.encoder import *
from utils.loss import sparse_rate_reduction
from timm.models.vision_transformer import PatchEmbed
from utils.pos_embed import get_2d_sincos_pos_embed


class CRATE_CTRL_AE(nn.Module):
    def __init__(self, 
                 dim=None, depth=16, num_heads=8, dim_head=None, dropout=0., step_size=1., 
                 image_size=32, patch_size=4, in_channels=3, output_norm=nn.LayerNorm,
                 output_mean=0, output_std=1
        ) -> None:
        super().__init__()

        # set up patch embedding
        assert image_size % patch_size == 0, "image dimensions must be compatible with patch size"
        dim = dim or image_size // patch_size
        dim_head = dim_head or dim // num_heads
        embed_dim = dim or (patch_size ** 2)

        if dim_head * num_heads > dim:
            print('WARNING: dim_head * num_heads > dim. Subspaces will not be orthogonal.')

        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size,in_chans=in_channels, embed_dim=embed_dim)

        # build transformer backbone
        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])

        for _ in range(depth):
            # NOTE: step_size is not true step size, see CRATE_Transformer forward()
            encoder = CRATE_Transformer_Encode(dim=dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout, step_size=step_size)
            decoder = CRATE_Transformer_Decode(dim=dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout, step_size=step_size)
            decoder.norm_attn.weight, decoder.norm_attn.bias = encoder.norm_attn.weight, encoder.norm_attn.bias
            decoder.norm_mlp.weight, decoder.norm_mlp.bias = encoder.norm_mlp.weight, encoder.norm_mlp.bias
            decoder.attn.UT.weight, decoder.mlp.weight = encoder.attn.UT.weight, encoder.mlp.weight
            self.encoders.append(encoder)
            self.decoders.append(decoder)
        self.decoders = self.decoders[::-1]
        self.norm = output_norm(dim)

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, dim), requires_grad=False)
        self.initialize_weights()

    def initialize_weights(self):
        """
        Based off of https://github.com/facebookresearch/DiT/blob/main/models.py
        """
        def _init_helper(module: nn.Module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_init_helper)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.patch_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.patch_embed.proj.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        From https://github.com/Ma-Lab-Berkeley/CRATE/blob/main/model/crate_ae/crate_ae.py
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs
    
    def encode(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for block in self.encoders[:-1]:
            x = block(x)
        Z, ZTU = self.encoders[-1](x, return_proj=True)
        return Z, ZTU

    def decode(self, x):
        for block in self.decoders:
            x = block(x)
        x = self.norm(x)
        x = self.unpatchify(x)
        return x

    def forward(self, x):
        Z, ZTU = self.encode(x)
        x_hat = self.decode(Z)
        Z_hat, ZTU_hat = self.encode(x_hat)
        return x_hat, Z, ZTU, Z_hat, ZTU_hat
    

#################################################################################
#                                 Model Configs                                 #
#################################################################################

def CTRL_CIFAR10_Base(**kwargs):
    model = CRATE_CTRL_AE(depth=12, num_heads=10, image_size=32, patch_size=8, **kwargs)
    return model

model_configs = {
    'CIFAR10-B': CTRL_CIFAR10_Base,
}