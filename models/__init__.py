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


class CRATE_CTRL_AE:
    def __init__(self, 
                 dim=None, depth=16, num_heads=8, dropout=0., step_size=1., 
                 image_size=32, patch_size=4, in_channels=3,
        ) -> None:
        super().__init__()

        # set up patch embedding
        assert image_size % patch_size == 0, "image dimensions must be compatible with patch size"
        dim = dim or image_size // patch_size
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size,in_chans=in_channels, embed_dim=dim)

        # build transformer backbone
        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])
        dim_head = dim // num_heads
        for _ in range(depth):
            encoder = CRATE_Transformer_Encode(dim=dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout, step_size=step_size)
            decoder = CRATE_Transformer_Decode(dim=dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout, step_size=step_size)
            decoder.norm_attn.weight, decoder.norm_attn.bias = encoder.norm_attn.weight, encoder.norm_attn.bias
            decoder.norm_mlp.weight, decoder.norm_mlp.bias = encoder.norm_mlp.weight, encoder.norm_mlp.bias
            decoder.attn.UT.weight, decoder.mlp.weight = encoder.attn.UT.weight, encoder.mlp.weight
            self.encoders.append(encoder)
            self.decoders.append(decoder)
        self.decoders = self.decoders[::-1]

        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, dim, bias=True)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, dim), requires_grad=False)
        self.initialize_weights()

    def initialize_weights(self):
        def 


    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        From https://github.com/Ma-Lab-Berkeley/CRATE/blob/main/model/crate_ae/crate_ae.py
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

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
    
    def encode(self, x, training=False):


    def decode(self, x)

    def forward(self, x):
