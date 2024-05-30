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
from utils.loss import get_2d_sincos_pos_embed
from timm.models.vision_transformer import PatchEmbed


class CRATE_CTRL_AE:
    def __init__(self, 
                 depth=16, num_heads=8, dim_head=8, dropout=0., step_size=1., 
                 image_size=32, patch_size=4, in_channels=3, embed_dim=48
        ) -> None:
        super().__init__()


        # set up patch embedding
        assert image_size % patch_size == 0, "image dimensions must be compatible with patch size"
        dim = image_size // patch_size
        self.patch_embed = PatchEmbed(img_size=image_size, patch_size=patch_size,in_chans=in_channels, embed_dim=embed_dim)

        # build transformer backbone
        self.encoders = nn.ModuleList([])
        self.decoders = nn.ModuleList([])
        for i in range(depth):
            encoder = CRATE_Transformer_Encode(dim=dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout, step_size=step_size)
            decoder = CRATE_Transformer_Decode(dim=dim, num_heads=num_heads, dim_head=dim_head, dropout=dropout, step_size=step_size)
            decoder.norm_attn.weight, decoder.norm_attn.bias = encoder.norm_attn.weight, encoder.norm_attn.bias
            decoder.norm_mlp.weight, decoder.norm_mlp.bias = encoder.norm_mlp.weight, encoder.norm_mlp.bias
            decoder.attn.UT.weight, decoder.mlp.weight = encoder.attn.UT.weight, encoder.mlp.weight
            self.encoders.append(encoder)
            self.decoders.append(decoder)
        self.decoders = self.decoders[::-1]

    def patchify(self, x):
