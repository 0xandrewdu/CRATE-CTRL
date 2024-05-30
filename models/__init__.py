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


class CRATE_CTRL_AE:
    def __init__(self, depth=16, dim=32, num_heads=8, dim_head=8, dropout=0., step_size=1.):
        super().__init__()
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
