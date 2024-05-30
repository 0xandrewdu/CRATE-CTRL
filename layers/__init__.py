import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init
from .decoder import *
from .encoder import *


class CRATE_CTRL_AE:
    def __init__(self, num_layers=16):
        super().__init__()
        self.encoders = []
        self.decoders = []
        for i in range(num_layers):
