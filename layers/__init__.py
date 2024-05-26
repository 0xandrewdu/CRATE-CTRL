import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import torch.nn.init as init
from .backward import *
from .forward import *


class CRATE_CTRL_AE:
    def __init__(self):
        return