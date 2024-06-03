# Modified from https://github.com/facebookresearch/DiT/blob/main/train.py
# TODO: figure out reasonable way to preserve the scale of X_hat wrt X? Current worry is that due to the layernorms, 
# the outputted X_hat might be "like" X but on a different scale? Probably just normalize both for now, but a reconstruction
# error term might be useful for encoding general images--issue is that part of the point is that the distance metric for
# images is messy and hard to define well, whereas distances in the latent space are "nice" due to Gaussian mixture model

import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

