import argparse
import logging
import datetime
import os
import math
import torch
from lion_pytorch import Lion
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from models import model_configs

"""
Utils for train.py

Includes command line argument parser, helpers for resuming training
"""

def get_args_parser():  
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(model_configs.keys()), default="CIFAR10-B")
    parser.add_argument("--image-size", type=int, choices=[32, 256, 512], default=32)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0004)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--wd', '--weight-decay', type=float, default=0.1)
    parser.add_argument('--optim', type=str, default="AdamW")
    parser.add_argument('--lambd_srr', type=float, default=0.1)
    parser.add_argument('--lambd_mse', type=float, default=0.5)
    return parser

def clean_dirs():
    """
    Clean out folders that were created during buggy training runs
    """
    raise NotImplementedError

def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    From https://github.com/facebookresearch/DiT/blob/main/train.py
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger

def create_dirs(results_dir, model_name):
    """
    Utility function to create directories for storing training checkpoints and logs

    Made with https://github.com/facebookresearch/DiT/blob/main/train.py as reference
    """
    # pseudocode: create results directory if dne, then create {model-name}_{date-time} directory, then
    # create logs and ckpts subdirectories, return these two paths
    rank = dist.get_rank()
    if rank == 0:
        os.makedirs(results_dir, exist_ok=True)
        model_string_name = model_name.replace("/", "-")  # e.g., DiT-XL/2 --> DiT-XL-2 (for naming folders) // this shouldn't trigger btw
        date_time_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        experiment_dir = f"{results_dir}/{model_string_name}_{date_time_string}"  # Create an experiment folder
        checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")
    else:
        logger = create_logger(None)
        checkpoint_dir = None
    return logger, checkpoint_dir


def training_setup(args, **kwargs):
    """
    Fetches model from checkpoints if provided and valid (file exists and same model), otherwise creates new model and optimizer
    
    Made with https://github.com/Ma-Lab-Berkeley/CRATE/blob/main/main.py and 
    https://github.com/facebookresearch/DiT/blob/main/train.py as reference
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    model_name, ckpt_path = args.model, args.ckpt_path
    
    if not model_name in model_configs.keys():
        print("Requested model name does not exist.")
        raise NotImplementedError
    
    load_ckpt = bool(ckpt_path)
    epoch = 0

    if load_ckpt:
        if not os.path.isfile(ckpt_path):
            print("Checkpoint file does not exist, creating new model.")
            load_ckpt = False
        else:
            ckpt = torch.load(ckpt_path)
            ckpt_model_name, epoch = ckpt['model_name'], ckpt['epoch']
            if ckpt_model_name != model_name:
                print("Checkpoint model type does not match requested model, creating new model.")
                load_ckpt = False
            else:
                print(f"Found valid checkpoint at {ckpt_path}, loading {model_name} weights from epoch {epoch}.")
                args.weight_decay = ckpt['args'].weight_decay
                args.lr = ckpt['args'].lr
                args.lambd_mse = ckpt['args'].lambd_mse
                args.lambd_srr = ckpt['args'].lambd_srr

    model = model_configs[model_name](**kwargs)
    model = DDP(model).cuda()

    if args.optimizer == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, 
                                    betas=(0.9, 0.999), 
                                    weight_decay=args.weight_decay)                      
    elif args.optimizer == "Lion":
        optimizer = Lion(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError
    
    warmup_steps = 20
    lr_func = lambda step: min((step + 1) / (warmup_steps + 1e-8), 
                               0.5 * (math.cos(step / args.epochs * math.pi) + 1))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func, verbose=True)

    if load_ckpt:
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        print(f"Created new {model_name}.")
    return model, optimizer, scheduler, epoch