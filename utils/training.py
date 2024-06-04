import argparse
import os
from ..models import model_configs

"""
Utils for train.py

Includes command line argument parser, helpers for resuming training
"""

def get_args_parser():  
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(model_configs.keys()), default="CIFAR10-B")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=50_000)
    parser.add_argument("--ckpt-path", type=str, default=None)

    parser.add_argument('--lr', '--learning-rate', default=0.0004, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=0.1, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')

    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')

    parser.add_argument('--optimizer', default="AdamW", type=str,
                        help='Optimizer to Use.')
    return parser

def clean_dirs():
    """
    For cleaning out folders that were created during buggy trainning runs
    """
    raise NotImplementedError

def create_dirs():
    """
    Utility function to create directories for storing training checkpoints and logs
    """


def training_setup(args, **kwargs):
    """
    Fetches model from checkpoints if provided and valid (file exists and same model), otherwise creates new model and optimizer
    """

    model_name, ckpt_path = args.model, args.ckpt_path
    if not model_name in model_configs.keys():
        print("Requested model name does not exist.")
        raise NotImplementedError
    load_ckpt = bool(ckpt_path)
    if load_ckpt:
        if not os.path.isfile(ckpt_path):
            print("Checkpoint file does not exist, creating new model.")
            load_ckpt = False
        else:
            ckpt_model_name, epoch = parse_ckpt_name(ckpt_path)
            if ckpt_model_name != model_name:
                print("Checkpoint model type does not match requested model, creating new model.")
                load_ckpt = False
            else:
                print(f"Found valid checkpoint at {ckpt_path}, loading {model_name} weights from epoch {epoch}.")

    model = model_configs[model_name](**kwargs)
    if load_ckpt:
        
    else:
        print(f"Created new {model_name}.")
    return model, optim, scheduler
        

def parse_ckpt_name(ckpt_path):
    ckpt_filename = ckpt_path.split('/')[-1]
    tokens = ckpt_filename.split('_') # NOTE: be careful with using underscores for spaces in filename, use dashes instead
    ckpt_model_name, epoch = tokens[0], tokens[1] # TODO: add hyperparams in checkpoint names
    return ckpt_model_name, epoch