# Copyright (c) Facebook, Inc. and its affiliates.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
Modified from detectron2(https://github.com/facebookresearch/detectron2)
"""

import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from accelerate import Accelerator
from typing import Union


accelerator: Union[Accelerator, None] = None
def lazy_init_accelerate(accel):
    global accelerator
    if accelerator is None:
        import weakref
        accelerator = weakref.proxy(accel)

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_world_size() -> int:
    return accelerator.num_processes


def get_rank() -> int:
    return accelerator.process_index


def get_local_rank() -> int:
    return accelerator.local_process_index


def is_main_process() -> bool:
    return accelerator.is_main_process

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    accelerator.wait_for_everyone()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sum_model_parameters(model):
    total_params = 0
    for param in model.parameters():
        total_params += torch.sum(param)
    return total_params


def convert_tensor_to_cuda(input_value):
    """Convert input tensors to cuda(non_blocking=True)"""
    if isinstance(input_value, torch.Tensor):
        return input_value.cuda(non_blocking=True)

    # convert tuple to list
    if isinstance(input_value, tuple):
        input_value = list(input_value)

    if isinstance(input_value, list):
        for i in range(len(input_value)):
            input_value[i] = convert_tensor_to_cuda(input_value[i])
        return input_value

    if isinstance(input_value, dict):
        for key in input_value.keys():
            input_value[key] = convert_tensor_to_cuda(input_value[key])
        return input_value

    raise NotImplementedError(f"Unsupported input type: {type(input_value)}")


def copy_codebase(save_path, exclude_dirs=None):
    """Copy codebase to save_path for future reference"""
    import shutil

    codebase_path = os.getcwd()
    save_path = os.path.join(save_path, "codebase")

    if exclude_dirs is None:
        exclude_dirs = ["__pycache__", "wandb", "out", "pretrained", "data", "clip-vit-base-patch16", "output"]

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for item in os.listdir(codebase_path):
        if item in exclude_dirs:
            continue
        s = os.path.join(codebase_path, item)
        d = os.path.join(save_path, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks=True,
                            ignore=shutil.ignore_patterns("*.pyc", "*.pth"), dirs_exist_ok=True)
        else:
            shutil.copy2(s, d)

