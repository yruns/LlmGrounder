"""
File: wrapper.py
Date: 2024/8/5
Author: yruns


"""
import argparse
import multiprocessing as mp
import os
import random
import sys

from functools import partial

import shutil
import torch
from os.path import join
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import trim.utils.comm as comm
from trim.training.collate_fn import point_collate_fn, collate_fn
from trim.training.default import worker_init_fn


def wrap_ddp_model(model, *, fp16_compression=False, **kwargs):
    """
    Create a DistributedDataParallel models if there are >1 processes.
    Args:
        model: a torch.nn.Module
        fp16_compression: add fp16 compression callbacks to the ddp object.
            See more at https://pytorch.org/docs/stable/ddp_comm_hooks.html#torch.distributed.algorithms.ddp_comm_hooks.default_hooks.fp16_compress_hook
        kwargs: other arguments of :module:`torch.nn.parallel.DistributedDataParallel`.
    """
    if comm.get_world_size() == 1:
        return model.cuda()
    # kwargs['find_unused_parameters'] = True
    if "device_ids" not in kwargs:
        kwargs["device_ids"] = [comm.get_rank()]
        if "output_device" not in kwargs:
            kwargs["output_device"] = [comm.get_rank()]
    ddp = DistributedDataParallel(model.cuda(), **kwargs)
    if fp16_compression:
        from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks

        ddp.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
    return ddp


def wrap_ddp_loader(args, dataset, collate_fn, batch_size=None, drop_last=True, shuffle=None, debug=False):
    if comm.get_world_size() > 1:
        sampler = DistributedSampler(dataset)
    else:
        sampler = None

    init_fn = (
        partial(
            worker_init_fn,
            num_workers=args.workers,
            rank=comm.get_rank(),
            seed=args.manual_seed,
        )
        if args.manual_seed is not None
        else None
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size if batch_size is not None else args.batch_size,
        shuffle=(sampler is None) if shuffle is None else shuffle,
        num_workers=args.workers if not debug else 1,
        sampler=sampler,
        collate_fn=collate_fn,
        pin_memory=True,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        persistent_workers=True,
    )

    return loader
