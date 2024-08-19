"""
Default training/testing logic
modified from Pointcept(https://github.com/Pointcept/Pointcept)

Please cite our work if the code is helpful to you.
"""

import multiprocessing as mp
import os
import shutil
from os.path import join

import torch

import trim.utils.comm as comm


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Worker init func for dataloader.

    The seed of each worker equals to num_worker * rank + worker_id + user_seed

    Args:
        worker_id (int): Worker id.
        num_workers (int): Number of workers.
        rank (int): The rank of current process.
        seed (int): The random seed to use.
    """

    worker_seed = num_workers * rank + worker_id + seed
    comm.seed_everything(worker_seed)


def default_setup(cfg):
    # scalar by world size
    world_size = comm.get_world_size()
    cfg.num_worker = cfg.num_worker if cfg.num_worker is not None else mp.cpu_count()
    cfg.num_worker_per_gpu = cfg.num_worker // world_size
    assert cfg.batch_size % world_size == 0
    assert cfg.batch_size_val is None or cfg.batch_size_val % world_size == 0
    assert cfg.batch_size_test is None or cfg.batch_size_test % world_size == 0
    cfg.batch_size_per_gpu = cfg.batch_size // world_size
    cfg.batch_size_val_per_gpu = (
        cfg.batch_size_val // world_size if cfg.batch_size_val is not None else 1
    )
    # settle random seed
    rank = comm.get_rank()
    seed = None if cfg.seed is None else cfg.seed * cfg.num_worker_per_gpu + rank
    comm.seed_everything(seed)
    return cfg


def save_checkpoint(state, is_best, save_path, filename='model_last.pth.tar'):
    filename = join(save_path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, join(save_path, 'model_best.pth.tar'))


def save_checkpoint_epoch(state, save_path, epoch):
    filename = join(save_path, f'model_epoch_{epoch}.pth.tar')
    torch.save(state, filename)


def load_state_dict(state_dict, model, logger, strict=True):
    try:
        load_state_info = model.load_state_dict(state_dict, strict=strict)
    except Exception:
        # The model was trained in a parallel manner, so need to be loaded differently
        from collections import OrderedDict
        weight = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                # remove module
                k = k[7:]  # module.xxx.xxx -> xxx.xxx
            else:
                # add module
                k = 'module.' + k  # xxx.xxx -> module.xxx.xxx
            weight[k] = v
        load_state_info = model.load_state_dict(weight, strict=strict)
    logger.info(f"Missing keys: {load_state_info[0]}")

    return model


def resume(weight, model, optimizer, scheduler, scaler, logger, strict=True):
    assert os.path.exists(weight), f"{weight} does not exist."

    logger.info("=> Loading checkpoint & weight at: {weight}")
    checkpoint = torch.load(
        weight,
        map_location=lambda storage, loc: storage.cuda(),
    )

    model = load_state_dict(checkpoint["state_dict"], model, logger, strict)
    logger.info(
        f"Resuming train at eval epoch: {checkpoint['epoch']}"
    )
    start_epoch = checkpoint["epoch"]
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])

    return start_epoch, model, optimizer, scheduler, scaler
