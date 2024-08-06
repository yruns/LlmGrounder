"""
Misc Callbacks

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import glob
import os
import shutil
import sys
import time
from collections import OrderedDict

import torch
import torch.utils.data

from trim.utils.timer import Timer
from trim.utils.comm import is_main_process

from trim.callbacks.default import CallbackBase


class IterationTimer(CallbackBase):

    def __init__(self, warmup_iter=2):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def on_training_phase_start(self):
        self._start_time = time.perf_counter()
        self._remain_iter = self.trainer.max_epoch * len(self.trainer.train_loader)

    def on_training_epoch_start(self):
        self._iter_timer.reset()

    def on_training_setp_start(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def on_training_setp_end(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter -= 1
        remain_time = self._remain_iter * self.trainer.storage.history("batch_time").avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = "{:02d}:{:02d}:{:02d}".format(int(t_h), int(t_m), int(t_s))
        if "iter_info" in self.trainer.comm_info.keys():
            info = (
                "Data {data_time_val:.3f} ({data_time_avg:.3f}) "
                "Batch {batch_time_val:.3f} ({batch_time_avg:.3f}) "
                "Remain {remain_time} ".format(
                    data_time_val=self.trainer.storage.history("data_time").val,
                    data_time_avg=self.trainer.storage.history("data_time").avg,
                    batch_time_val=self.trainer.storage.history("batch_time").val,
                    batch_time_avg=self.trainer.storage.history("batch_time").avg,
                    remain_time=remain_time,
                )
            )
            self.trainer.comm_info["iter_info"] += info
        if self.trainer.comm_info["iter"] <= self._warmup_iter:
            self.trainer.storage.history("data_time").reset()
            self.trainer.storage.history("batch_time").reset()


class InformationWriter(CallbackBase):
    def __init__(self, log_interval=10):
        self.curr_iter = 0
        self.log_interval = log_interval
        self.model_output_keys = []

    def on_training_phase_start(self):
        self.trainer.comm_info["iter_info"] = ""
        self.curr_iter = self.trainer.start_epoch * len(self.trainer.train_loader)
        self.trainer.logger.info(self.trainer.args)

    def on_training_setp_start(self):
        self.curr_iter += 1
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=self.trainer.comm_info["iter"] + 1,
            max_iter=len(self.trainer.train_loader),
        )
        self.trainer.comm_info["iter_info"] += info

    def on_training_setp_end(self):
        current_iter = self.trainer.epoch * len(self.trainer.train_loader) + self.trainer.comm_info["iter"]

        # Anything you want to log in terminal and file
        if "terminal_log" in self.trainer.comm_info.keys():
            terminal_log = self.trainer.comm_info["terminal_log"]
            self.model_output_keys = terminal_log.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, terminal_log[key].item())
                self.trainer.wandb.log({
                    key: terminal_log[key],
                }, step=current_iter)

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.state_dict()["param_groups"][0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)

        # log in terminal and file
        if (self.curr_iter + 1) % self.log_interval == 0:
            self.trainer.logger.info(self.trainer.comm_info["iter_info"])

        # Anything you want to log in wandb
        if "wandb_log" in self.trainer.comm_info.keys():
            wandb_log = self.trainer.comm_info["wandb_log"]
            for key in wandb_log.keys():
                self.trainer.wandb.log({
                    key: wandb_log[key],
                }, step=current_iter)

        self.trainer.comm_info["iter_info"] = ""  # reset iter info

class CheckpointSaver(CallbackBase):
    """
    CheckpointSaver

    If you are using this callback, be sure to set `self.trainer.comm_info["current_metric_value"]` and
    `self.trainer.comm_info["current_metric_name"]` before executing this callback.
    It is recommended to set these values in the `Evaluator` callback.
    """

    def __init__(self, save_freq=None):
        self.save_freq = save_freq  # None or int, None indicate only save models last

    def on_training_epoch_end(self):
        is_best = False
        current_metric_value = self.trainer.comm_info["current_metric_value"]
        current_metric_name = self.trainer.comm_info["current_metric_name"]
        if current_metric_value > self.trainer.best_metric_value:
            self.trainer.best_metric_value = current_metric_value
            self.trainer.best_metric_epoch = self.trainer.epoch + 1
            is_best = True
            self.trainer.logger.info(
                "Best validation {} updated to: {:.4f}".format(
                    current_metric_name, current_metric_value
                )
            )
        self.trainer.logger.info(
            "Currently Best {}: {:.4f} at epoch {}".format(
                current_metric_name, self.trainer.best_metric_value, self.trainer.best_metric_epoch
            )
        )
        self.trainer.wandb.update({
            current_metric_name: self.trainer.best_metric_value,
            f"best_{current_metric_name}_epoch": self.trainer.best_metric_epoch
        })

        filename = os.path.join(
            self.trainer.save_path, "model", "model_last.pth"
        )
        self.trainer.logger.info("Saving checkpoint to: " + filename)
        if is_main_process():
            torch.save(
                {
                    "epoch": self.trainer.epoch + 1,
                    "state_dict": self.trainer.model.state_dict(),
                    "optimizer": self.trainer.optimizer.state_dict(),
                    "scheduler": self.trainer.scheduler.state_dict(),
                    "scaler": self.trainer.scaler.state_dict()
                    if self.trainer.scaler else None,
                    "best_metric_value": self.trainer.best_metric_value,
                    "best_metric_epoch": self.trainer.best_metric_epoch,
                },
                filename + ".tmp",
            )
            os.replace(filename + ".tmp", filename)
        if is_best and is_main_process():
            shutil.copyfile(
                filename,
                os.path.join(self.trainer.save_path, "model", "model_best.pth"),
            )
        if self.save_freq and (self.trainer.epoch + 1) % self.save_freq == 0 and is_main_process():
            shutil.copyfile(
                filename,
                os.path.join(
                    self.trainer.save_path,
                    "model",
                    f"epoch_{self.trainer.epoch + 1}.pth",
                ),
            )

class CheckpointLoader(CallbackBase):
    def __init__(self, keywords="", replacement=None, strict=True):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    def on_training_phase_start(self):
        self.trainer.logger.info("=> Loading checkpoint & weight ...")
        if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
            self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
            checkpoint = torch.load(
                self.trainer.cfg.weight,
                map_location=lambda storage, loc: storage.cuda(),
            )
            self.trainer.logger.info(
                f"Loading layer weights with keyword: {self.keywords}, "
                f"replace keyword with: {self.replacement}"
            )
            weight = OrderedDict(
                [
                    (key.replace(self.keywords, self.replacement), value)
                    for key, value in checkpoint["state_dict"].items()
                    if self.keywords in key
                ]
            )
            # weight = OrderedDict()
            # for k, v in checkpoint["state_dict"].items():
            #     if k.startswith('module.'):
            #         # remove module
            #         k = k[7:]  # module.xxx.xxx -> xxx.xxx
            #     else:
            #         # add module
            #         k = 'module.' + k  # xxx.xxx -> module.xxx.xxx
            #     weight[k] = v
            load_state_info = self.trainer.model.load_state_dict(
                weight, strict=self.strict
            )
            self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
            if self.trainer.cfg.resume:
                self.trainer.logger.info(
                    f"Resuming train at eval epoch: {checkpoint['epoch']}"
                )
                self.trainer.start_epoch = checkpoint["epoch"]
                self.trainer.best_metric_value = checkpoint["best_metric_value"]
                self.trainer.best_metric_epoch = checkpoint["epoch"]
                self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
                self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
                if self.trainer.cfg.enable_amp:
                    self.trainer.scaler.load_state_dict(checkpoint["scaler"])
        else:
            self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")