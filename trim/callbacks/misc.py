"""
Misc Callbacks

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import os
import shutil
import time

from trim.callbacks.default import CallbackBase
from trim.utils.timer import Timer


class IterationTimer(CallbackBase):

    def __init__(self, warmup_iter=2):
        self._warmup_iter = warmup_iter
        self._start_time = time.perf_counter()
        self._iter_timer = Timer()
        self._remain_iter = 0

    def on_training_phase_start(self):
        self._start_time = time.perf_counter()
        self._remain_iter = self.trainer.total_train_steps - self.trainer.completed_steps

    def on_training_epoch_start(self):
        self._iter_timer.reset()

    def on_training_step_start(self):
        data_time = self._iter_timer.seconds()
        self.trainer.storage.put_scalar("data_time", data_time)

    def on_training_step_end(self):
        batch_time = self._iter_timer.seconds()
        self._iter_timer.reset()
        self.trainer.storage.put_scalar("batch_time", batch_time)
        self._remain_iter = self.trainer.total_train_steps - self.trainer.completed_steps
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
        # self.trainer.logger.info(self.trainer.hparams)

    def on_training_step_start(self):
        info = "Train: [{epoch}/{max_epoch}][{iter}/{max_iter}] ".format(
            epoch=self.trainer.epoch + 1,
            max_epoch=self.trainer.max_epoch,
            iter=(self.trainer.completed_steps % self.trainer.num_update_steps_per_epoch) + 1,
            max_iter=self.trainer.num_update_steps_per_epoch,
        )
        self.trainer.comm_info["iter_info"] += info

    def on_training_step_end(self):
        self.trainer.completed_steps += 1
        current_iter = self.trainer.completed_steps

        # Anything you want to log in terminal and file
        if "terminal_log" in self.trainer.comm_info.keys():
            terminal_log = self.trainer.comm_info["terminal_log"]
            self.model_output_keys = terminal_log.keys()
            for key in self.model_output_keys:
                self.trainer.storage.put_scalar(key, terminal_log[key].item())

        for key in self.model_output_keys:
            self.trainer.comm_info["iter_info"] += "{key}: {value:.4f} ".format(
                key=key, value=self.trainer.storage.history(key).val
            )
        lr = self.trainer.optimizer.param_groups[0]["lr"]
        self.trainer.comm_info["iter_info"] += "Lr: {lr:.5f}".format(lr=lr)

        # log in terminal and file
        if (current_iter + 1) % self.log_interval == 0:
            self.trainer.logger.info(self.trainer.comm_info["iter_info"])

        # Anything you want to log in wandb
        if "wandb_log" in self.trainer.comm_info.keys():
            wandb_log = self.trainer.comm_info["wandb_log"]
            for key in wandb_log.keys():
                self.trainer.wandb.log({
                    key: wandb_log[key],
                    "epoch": self.trainer.epoch,
                    "step": current_iter
                })

        self.trainer.comm_info["iter_info"] = ""  # reset iter info


class CheckpointSaver(CallbackBase):
    """
    CheckpointSaver

    If you are using this callback, be sure to set `self.trainer.comm_info["current_metric_value"]` and
    `self.trainer.comm_info["current_metric_name"]` before executing this callback.
    It is recommended to set these values in the `Evaluator` callback.
    """

    def __init__(self, save_freq, save_last_only=False):
        self.save_freq = save_freq
        self.save_lastest_only = save_last_only
        self.checkpointing_steps = None
        self.last_checkpoint = None

    def on_training_phase_start(self):
        if isinstance(self.save_freq, int):
            # step based
            self.checkpointing_steps = self.save_freq
        elif isinstance(self.save_freq, str) and self.save_freq == "epoch":
            # epoch based
            self.checkpointing_steps = self.trainer.num_update_steps_per_epoch

    def on_training_step_end(self):
        if (
                hasattr(self, "checkpointing_steps")
                and self.trainer.completed_steps % self.checkpointing_steps == 0
        ):
            self.save_checkpoint()

    def save_checkpoint(self):
        output_dir = "epoch_{epoch}".format(epoch=self.trainer.epoch + 1) if self.save_freq == "epoch" else \
            "step_{step}".format(step=self.trainer.completed_steps)
        self.trainer.logger.info("=> Saving checkpoint to: " + output_dir)
        output_dir = os.path.join(self.trainer.output_dir, "checkpoints", output_dir)
        self.accelerator.save_state(output_dir)

        import torch
        from ..utils import comm
        self.trainer.logger.info(f"### Model weight: {comm.sum_model_parameters(self.trainer.model)}")
        # self.trainer.logger.info(f"### Optimizer weight: {self.trainer.optimizer.param_groups}")
        self.trainer.logger.info(f"### Optimizer weight: {torch.sum(list(self.trainer.optimizer.state.keys())[0])}")

        if self.save_lastest_only and self.last_checkpoint is not None and self.accelerator.is_main_process:
            shutil.rmtree(self.last_checkpoint)
            self.last_checkpoint = output_dir


class Resumer(CallbackBase):

    def __init__(self, checkpoint=None):
        self.checkpoint = checkpoint
        self.stored_train_loader = None

    def on_training_phase_start(self):
        if self.checkpoint is not None:
            self.resume()
        else:
            self.trainer.logger.info("=> No checkpoint given, training from scratch")

    def resume(self):
        if not os.path.exists(self.checkpoint):
            raise FileNotFoundError(f"=> No checkpoint found at: {self.checkpoint}")
        self.trainer.logger.info(f"=> Resuming from checkpoint: {self.checkpoint}")
        self.accelerator.load_state(self.checkpoint)

        self.accelerator.wait_for_everyone()

        import torch
        from ..utils import comm
        self.trainer.logger.info(f"### Model weight: {comm.sum_model_parameters(self.trainer.model)}")
        self.trainer.logger.info(f"### Optimizer weight: {torch.sum(list(self.trainer.optimizer.state.keys())[0])}")

        path = os.path.basename(self.checkpoint)
        training_difference = os.path.splitext(path)[0]

        resume_step = 0
        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", ""))
            completed_steps = starting_epoch * self.trainer.num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", ""))
            completed_steps = resume_step
            starting_epoch = resume_step // self.trainer.num_update_steps_per_epoch
            resume_step -= starting_epoch * self.trainer.num_update_steps_per_epoch

        self.trainer.resume_step = resume_step
        self.trainer.completed_steps = completed_steps
        self.trainer.start_epoch = starting_epoch

        # Although I don't know why accelete does not restore the state of dataloader,
        # the following code can solve it
        # if self.accelerator.num_processes > 1:
        #     from tqdm import tqdm
        #     for i in range(starting_epoch):
        #         self.trainer.logger.info(f"=> Skip epoch: {i + 1}")
        #         for _ in tqdm(self.trainer.train_loader):
        #             pass
                    # self.trainer.logger.info(f"data: {torch.mean(data)}")

        # store the train_loader
        self.stored_train_loader = self.trainer.train_loader
        self.trainer.train_loader = \
            self.accelerator.skip_first_batches(self.trainer.train_loader,
                                                resume_step * self.accelerator.gradient_accumulation_steps)

    def on_training_epoch_end(self):
        # We need to reset the train_loader in next epoch
        if self.stored_train_loader is not None:
            self.trainer.train_loader = self.stored_train_loader
            self.stored_train_loader = None


class CheckpointLoader(CallbackBase):
    def __init__(self, keywords="", replacement=None, strict=True):
        self.keywords = keywords
        self.replacement = replacement if replacement is not None else keywords
        self.strict = strict

    # def on_training_phase_start(self):
    #     self.trainer.logger.info("=> Loading checkpoint & weight ...")
    #     if self.trainer.cfg.weight and os.path.isfile(self.trainer.cfg.weight):
    #         self.trainer.logger.info(f"Loading weight at: {self.trainer.cfg.weight}")
    #         checkpoint = torch.load(
    #             self.trainer.cfg.weight,
    #             map_location=lambda storage, loc: storage.cuda(),
    #         )
    #         self.trainer.logger.info(
    #             f"Loading layer weights with keyword: {self.keywords}, "
    #             f"replace keyword with: {self.replacement}"
    #         )
    #         weight = OrderedDict(
    #             [
    #                 (key.replace(self.keywords, self.replacement), value)
    #                 for key, value in checkpoint["state_dict"].items()
    #                 if self.keywords in key
    #             ]
    #         )
    #         # weight = OrderedDict()
    #         # for k, v in checkpoint["state_dict"].items():
    #         #     if k.startswith('module.'):
    #         #         # remove module
    #         #         k = k[7:]  # module.xxx.xxx -> xxx.xxx
    #         #     else:
    #         #         # add module
    #         #         k = 'module.' + k  # xxx.xxx -> module.xxx.xxx
    #         #     weight[k] = v
    #         load_state_info = self.trainer.model.load_state_dict(
    #             weight, strict=self.strict
    #         )
    #         self.trainer.logger.info(f"Missing keys: {load_state_info[0]}")
    #         if self.trainer.cfg.resume:
    #             self.trainer.logger.info(
    #                 f"Resuming train at eval epoch: {checkpoint['epoch']}"
    #             )
    #             self.trainer.start_epoch = checkpoint["epoch"]
    #             self.trainer.best_metric_value = checkpoint["best_metric_value"]
    #             self.trainer.best_metric_epoch = checkpoint["epoch"]
    #             self.trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    #             self.trainer.scheduler.load_state_dict(checkpoint["scheduler"])
    #             if self.trainer.cfg.enable_amp:
    #                 self.trainer.scaler.load_state_dict(checkpoint["scaler"])
    #     else:
    #         self.trainer.logger.info(f"No weight found at: {self.trainer.cfg.weight}")
