"""
File: main.py
Date: 2024/8/19
Author: yruns
"""
import time
start = time.time()
from os import path
from typing import Optional

import math
import torch.optim
import torch.optim
import torch.utils.data
import torch.utils.data
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DummyOptim, DummyScheduler
from loguru import logger
from transformers import (
    get_scheduler, AutoTokenizer, PreTrainedTokenizerBase
)

import scripts.v1.grounder_reg as hparams

from datasets.referit3d import build_dataloader
from spatialreasoner.core.resoner import SpatialReasonerForCausalLM
from staticvars.const import REG_TOKEN

from trim.thirdparty.logging import WandbWrapper
from trim.thirdparty.logging import logger
from trim import TrainerBase
from trim.callbacks.misc import *
from trim.utils import comm


class Trainer(TrainerBase):

    def __init__(self, hparams, accelerator: Accelerator, logger, debug=False, callbacks=None):
        super().__init__()
        self.hparams = hparams
        self.accelerator: Accelerator = accelerator
        self.logger = logger
        self.max_epoch = hparams.num_train_epochs
        self.output_dir = hparams.output_dir
        self.callbacks = callbacks or []
        self.debug = debug

        self.tokenizer: Optional[PreTrainedTokenizerBase] = None
        self.compute_dtype = comm.convert_str_to_dtype(accelerator.mixed_precision)

        end_time = time.time()
        logger.info(f"### => Loading with {end_time - start} seconds")

    def configure_model(self):
        logger.info("### => Creating model ...")

        pretrained_model_path = str(path.join(self.hparams.pretrained_state_dir, self.hparams.llm_name))
        assert path.exists(pretrained_model_path), f"Pretrained model `{pretrained_model_path}` not exists"
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path,
            model_max_length=self.hparams.model_max_length,
        )
        self.tokenizer.pad_token = self.tokenizer.unk_token
        self.tokenizer.add_tokens([REG_TOKEN], special_tokens=True)
        self.hparams.tokenizer = self.tokenizer

        self.model = SpatialReasonerForCausalLM.from_pretrained(
            pretrained_model_path,
            torch_dtype=self.compute_dtype,
            attn_implementation=self.hparams.attn_implementation,
        )
        # Configure model tokens
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        ## Initialize vision modules
        self.model.model.initialize_vision_modules(self.hparams)
        self.model.config.tokenizer_padding_side = self.tokenizer.padding_side
        self.model.config.tokenizer_model_max_length = self.tokenizer.model_max_length
        self.model.config.compute_dtype = self.compute_dtype

        self.model.resize_token_embeddings(len(self.tokenizer))

        if self.hparams.freeze_backbone:
            self.model.model.requires_grad_(False)
        num_parameters = comm.count_parameters(self.model)
        logger.info(f"Number of learnable parameters: {num_parameters}")

    def on_training_phase_start(self):
        super().on_training_phase_start()
        self.model.reset_detector_precision(torch.float32)

    def configure_dataloader(self):
        logger.info("### => Creating dataloader...")

        self.train_loader = build_dataloader(self.hparams, split="train")
        self.val_loader = build_dataloader(self.hparams, split="val")

    def configure_optimizers(self):
        logger.info("### => Creating optimizer and scheduler...")

        optimizer_cls = (
            torch.optim.AdamW
            if self.accelerator.state.deepspeed_plugin is None
               or "optimizer" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
            else DummyOptim
        )
        self.optimizer = optimizer_cls(self.model.parameters(), lr=self.hparams.lr)

        # Scheduler and math around the number of training steps.
        num_update_steps_per_epoch_for_scheduler = math.ceil(
            len(self.train_loader) / self.accelerator.gradient_accumulation_steps)
        max_train_steps_for_scheduler = num_update_steps_per_epoch_for_scheduler * self.hparams.num_train_epochs
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_loader) / self.accelerator.gradient_accumulation_steps / self.accelerator.num_processes)
        self.total_train_steps = self.hparams.num_train_epochs * self.num_update_steps_per_epoch

        warmup_num_steps = int(self.hparams.warmup_ratio * max_train_steps_for_scheduler)
        if (
                self.accelerator.state.deepspeed_plugin is None
                or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            self.lr_scheduler = get_scheduler(
                name=self.hparams.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=warmup_num_steps,
                num_training_steps=max_train_steps_for_scheduler,
            )
        else:
            self.lr_scheduler = DummyScheduler(
                self.optimizer, total_num_steps=max_train_steps_for_scheduler,
                warmup_num_steps=warmup_num_steps
            )

    def configure_wandb(self):
        # When debugging, we don't need to log anything.
        self.wandb = WandbWrapper(
            project=self.hparams.log_project,
            name=self.hparams.log_tag,
            config={
                "log_tag": self.hparams.log_tag,
            },
            save_code=False,
            resume=False,
            file_prefix=os.path.join(self.output_dir, "codebase"),
            save_files=[__file__],
            debug=True
        )

    def training_setp(self, batch_data, batch_index):
        with self.accelerator.accumulate(self.model):
            batch_data = comm.convert_tensor_to_dtype(
                batch_data, self.accelerator.mixed_precision,
                ignore_keys="scene_data_dict"
            )

            output = self.model(**batch_data)
            loss = output.loss
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                reduced_loss = self.accelerator.reduce(loss)

                # Anything you want to log in terminal
                self.comm_info["terminal_log"] = {"loss": reduced_loss}
                # Anything you want to log in wandb
                self.comm_info["wandb_log"] = {"loss": reduced_loss}


def main(hparams):
    """Main function."""
    comm.seed_everything(hparams.seed)
    comm.copy_codebase(
        hparams.output_dir,
        exclude_dirs=["__pycache__", "wandb", "pretrained", "data", "clip-vit-base-patch16", "output", "tutorials"]
    )

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config="configs/zero_3_stage.json",
        )
    )

    trainer = Trainer(hparams, accelerator, logger, debug=True, callbacks=[
        Resumer(checkpoint=hparams.resume_from_checkpoint),
        IterationTimer(warmup_iter=1),
        InformationWriter(log_interval=hparams.log_interval),
        # Evaluator(),
        CheckpointSaver(save_freq=hparams.save_freq),
    ])
    trainer.fit()


if __name__ == "__main__":
    main(hparams)
