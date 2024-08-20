"""
File: main.py
Date: 2024/8/19
Author: yruns
"""
from os import path
from typing import Optional

import math
import torch.optim
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DummyOptim, DummyScheduler
from transformers import (
    get_scheduler, AutoTokenizer, PreTrainedTokenizerBase
)
from peft import LoraConfig, get_peft_model

from datasets.referit3d import build_dataloader
from scripts.v1.config import lora_config
from spatialreasoner.core.resoner import SpatialReasonerForCausalLM
from staticvars.const import REG_TOKEN

from trim.thirdparty.logging import WandbWrapper
from trim.thirdparty.logging import logger
from trim.engine import TrainerBase
from trim.callbacks.misc import *
from trim.utils import comm
from trim.utils.config import setup_hparams, DictAction


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

    @staticmethod
    def setup_lora_config(model, lora_config):
        """ Configure LoRA settings for the model. """
        def find_proj_layers(model, target_modules):
            """ Identify projection layers in the model for LoRA adaptation. """
            linear_cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (isinstance(module, linear_cls) and all(
                        x not in name for x in ["mm_detector", "mm_projector"]
                ) and any(x in name for x in target_modules)):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        # Extracting LoRA target modules
        lora_target_modules = lora_config.lora_target_modules.split(",")
        lora_module_names = find_proj_layers(model, lora_target_modules)

        # Configuring LoRA
        lora_config = LoraConfig(
            r=lora_config.lora_r, lora_alpha=lora_config.lora_alpha, target_modules=lora_module_names,
            bias=lora_config.bias, task_type=lora_config.task_type, lora_dropout=lora_config.lora_dropout
        )
        return lora_config

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
        ## Configure model tokens
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        ## Initialize vision modules
        self.model.model.initialize_vision_modules(self.hparams)
        self.model.config.tokenizer_padding_side = self.tokenizer.padding_side
        self.model.config.tokenizer_model_max_length = self.tokenizer.model_max_length
        self.model.config.compute_dtype = self.compute_dtype
        self.model.reset_detector_precision(torch.float32)

        # Freeze backbone(LLM)
        if self.hparams.freeze_backbone:
            self.model.model.requires_grad_(False)
        # Gradient checkpointing
        if self.hparams.gradient_checkpointing:
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()

        # LoRA
        if hasattr(self.hparams, "lora_config") and self.hparams.lora_config.enable:
            lora_config = self.setup_lora_config(self.model, self.hparams.lora_config)
            self.model = get_peft_model(self.model, lora_config)
        self.model.resize_token_embeddings(len(self.tokenizer))

        num_parameters = comm.count_parameters(self.model)
        logger.info(f"Number of learnable parameters: {num_parameters}")
        self.accelerator.wait_for_everyone()

    def on_training_phase_start(self):
        super().on_training_phase_start()


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

        # Scheduler and math around the number of engine steps.
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
                reduced_loss = self.accelerator.reduce(loss) / self.accelerator.num_processes
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
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hparams-file", default=path.join(path.dirname(__file__), "config.py"),
        type=str, help="path to hparams file"
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    args = parser.parse_args()

    hparams = setup_hparams(args.hparams_file, args.options)
    main(hparams)
