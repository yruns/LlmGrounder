"""
File: main.py
Date: 2024/8/19
Author: yruns
"""
import os
from os import path
from typing import Optional

import math
import torch.optim
from accelerate import Accelerator, DeepSpeedPlugin
from transformers import (
    AutoTokenizer, PreTrainedTokenizerBase
)

from staticvars.const import *

from trim.thirdparty.logging import WandbWrapper
from trim.thirdparty.logging import logger
from trim.engine import TrainerBase
from trim.callbacks.misc import InformationWriter, IterationTimer, Resumer, CheckpointSaver
from trim.utils import comm
from trim.utils.config import setup_hparams, DictAction, Config


class Trainer(TrainerBase):

    def __init__(self, hparams: Config, accelerator: Accelerator, logger, debug=False, callbacks=None):
        super().__init__()
        self.hparams: Config = hparams
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
                        x not in name for x in [
                            "pointcloud_tower", "pointcloud_projector",
                            "grounding_tower", "grounding_projector", "grounding_cross_attn"
                        ]
                ) and any(x in name for x in target_modules)):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))

        # Extracting LoRA target modules
        lora_target_modules = lora_config.lora_target_modules.split(",")
        lora_module_names = find_proj_layers(model, lora_target_modules)

        # Configuring LoRA
        from peft import LoraConfig
        lora_config = LoraConfig(
            r=lora_config.lora_r, lora_alpha=lora_config.lora_alpha, target_modules=lora_module_names,
            bias=lora_config.bias, task_type=lora_config.task_type, lora_dropout=lora_config.lora_dropout
        )
        return lora_config

    def configure_model(self):
        pretrained_model_path = str(path.join(self.hparams.pretrained_state_dir, self.hparams.llm_name))
        assert path.exists(pretrained_model_path), f"Pretrained model `{pretrained_model_path}` not exists"
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_path,
            model_max_length=self.hparams.model_max_length,
        )
        ## Configure tokenizer
        tokenizer.pad_token = tokenizer.unk_token
        tokenizer.add_tokens([REF_TOKEN, REPLY_END_TOKEN, SOP_TOKEN, EOP_TOKEN], special_tokens=True)
        self.hparams.ref_token = REF_TOKEN
        self.hparams.ref_token_index = tokenizer.encode(REF_TOKEN, add_special_tokens=False)[0]
        self.hparams.reply_end_token = REPLY_END_TOKEN
        self.hparams.reply_end_token_index = tokenizer.encode(REPLY_END_TOKEN, add_special_tokens=False)[0]
        self.hparams.sop_token = SOP_TOKEN
        self.hparams.sop_token_index = tokenizer.encode(SOP_TOKEN, add_special_tokens=False)[0]
        self.hparams.eop_token = EOP_TOKEN
        self.hparams.eop_token_index = tokenizer.encode(EOP_TOKEN, add_special_tokens=False)[0]

        if self.hparams.use_scene_start_end:
            tokenizer.add_tokens([SCENE_START_TOKEN, SCENE_END_TOKEN], special_tokens=True)
            self.hparams.scene_start_token = SCENE_START_TOKEN
            self.hparams.scene_start_token_index = tokenizer.encode(SCENE_START_TOKEN, add_special_tokens=False)[0]
            self.hparams.scene_end_token = SCENE_END_TOKEN
            self.hparams.scene_end_token_index = tokenizer.encode(SCENE_END_TOKEN, add_special_tokens=False)[0]

        self.tokenizer = tokenizer
        self.hparams.tokenizer = tokenizer

        ## Initialize model
        from spatialreasoner.core.resoner import SpatialReasonerForCausalLM
        self.model = SpatialReasonerForCausalLM.from_pretrained(
            pretrained_model_path,
            torch_dtype=self.compute_dtype,
            attn_implementation=self.hparams.attn_implementation,
        )
        for k in [
            "use_scene_start_end", "num_encoded_scene_token",
            "grounding_loss_weight", "llm_loss_weight",
            "ref_token_index", "reply_end_token_index", "sop_token_index",
            "eop_token_index", "scene_start_token_index", "scene_end_token_index",

        ]:
            setattr(self.model.config, k, getattr(self.hparams, k, None))

        ##  Configure model tokens
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.resize_token_embeddings(len(self.tokenizer))

        ##  Initialize vision modules
        self.logger.info("##  => Initialize Pointcloud Tower ...")
        self.model.model.initialize_pointcloud_tower(self.hparams)
        self.logger.info("##  => Initialize Grounding Tower ...")
        self.model.model.initialize_grounding_tower(self.hparams)
        self.model.config.tokenizer_padding_side = self.tokenizer.padding_side
        self.model.config.tokenizer_model_max_length = self.tokenizer.model_max_length
        self.model.config.compute_dtype = self.compute_dtype

        ##  Gradient checkpointing
        if self.hparams.gradient_checkpointing:
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable()

        ##  LoRA
        if hasattr(self.hparams, "lora_config") and self.hparams.lora_config.enable:
            self.logger.info("##  => Applying LoRA ...")
            from peft import LoraConfig, get_peft_model
            lora_config = self.setup_lora_config(self.model, self.hparams.lora_config)
            self.model = get_peft_model(self.model, lora_config)
            trainable_params, all_param = self.model.get_nb_trainable_parameters()
            self.logger.info(
                f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param:.4f}"
            )

        ## Freeze parameters
        if hasattr(self.hparams, "lora_config") and self.hparams.lora_config.enable:
            model_ref = self.model.base_model.model.model
        else:
            model_ref = self.model.model

        if self.hparams.freeze_llm:
            for p in model_ref.parameters():
                p.requires_grad = False

        for p in model_ref.pointcloud_tower.parameters():
            p.requires_grad = self.hparams.tune_pointcloud_tower
        for p in model_ref.grounding_tower.parameters():
            p.requires_grad = self.hparams.tune_grounding_tower
        for p in model_ref.pointcloud_projector.parameters():
            p.requires_grad = self.hparams.tune_pointcloud_projector
        for p in model_ref.grounding_projector.parameters():
            p.requires_grad = self.hparams.tune_grounding_projector
        for p in model_ref.grounding_cross_attn.parameters():
            p.requires_grad = self.hparams.tune_grounding_cross_attn

        for n, p in self.model.named_parameters():
            if any([
                x in n
                for x in ["lm_head", "embed_tokens"]
            ]):
                p.requires_grad = True

        model_ref.load_pretrained_adapters(self.hparams.pretrained_adapters)


    def on_training_step_start(self):
        # Because Mask3D(MinkowskiEngine) and PTv3(spconv) not implemented for 'BFloat16'
        if hasattr(self.hparams, "lora_config") and self.hparams.lora_config.enable:
            model = self.model.base_model.model.model
        else:
            model = self.model.model

        model.reset_pointcloud_tower_precision(torch.float32)
        model.reset_grounding_tower_precision(torch.float32)
        super().on_training_step_start()

    def configure_dataloader(self):
        from datasets.grounded3d import build_dataloader
        assert (self.hparams.batch_size / self.accelerator.num_processes / self.accelerator.gradient_accumulation_steps).is_integer(), \
            "batch_size must be divisible by the number of gpus and gradient_accumulation_steps"
        per_device_train_batch_size: int = int(self.hparams.batch_size / self.accelerator.num_processes / self.accelerator.gradient_accumulation_steps)
        per_device_eval_batch_size: int = 1
        self.hparams.per_device_train_batch_size = per_device_train_batch_size
        self.hparams.per_device_eval_batch_size = per_device_eval_batch_size

        self.train_loader = build_dataloader(self.hparams, split="train")
        self.val_loader = build_dataloader(self.hparams, split="val")

    def configure_optimizers(self):
        optimizer_type = self.hparams.optimizer["type"]
        if optimizer_type == "AdamW":
            optimizer_cls = torch.optim.AdamW
        else:
            raise NotImplementedError(f"Optimizer {optimizer_type} is not implemented")

        self.optimizer = optimizer_cls(self.model.parameters(), **self.hparams.optimizer["params"])

        # Scheduler and math around the number of engine steps.
        num_update_steps_per_epoch_for_scheduler = math.ceil(
            len(self.train_loader) / self.accelerator.gradient_accumulation_steps)
        max_train_steps_for_scheduler = num_update_steps_per_epoch_for_scheduler * self.hparams.num_train_epochs
        self.num_update_steps_per_epoch = math.ceil(
            len(self.train_loader) / self.accelerator.gradient_accumulation_steps / self.accelerator.num_processes)
        self.total_train_steps = self.hparams.num_train_epochs * self.num_update_steps_per_epoch

        warmup_num_steps = int(self.hparams.warmup_ratio * max_train_steps_for_scheduler)
        from deepspeed.runtime.lr_schedules import WarmupDecayLR
        self.lr_scheduler = WarmupDecayLR(
            self.optimizer, total_num_steps=max_train_steps_for_scheduler,
            warmup_num_steps=warmup_num_steps, warmup_type="linear"
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
            debug=False
        )

    def training_step(self, batch_data, batch_index):
        with self.accelerator.accumulate(self.model):
            _ = batch_data.pop("uid", None)
            batch_data = comm.convert_tensor_to_dtype(
                batch_data, self.accelerator.mixed_precision,
                ignore_keys=["mask3d_data_dict", "ptv3_data_dict"]
            )

            final_loss, (llm_loss, grounding_loss) = self.model(**batch_data)
            self.accelerator.backward(final_loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                reduced_loss = self.accelerator.reduce(final_loss) / self.accelerator.num_processes
                reduced_llm_loss = self.accelerator.reduce(llm_loss) / self.accelerator.num_processes
                reduced_grounding_loss = self.accelerator.reduce(grounding_loss) / self.accelerator.num_processes

                # Anything you want to log in terminal
                loss_dict = dict(
                    loss=reduced_loss,
                    llm_loss=reduced_llm_loss,
                    grounding_loss=reduced_grounding_loss
                )
                self.comm_info["terminal_log"] = loss_dict
                # Anything you want to log in wandb
                self.comm_info["wandb_log"] = loss_dict


def main(hparams):
    """Main function."""
    comm.seed_everything(hparams.seed)

    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=1,
        deepspeed_plugin=DeepSpeedPlugin(
            hf_ds_config="configs/zero_3_stage.json",
        )
    )

    if accelerator.is_main_process:
        comm.copy_codebase(
            hparams.output_dir,
            exclude_dirs=[
                "__pycache__", "wandb", "pretrained", "data", "clip-vit-base-patch16", "output", "tutorials",
                "langdata"
            ]
        )

    from scripts.v1.evaluator import Evaluator
    from scripts.v1.misc import ModelSaver, ModelLoader
    trainer = Trainer(hparams, accelerator, logger, debug=False, callbacks=[
        Resumer(checkpoint=hparams.resume_from_checkpoint),
        IterationTimer(warmup_iter=1),
        InformationWriter(log_interval=hparams.log_interval),
        # ModelLoader(),
        Evaluator(eval_freq=hparams.save_freq),
        CheckpointSaver(save_freq=hparams.save_freq),
        # ModelSaver(save_freq=hparams.save_freq),
    ])
    trainer.fit()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--hparams-file", default=path.join(path.dirname(__file__), "hparams.py"),
        type=str, help="path to hparams file"
    )
    parser.add_argument(
        "--options", nargs="+", action=DictAction, help="custom options"
    )
    args = parser.parse_args()

    hparams = setup_hparams(args.hparams_file, args.options)
    main(hparams)
