import os
import shutil

import torch

from trim.callbacks.default import CallbackBase


class ModelSaver(CallbackBase):
    """
    modelSaver

    If you are using this callback, be sure to set `self.trainer.comm_info["current_metric_value"]` and
    `self.trainer.comm_info["current_metric_name"]` before executing this callback.
    It is recommended to set these values in the `Evaluator` callback.
    """

    def __init__(self, save_freq, save_last_only=False):
        self.save_freq = save_freq
        self.save_lastest_only = save_last_only
        self.save_steps = None
        self.last_model = None

    def on_training_phase_start(self):
        if isinstance(self.save_freq, int):
            # step based
            self.save_steps = self.save_freq
        elif isinstance(self.save_freq, str) and self.save_freq == "epoch":
            # epoch based
            self.save_steps = self.trainer.num_update_steps_per_epoch

    def on_training_step_end(self):
        if (
                hasattr(self, "save_steps")
                and self.trainer.completed_steps % self.save_steps == 0
        ):
            self.save_model()

    def save_model(self):
        output_dir = "model_epoch_{epoch}".format(epoch=self.trainer.epoch + 1) if self.save_freq == "epoch" else \
            "model_step_{step}".format(step=self.trainer.completed_steps)
        self.trainer.logger.info("=> Saving model to: " + output_dir)
        output_dir = os.path.join(self.trainer.output_dir, output_dir)

        if hasattr(self.trainer.hparams, "lora_config") and self.trainer.hparams.lora_config.enable:
            model = self.trainer.model.base_model.model.model
        else:
            model = self.trainer.model.model

        ## Save model(LLM part)
        self.accelerator.unwrap_model(model).save_pretrained(
            save_directory=output_dir,
            is_main_process=self.accelerator.is_main_process,
            state_dict=self.accelerator.get_state_dict(model),
            save_func=self.accelerator.save
        )

        ## Save other components
        if self.accelerator.is_main_process:
            pointcloud_projector = model.get_model().get_pointcloud_projector()
            grounding_projector = model.get_model().get_grounding_projector()
            grounding_cross_attn = model.get_model().get_grounding_cross_attn()
            pointcloud_tower = model.get_model().get_pointcloud_tower()
            grounding_tower = model.get_model().get_grounding_tower()

            torch.save({
                "pointcloud_projector": pointcloud_projector.state_dict(),
                "grounding_projector": grounding_projector.state_dict(),
                "grounding_cross_attn": grounding_cross_attn.state_dict(),
                "pointcloud_tower": pointcloud_tower.state_dict(),
                "grounding_tower": grounding_tower.state_dict(),
            }, os.path.join(output_dir, "other_components.pth"))

        if self.save_lastest_only and self.last_model is not None and self.accelerator.is_main_process:
            shutil.rmtree(self.last_model)
            self.last_model = output_dir


class ModelLoader(CallbackBase):
    def __init__(self, model_path):
        self.model_path = model_path

    def on_training_phase_start(self):
        self.trainer.logger.info("=> Loading model & weight ...")

        ## Load model(LLM part)
        from spatialreasoner.core.resoner import SpatialReasonerForCausalLM
        self.trainer.model = SpatialReasonerForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=self.trainer.compute_dtype,
            attn_implementation=self.trainer.hparams.attn_implementation,
        )

        ## Load other components
        from trim.engine.default import load_state_dict
        other_components = torch.load(os.path.join(self.model_path, "other_components.pth"))
        load_state_dict(
            other_components["pointcloud_projector"],
            self.trainer.model.get_model().get_pointcloud_projector(),
            logger=self.trainer.logger, strict=True
        )
        load_state_dict(
            other_components["grounding_projector"],
            self.trainer.model.get_model().get_grounding_projector(),
            logger=self.trainer.logger, strict=True
        )
        load_state_dict(
            other_components["grounding_cross_attn"],
            self.trainer.model.get_model().get_grounding_cross_attn(),
            logger=self.trainer.logger, strict=True
        )
        load_state_dict(
            other_components["pointcloud_tower"],
            self.trainer.model.get_model().get_pointcloud_tower(),
            logger=self.trainer.logger, strict=False
        )
        load_state_dict(
            other_components["grounding_tower"],
            self.trainer.model.get_model().get_grounding_tower(),
            logger=self.trainer.logger, strict=False
        )

        self.trainer.logger.info("=> Model loaded successfully!")
