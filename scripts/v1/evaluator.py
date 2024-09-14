import torch

from trim.callbacks.default import CallbackBase
from trim.utils import comm

from utils.votenet_utils.metric_util import calc_iou
from tqdm import tqdm


class Evaluator(CallbackBase):
    
    def __init__(self, eval_freq):
        self.eval_freq = eval_freq
        self.eval_steps = None

        if isinstance(self.eval_freq, int):
            # step based
            self.eval_steps = self.eval_freq
        elif isinstance(self.eval_freq, str) and self.eval_freq == "epoch":
            # epoch based
            self.eval_steps = self.trainer.num_update_steps_per_epoch
    
    # def on_training_epoch_end(self):
    #     self.trainer.model.eval()
    #     self.eval()
    #     self.trainer.model.train()
    #
    # def on_training_phase_start(self):
    #     self.trainer.model.eval()
    #     self.eval()
    #     self.trainer.model.train()


    def on_training_step_end(self):
        if (
                hasattr(self, "eval_steps")
                and self.trainer.completed_steps % self.eval_steps == 0
        ):
            self.trainer.model.eval()
            self.eval()
            self.trainer.model.train()

    def eval(self):
        self.trainer.logger.info("\n>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        if hasattr(self.trainer.hparams, "lora_config") and self.trainer.hparams.lora_config.enable:
            model = self.trainer.model.base_model.model.model
        else:
            model = self.trainer.model.model
        model.reset_pointcloud_tower_precision(torch.float32)
        model.reset_grounding_tower_precision(torch.float32)

        val_iter = tqdm(self.trainer.val_loader, desc="Validation") \
            if self.accelerator.is_main_process else self.trainer.val_loader

        # Metrics
        bbox_ious = []
        bbox_counter = 0
        bbox_iou_25 = 0
        bbox_iou_50 = 0

        for i, batch in enumerate(val_iter):
            uid = batch.pop("uid", None)
            batch_data = comm.convert_tensor_to_dtype(
                batch, self.accelerator.mixed_precision,
                ignore_keys=["mask3d_data_dict", "ptv3_data_dict"]
            )
            with torch.no_grad():
                output_ids, grounding_outputs = self.trainer.model(**batch_data)

            ious = []
            bbox_counter += 1   # `only for 1 sample`
            for grounding_output in grounding_outputs:
                if grounding_output is None:
                    # No grounding result
                    continue

                pred_bboxes, gt_bboxes = grounding_output["pred_bboxes"], grounding_output["gt_bboxes"]
                if len(pred_bboxes) != len(gt_bboxes):
                    self.trainer.logger.warning(f"Uid: {uid[0]}: Number of pred_bboxes and gt_bboxes are not equal!")

                for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
                    iou = torch.tensor(calc_iou(pred_bbox, gt_bbox)).cuda()
                    ious.append(iou)

            ious = self.accelerator.gather(ious)
            if len(ious) == 0:
                continue
            ious = torch.cat(ious, dim=0).tolist()
            for iou in ious:
                bbox_ious.append(iou)
                if iou >= 0.25:
                    bbox_iou_25 += 1
                    if iou >= 0.5:
                        bbox_iou_50 += 1

        self.accelerator.wait_for_everyone()
        mean_iou = sum(bbox_ious) / bbox_counter
        bbox_iou_25 = bbox_iou_25 / bbox_counter
        bbox_iou_50 = bbox_iou_50 / bbox_counter

        self.trainer.logger.info(
            "Mean_iou: {:.4f}, Acc@0.25: {:4f}, Acc@0.50: {:4f}".format(
                mean_iou, bbox_iou_25, bbox_iou_50
            )
        )

        current_epoch = self.trainer.epoch + 1
        self.trainer.wandb.log({
            "mean_iou": mean_iou,
            "Acc@0.25": bbox_iou_25,
            "Acc@0.50": bbox_iou_50,
            "epoch": current_epoch
        })

        self.trainer.logger.info(">>>>>>>>>>>>>>>> Evaluation Over >>>>>>>>>>>>>>>>\n")

