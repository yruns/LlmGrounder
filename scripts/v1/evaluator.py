import torch
from tqdm import tqdm
import numpy as np

from trim.callbacks.default import CallbackBase
from trim.utils import comm
from utils.votenet_utils.metric_util import calc_iou


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
        bbox_preds = []
        bbox_gts = []
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
            bbox_counter += 1  # `only for 1 sample`
            for grounding_output in grounding_outputs:
                if grounding_output is None:
                    # No grounding result
                    ious.append(torch.tensor(0).cuda())
                    bbox_preds.append(torch.zeros(6))
                    bbox_gts.append(torch.zeros(6))
                    continue

                pred_bbox, gt_bbox = grounding_output["pred_bboxes"], grounding_output["gt_bboxes"]
                # if len(pred_bbox) != len(gt_bbox):
                #     self.trainer.logger.warning(f"Uid: {uid[0]}: Number of pred_bboxes and gt_bboxes are not equal!")

                iou = calc_iou(pred_bbox[1], gt_bbox[1])
                if iou >= 1:
                    self.trainer.logger.warning(f"Begin, Uid: {uid[0]}: IoU is greater than 1!")
                ious.append(iou)
                bbox_preds.append(pred_bbox[1])
                bbox_gts.append(gt_bbox[1])

            ious = torch.tensor(ious, dtype=torch.float32).cuda()

            ious_gathered = self.accelerator.gather(ious)
            if len(ious) == 0:
                continue
            for iou_ in ious_gathered:
                bbox_ious.append(iou_)
                if iou_ >= 0.25:
                    bbox_iou_25 += 1
                    if iou_ >= 0.5:
                        bbox_iou_50 += 1
                if iou_ >= 1:
                    self.trainer.logger.warning(f"Uid: {uid[0]}: IoU is greater than 1!")

        self.accelerator.wait_for_everyone()

        bbox_preds = torch.from_numpy(np.stack(bbox_preds)).cuda()
        bbox_gts = torch.from_numpy(np.stack(bbox_gts)).cuda()
        bbox_preds = self.accelerator.gather(bbox_preds)
        bbox_gts = self.accelerator.gather(bbox_gts)

        if self.accelerator.is_main_process:
            import json
            import os
            output_dir = os.path.join(self.trainer.output_dir, "bboxes")
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "bbox_preds.json"), "w") as f:
                json.dump(bbox_preds.tolist(), f)
            with open(os.path.join(output_dir, "bbox_gts.json"), "w") as f:
                json.dump(bbox_gts.tolist(), f)

            self.trainer.logger.info(f"Saved bbox_preds and bbox_gts to {output_dir}")

        # bbox_counter = self.accelerator.reduce(bbox_counter, reduction="sum")
        bbox_counter = bbox_counter * self.accelerator.num_processes
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
