import torch

from trim.callbacks.default import CallbackBase
from trim.utils import comm

from utils.votenet_utils.metric_util import calc_iou


class Evaluator(CallbackBase):
    def on_training_epoch_end(self):
        self.trainer.model.eval()
        self.eval()
        self.trainer.model.train()

    def on_training_phase_start(self):
        self.trainer.model.eval()
        self.eval()
        self.trainer.model.train()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")

        for i, batch in enumerate(self.trainer.val_loader):
            batch_data = comm.convert_tensor_to_dtype(
                batch, self.accelerator.mixed_precision,
                ignore_keys=["mask3d_data_dict", "ptv3_data_dict"]
            )
            with torch.no_grad():
                loss, grounding_outputs = self.trainer.model(**batch_data)

            pred_bboxes, gt_bboxes = grounding_outputs[-2], grounding_outputs[-1]
            reduced_loss = self.accelerator.reduce(loss) / self.accelerator.num_processes
            for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
                iou = calc_iou(pred_bbox, gt_bbox)
                self.trainer.storage.put_scalar("bbox_iou", iou)
                if iou >= 0.25:
                    self.trainer.storage.put_scalar("bbox_iou_25", 1)
                    if iou >= 0.5:
                        self.trainer.storage.put_scalar("bbox_iou_50", 1)

            self.trainer.storage.put_scalar("bbox_counter", len(pred_bboxes))
            self.trainer.storage.put_scalar("val_loss", reduced_loss)

        loss_avg = self.trainer.storage.history("val_loss").avg
        mean_iou = self.trainer.storage.history("bbox_iou").avg
        bbox_counter = self.trainer.storage.history("bbox_counter").total
        bbox_iou_25 = self.trainer.storage.history("bbox_iou_25").total / bbox_counter
        bbox_iou_50 = self.trainer.storage.history("bbox_iou_50").total / bbox_counter

        self.trainer.logger.info(
            "Validation loss: {:.4f}, mean_iou: {:.4f}, Acc@0.25: {:4f}, Acc@0.50: {:4f}".format(
                loss_avg, mean_iou, bbox_iou_25, bbox_iou_50
            )
        )


    def on_training_phase_end(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}, epoch at {:2d}".format("mIoU",
                                                     self.trainer.best_metric_value,
                                                     self.trainer.best_metric_epoch)
        )
