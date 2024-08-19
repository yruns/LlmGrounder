"""
Evaluate Hook

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import torch

from trim.callbacks.default import CallbackBase
from trim.utils import comm


class Evaluator(CallbackBase):
    def on_training_epoch_end(self):
        self.eval()

    def eval(self):
        self.trainer.logger.info(">>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>")
        self.trainer.model.eval()
        test_loss = 0.0
        correct = 0
        for i, batch in enumerate(self.trainer.val_loader):
            with torch.no_grad():
                data, target = comm.convert_tensor_to_dtype(batch, self.accelerator.mixed_precision)
                output, loss = self.trainer.model(data, target)
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum()

            test_loss += loss

        test_loss = self.accelerator.reduce(test_loss) / len(self.trainer.val_loader.dataset)
        acc = self.accelerator.reduce(correct) / len(self.trainer.val_loader.dataset)

        self.trainer.logger.info(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {acc:.4f}")

        self.trainer.wandb.log({"val_loss": test_loss, "val_acc": acc}, step=self.trainer.completed_steps)

        self.trainer.logger.info("<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<")
        self.trainer.comm_info["current_metric_value"] = acc  # save for saver
        self.trainer.comm_info["current_metric_name"] = "acc"  # save for saver

        if self.trainer.best_metric_value < acc:
            self.trainer.best_metric_value = acc
            self.trainer.best_metric_epoch = self.trainer.epoch
            self.trainer.logger.info("New best metric!")

    def on_training_phase_end(self):
        self.trainer.logger.info(
            "Best {}: {:.4f}, epoch at {:2d}".format("mIoU",
                                                     self.trainer.best_metric_value,
                                                     self.trainer.best_metric_epoch)
        )
