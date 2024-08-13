import argparse
import math
from os import path

import torch.nn.functional as F
import torch.optim
import torch.optim
import torch.utils.data
import torch.utils.data
import torchvision.transforms as T
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.utils import DummyOptim, DummyScheduler
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from transformers import (
    get_scheduler,
)

from trim import TrainerBase
from trim.callbacks.misc import *
from trim.thirdparty.logging import WandbWrapper
from trim.thirdparty.logging import logger
from trim.utils import comm

DATASETS_PATH = path.join(path.dirname(__file__), "..", "..", "..", "Datasets")


class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, target):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        loss = F.nll_loss(output, target)
        return output, loss


class Trainer(TrainerBase):

    def __init__(self, hparams, accelerator, logger, debug=False, callbacks=None):
        super().__init__()
        self.hparams = hparams
        self.accelerator = accelerator
        self.logger = logger
        self.max_epoch = hparams.num_train_epochs
        self.output_dir = hparams.output_dir
        self.callbacks = callbacks or []
        self.debug = debug

    def configure_model(self):
        logger.info("=> creating model ...")
        self.model = Net()
        num_parameters = comm.count_parameters(self.model)
        logger.info(f"Number of parameters: {num_parameters}")

    def configure_dataloader(self):
        # Get the dataset
        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
        with self.accelerator.main_process_first():
            train_dataset = MNIST(DATASETS_PATH, download=self.accelerator.is_main_process, train=True,
                                  transform=transform)
            eval_dataset = MNIST(DATASETS_PATH, download=self.accelerator.is_main_process, train=False,
                                 transform=transform)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.hparams.per_device_train_batch_size,
        )
        self.val_loader = DataLoader(eval_dataset, batch_size=self.hparams.per_device_eval_batch_size)

    def configure_optimizers(self):
        optimizer_cls = (
            torch.optim.Adadelta
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

        if (
                self.accelerator.state.deepspeed_plugin is None
                or "scheduler" not in self.accelerator.state.deepspeed_plugin.deepspeed_config
        ):
            self.lr_scheduler = get_scheduler(
                name=self.hparams.lr_scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.hparams.num_warmup_steps,
                num_training_steps=max_train_steps_for_scheduler,
            )
        else:
            self.lr_scheduler = DummyScheduler(
                self.optimizer, total_num_steps=max_train_steps_for_scheduler, warmup_num_steps=self.hparams.num_warmup_steps
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
            data, target = batch_data
            output, loss = self.model(data, target)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()

            if self.accelerator.sync_gradients:
                # self.completed_steps += 1
                reduced_loss = self.accelerator.reduce(loss)
                # if self.completed_steps % self.hparams.log_interval == 0:
                #     self.logger.info(f"Step {self.completed_steps}: loss {reduced_loss}")

                # Anything you want to log in terminal
                self.comm_info["terminal_log"] = {"loss": reduced_loss}
                # Anything you want to log in wandb
                self.comm_info["wandb_log"] = {"loss": reduced_loss}



def main(hparams):
    """Main function."""
    hparams.save_path = "output/"
    hparams.log_project = "accl_test"
    hparams.log_tag = "init_1"

    comm.seed_everything(hparams.seed)
    comm.copy_codebase(hparams.save_path)

    accelerator = Accelerator()

    from trim.callbacks.evaluator import Evaluator
    trainer = Trainer(hparams, accelerator, logger, debug=False, callbacks=[
        # Resumer(checkpoint="output/step_600"),
        IterationTimer(warmup_iter=1),
        InformationWriter(log_interval=1),
        Evaluator(),
        CheckpointSaver(save_freq=300),
    ])
    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fabric MNIST Example")
    parser.add_argument("--num_train_epochs", type=int, default=8, metavar="N",
                        help="number of epochs to train (default: 14)")
    parser.add_argument("--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=1,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=32,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=1,
        help="Batch size (per device) for the evaluation dataloader.",
    )

    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument("--output_dir", type=str, default="./output", help="Where to store the final model.")
    parser.add_argument(
        "--block_size",
        type=int,
        default=None,
        help=(
            "Optional input sequence length after tokenization. The training dataset will be truncated in block of"
            " this size for training. Default to the model max input length for single sentence inputs (take into"
            " account special tokens)."
        ),
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=str,
        default="300",
        help="Whether the various states should be saved at the end of every n steps, or 'epoch' for each epoch.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        # default=None,
        default="output/step_300",
        help="If the training should continue from a checkpoint folder.",
    )
    # Whether to load the best model at the end of training
    parser.add_argument(
        "--load_best_model",
        action="store_true",
        help="Whether to load the best model at the end of training",
    )
    parser.add_argument(
        "--with_tracking",
        action="store_true",
        help="Whether to enable experiment trackers for logging.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="all",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"`, and `"dvclive"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )

    hparams = parser.parse_args()
    main(hparams)
