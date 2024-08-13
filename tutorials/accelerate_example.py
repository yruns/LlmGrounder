import argparse
import json
import math
import os
import random
from os import path

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from accelerate import Accelerator, DistributedType, DeepSpeedPlugin
from accelerate.utils import DummyOptim, DummyScheduler, set_seed
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from tqdm.auto import tqdm
from transformers import (
    get_scheduler,
)

STR_TO_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "half": torch.half,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "uint8": torch.uint8,
    "bool": torch.bool,
}

def convert_str_to_dtype(dtype_str: str) -> torch.dtype:
    """Convert string to torch.dtype"""
    if dtype_str not in STR_TO_DTYPE:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

    return STR_TO_DTYPE[dtype_str]

def convert_tensor_to_dtype(input_value, dtype):
    """Move input tensors to device"""
    if isinstance(dtype, str):
        dtype = convert_str_to_dtype(dtype)

    if isinstance(input_value, torch.Tensor):
        if input_value.dtype in [torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool]:
            return input_value
        return input_value.to(dtype)

    # convert tuple to list
    if isinstance(input_value, tuple):
        return tuple(convert_tensor_to_dtype(list(input_value), dtype))

    if isinstance(input_value, list):
        for i in range(len(input_value)):
            input_value[i] = convert_tensor_to_dtype(input_value[i], dtype)
        return input_value

    if isinstance(input_value, dict):
        for key in input_value.keys():
            input_value[key] = convert_tensor_to_dtype(input_value[key], dtype)
        return input_value

    raise NotImplementedError(f"Unsupported input type: {type(input_value)}")

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


# New Code #
def evaluate(hparams, model, eval_dataloader, accelerator):
    model.eval()
    test_loss = 0.0
    correct = 0
    for step, (data, target) in enumerate(eval_dataloader):
        with torch.no_grad():
            # data, target = data.to(accelerator.device, torch.bfloat16), target.to(accelerator.device, torch.long)
            data = convert_tensor_to_dtype(data, accelerator.mixed_precision)
            target = convert_tensor_to_dtype(target, accelerator.mixed_precision)
            output, loss = model(data, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum()

        test_loss += loss

    test_loss = accelerator.reduce(test_loss) / len(eval_dataloader.dataset)
    acc = accelerator.reduce(correct) / len(eval_dataloader.dataset)

    return acc, test_loss

def setup_dataloader(accelerator, hparams):
    # Get the dataset
    transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])
    # Let rank 0 download the data first, then everyone will load MNIST
    with accelerator.main_process_first():  # set `local=True` if your filesystem is not shared between machines
        train_dataset = MNIST(DATASETS_PATH, download=accelerator.is_main_process, train=True, transform=transform)
        eval_dataset = MNIST(DATASETS_PATH, download=accelerator.is_main_process, train=False, transform=transform)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.per_device_train_batch_size,
    )
    eval_dataloader = DataLoader(eval_dataset, batch_size=hparams.per_device_eval_batch_size)

    return train_dataloader, eval_dataloader

def setup_optimizer_scheduler(hparams, optimizer_grouped_parameters, train_dataloader, accelerator):
    # Creates Dummy Optimizer if `optimizer` was specified in the config file else creates Adam Optimizer
    optimizer_cls = (
        torch.optim.Adadelta
        if accelerator.state.deepspeed_plugin is None
        or "optimizer" not in accelerator.state.deepspeed_plugin.deepspeed_config
        else DummyOptim
    )
    optimizer = optimizer_cls(optimizer_grouped_parameters, lr=hparams.lr)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    hparams.max_train_steps = hparams.num_train_epochs * num_update_steps_per_epoch

    # Creates Dummy Scheduler if `scheduler` was specified in the config file else creates `hparams.lr_scheduler_type` Scheduler
    if (
            accelerator.state.deepspeed_plugin is None
            or "scheduler" not in accelerator.state.deepspeed_plugin.deepspeed_config
    ):
        lr_scheduler = get_scheduler(
            name=hparams.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=hparams.num_warmup_steps,
            num_training_steps=hparams.max_train_steps,
        )
    else:
        lr_scheduler = DummyScheduler(
            optimizer, total_num_steps=hparams.max_train_steps, warmup_num_steps=hparams.num_warmup_steps
        )
    return optimizer, lr_scheduler


def main(hparams):
    # set_seed(hparams.seed)

    def seed_everything(seed):
        import numpy as np
        import torch.backends.cudnn as cudnn
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        os.environ["PYTHONHASHSEED"] = str(seed)

    seed_everything(hparams.seed)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment

    # df_plugin = DeepSpeedPlugin(
    #     zero_stage=2,
    #     gradient_accumulation_steps=hparams.gradient_accumulation_steps,
    # )

    accelerator = (
        Accelerator(
            log_with=hparams.report_to,
            project_dir=hparams.output_dir,
            # deepspeed_plugin=df_plugin,
            gradient_accumulation_steps=hparams.gradient_accumulation_steps,
        )
        if hparams.with_tracking
        else Accelerator(gradient_accumulation_steps=hparams.gradient_accumulation_steps)
    )

    if accelerator.is_main_process:
        if hparams.output_dir is not None:
            os.makedirs(hparams.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    train_dataloader, eval_dataloader = setup_dataloader(accelerator, hparams)

    # Create Model
    model = Net()
    optimizer, lr_scheduler = setup_optimizer_scheduler(hparams, model.parameters(), train_dataloader, accelerator)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.gradient_accumulation_steps)
    hparams.max_train_steps = hparams.num_train_epochs * num_update_steps_per_epoch

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = hparams.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # Train!
    total_batch_size = (
            hparams.per_device_train_batch_size * accelerator.num_processes * accelerator.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataloader.dataset)}")
    logger.info(f"  Num Epochs = {hparams.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {hparams.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {accelerator.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {hparams.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(hparams.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    starting_epoch = 0
    best_metric = None
    best_metric_checkpoint = None

    # Potentially load in the weights and states from a previous save
    resume_step = None
    if hparams.resume_from_checkpoint:
        accelerator.load_state(hparams.resume_from_checkpoint)
        accelerator.print(f"Resumed from checkpoint: {hparams.resume_from_checkpoint}")
        path = os.path.basename(hparams.resume_from_checkpoint)
        training_difference = os.path.splitext(path)[0]

        if "epoch" in training_difference:
            starting_epoch = int(training_difference.replace("epoch_", "")) + 1
            completed_steps = starting_epoch * num_update_steps_per_epoch
        else:
            resume_step = int(training_difference.replace("step_", ""))
            completed_steps = resume_step
            starting_epoch = resume_step // num_update_steps_per_epoch
            resume_step -= starting_epoch * num_update_steps_per_epoch


    # update progress bar if resumed from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(starting_epoch, hparams.num_train_epochs):
        model.train()
        if hparams.with_tracking:
            total_loss = 0

        # skip new `skip_first_batches` to skip the batches when resuming from ckpt
        if hparams.resume_from_checkpoint and epoch == starting_epoch and resume_step is not None:
            active_dataloader = accelerator.skip_first_batches(train_dataloader, resume_step)
        else:
            active_dataloader = train_dataloader
        for step, (data, target) in enumerate(active_dataloader):
            with accelerator.accumulate(model):
                # data, target = data.to(accelerator.device, torch.bfloat16), target.to(accelerator.device, torch.long)
                data = convert_tensor_to_dtype(data, accelerator.mixed_precision)
                target = convert_tensor_to_dtype(target, accelerator.mixed_precision)
                output, loss = model(data, target)
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if accelerator.sync_gradients:
                    progress_bar.update(1)
                    completed_steps += 1

                    reduced_loss = accelerator.reduce(loss)
                    if completed_steps % hparams.log_interval == 0 and accelerator.is_main_process:
                        logger.info(f"Step {completed_steps}: loss {reduced_loss} , lr {optimizer.param_groups[0]['lr']}")

            # We keep track of the loss at each epoch
            if hparams.with_tracking:
                step_loss = accelerator.reduce(loss.detach().clone()).item()
                total_loss += step_loss

            if isinstance(checkpointing_steps, int):
                if completed_steps % checkpointing_steps == 0:
                    output_dir = f"step_{completed_steps}"
                    if hparams.output_dir is not None:
                        output_dir = os.path.join(hparams.output_dir, output_dir)
                    accelerator.save_state(output_dir)
            if completed_steps >= hparams.max_train_steps:
                break

        perplexity, eval_loss = evaluate(hparams, model, eval_dataloader, accelerator)
        logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

        if hparams.with_tracking:
            accelerator.log(
                {
                    "perplexity": perplexity,
                    "eval_loss": eval_loss,
                    "train_loss": total_loss / len(train_dataloader),
                    "epoch": epoch,
                    "step": completed_steps,
                },
                step=completed_steps,
            )

        if isinstance(checkpointing_steps, str) and checkpointing_steps == "epoch":
            accelerator.save_state(os.path.join(hparams.output_dir, f"epoch_{epoch}"))

        # New Code #
        # Tracks the best checkpoint and best metric
        if best_metric is None or best_metric < perplexity:
            best_metric = perplexity
            best_metric_checkpoint = os.path.join(hparams.output_dir, "best_checkpoint")
            accelerator.save_state(best_metric_checkpoint)
            accelerator.print(f"New best metric: {best_metric} at epoch {epoch}")
            accelerator.print(f"best_metric_checkpoint: {best_metric_checkpoint}")

    # New Code #
    # Loads the best checkpoint after the training is finished
    if hparams.load_best_model:
        accelerator.load_state(best_metric_checkpoint)

    # New Code #
    # Evaluates using the best checkpoint
    perplexity, eval_loss = evaluate(hparams, model, eval_dataloader, accelerator)
    logger.info(f"Best model metrics: perplexity: {perplexity} eval_loss: {eval_loss}")
    if perplexity != best_metric:
        raise AssertionError(
            f"Best metric {best_metric} does not match the metric {perplexity} of the loaded best model."
        )

    if hparams.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        torch.save(unwrapped_model.state_dict(), os.path.join(hparams.output_dir, "pytorch_model.bin"))
        # unwrapped_model.save_pretrained(
        #     hparams.output_dir,
        #     is_main_process=accelerator.is_main_process,
        #     save_function=accelerator.save,
        #     state_dict=accelerator.get_state_dict(model),
        # )

        # with open(os.path.join(hparams.output_dir, "all_results.json"), "w") as f:
        #     json.dump({"perplexity": perplexity, "eval_loss": eval_loss.item()}, f)


if __name__ == "__main__":
    import os
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

    # Arguments can be passed in through the CLI as normal and will be parsed here
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
        "--gradient_accumulation_steps",
        type=int,
        default=2,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
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
        default=None,
        # default="output/step_1500",
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
