"""
File: grounder_reg.py
Date: 2024/8/16
Author: yruns
"""
import time
from typing import *

"""
Configuration file for grounder(reg).
"""
now = time.strftime("%Y%m%d-%H%M%S", time.localtime())


# file paths
data_path: str = "data/referit3d/"
pretrained_state_dir: str = "pretrained/"
output_dir: str = f"output/grounder_reg_{now}"

# data
dataset_name: Literal["referit3d"] = "referit3d"

# model
llm_name="Meta-Llama-3.1-8B-Instruct"
detector_name="V-DETR"

# train
seed: int = 42
gpus: List[int] = [0]
batch_size: int = 4
gradient_accumulation_steps: int = 1

assert (batch_size / len(gpus) / gradient_accumulation_steps).is_integer(), \
    "batch_size must be divisible by the number of gpus and gradient_accumulation_steps"
per_device_train_batch_size: int = int(batch_size / len(gpus) / gradient_accumulation_steps)
per_device_eval_batch_size: int = 1

deepspeed_config: str = "configs/zero_3_stage.json"

optimizer: Literal["adamw_torch"] = "adamw_torch"
scheduler: Literal["linear", "cosine"] = "linear"

num_workers: int = 4
learning_rate: float = 1e-4
num_train_epochs: int = 10






if __name__ == "__main__":
    # print(Config.model.log_self())
    pass