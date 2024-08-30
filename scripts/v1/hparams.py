"""
File: hparams.py
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
scan_root: str = "/data3/ysh/Datasets/ScanNet/scans"
data_path: str = "data/referit3d/"
pretrained_state_dir: str = "pretrained/"
output_dir: str = f"output/grounder_reg_{now}"


# data
num_workers: int = 0
dataset_name: Literal["referit3d"] = "referit3d"
grounding_granularity: Literal["reg", "seg"] = "reg"

from configs.mask3d_conf import mask3d_cfg
pointcloud_tower_cfg: Dict = mask3d_cfg
pointcloud_output_dim: int = 256

# model
# llm_name="Meta-Llama-3.1-8B-Instruct"
llm_name = "vicuna-7b-v1.3"
model_max_length = 2048
attn_implementation = "flash_attention_2"
freeze_llm_backbone = True
freeze_mm_tower = True
detector_name = "V-DETR"
lora_config = dict(
    enable=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    lora_target_modules="q_proj,v_proj,lm_head",
    task_type="CAUSAL_LM",
)

# train
seed: int = 42
gpus: List[int] = [0]
batch_size: int = 2
gradient_accumulation_steps: int = 1

assert (batch_size / len(gpus) / gradient_accumulation_steps).is_integer(), \
    "batch_size must be divisible by the number of gpus and gradient_accumulation_steps"
per_device_train_batch_size: int = int(batch_size / len(gpus) / gradient_accumulation_steps)
per_device_eval_batch_size: int = 1

deepspeed_config: str = "configs/zero_3_stage.json"

lr: float = 2e-5
optimizer: Literal["adamw_torch"] = "adamw_torch"
warmup_ratio: float = 0.00
lr_scheduler_type: Literal["linear", "cosine"] = "linear"

gradient_checkpointing: bool = True

num_train_epochs: int = 10
save_freq: Union[str, int] = 300  # or "epoch"
resume_from_checkpoint: Optional[str] = None

# logging
log_interval: int = 1
log_project: str = "grounder_reg"
log_tag: str = "test1"


