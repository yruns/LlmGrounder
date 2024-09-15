"""
File: hparams.py
Date: 2024/8/16
Author: yruns
"""
import time
from typing import *

"""
Configuration file for grounder(seg).
"""
now = time.strftime("%Y%m%d-%H%M%S", time.localtime())

# *************** file paths ***************
scan_root: str = "/data3/ysh/Datasets/ScanNet/scans"
data_path: str = "data/referit3d/"
pretrained_state_dir: str = "pretrained/"
output_dir: str = f"output/grounder_reg_{now}"

# *************** model ***************
# llm_name="Meta-Llama-3.1-8B-Instruct"
llm_name = "vicuna-7b-v1.3"
model_max_length = 2048
attn_implementation = "flash_attention_2"

freeze_llm = True
tune_pointcloud_tower = False
tune_grounding_tower = True
tune_pointcloud_projector = True
tune_grounding_projector = True
tune_grounding_cross_attn = True

lora_config = dict(
    enable=True,
    lora_r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    lora_target_modules="q_proj,v_proj",
    task_type="CAUSAL_LM",
)
use_scene_start_end = False
num_encoded_scene_token: int = 384

from configs.ptv3_conf import ptv3_cfg
from configs.mask3d_conf import mask3d_cfg

ptv3_cfg["model"]["K"] = num_encoded_scene_token
pointcloud_tower_cfg: Dict = ptv3_cfg
pointcloud_output_dim: int = ptv3_cfg["model"]["enc_channels"][-1] + 3  # Note `+3` for xyz
grounding_tower_cfg = mask3d_cfg

pretrained_adapters = dict(
    pointcloud_tower=dict(path="pretrained/PTv3-Scannet200.pth", strict=False),
    grounding_tower=dict(path="pretrained/Mask3D-Scannet200.pth", strict=False),
)

grounding_loss_weight: float = 1.0
llm_loss_weight: float = 1.0

# *************** data ***************
num_workers: int = 4
dataset_name: Literal["referit3d"] = "referit3d"
grounding_granularity: Literal["reg", "seg"] = "seg"

# *************** training ***************
seed: int = 42
batch_size: int = 96
gradient_accumulation_steps: int = 2

deepspeed_config: str = "configs/zero_3_stage.json"

lr: float = 3e-4
optimizer: Dict = dict(
    type="AdamW",
    params=dict(
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.0
    )
)
warmup_ratio: float = 0.03
scheduler: Dict = dict(
    type="WarmupDecayLR",
    params=dict(
        warmup_type="linear"
    )
)

gradient_checkpointing: bool = True

num_train_epochs: int = 10
save_freq: Union[str, int] = 3  # or "epoch"
# resume_from_checkpoint: Optional[str] = None
resume_from_checkpoint: Optional[str] = "output/grounder_reg_20240914-164840/checkpoints/step_150"

# *************** logging ***************
log_interval: int = 1
log_project: str = "grounder_reg"
log_tag: str = "referit3-val1000"
