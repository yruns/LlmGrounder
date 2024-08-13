"""
File: hf_accelerate.py
Date: 2024/8/12
Author: yruns
"""
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.tracking import WandBTracker
import torch
import time
from accelerate.utils import set_seed

def main():
    set_seed(42)

    ds_plugin = DeepSpeedPlugin(
        gradient_accumulation_steps=2,
        gradient_clipping=1.0,
        hf_ds_config="ds_config.json",
    )


    print("2", time.time())

    accelerator = Accelerator()
    print("3", time.time())


    num = torch.tensor(accelerator.process_index).to(accelerator.device)

    print(accelerator.reduce(num))

    print(accelerator.process_index, accelerator.num_processes, accelerator.is_main_process)

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
    print("1", time.time())

    main()
