"""
File: lmmgroundercfg.py
Date: 2024/7/31
Author: yruns


"""

from typing import Literal

from configs import GlobalConfig


class LMMGrounderConfig(GlobalConfig):
    lmm_name: Literal["llava:34b", "gpt-4o", "qwen-vl-max"] = "gpt-4o"
    vote_nums: int = 1
    render_view_nums: int = 4
    render_quality: Literal["low", "high"] = "high"
    temperature: float = 0.0


if __name__ == "__main__":
    from loguru import logger

    config = LMMGrounderConfig()

    config.log_self(logger)
