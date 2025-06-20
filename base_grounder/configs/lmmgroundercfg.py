"""
File: lmmgroundercfg.py
Date: 2024/7/31
Author: yruns


"""

from typing import Literal

from ..configs.baseconfig import BaseConfig


class LMMGrounderConfig(BaseConfig):
    lmm_name: Literal["llava:34b", "gpt-4o", "qwen-vl-max"] = "gpt-4o"
    vote_nums: int = 1
    render_view_nums: int = 4
    render_quality: Literal["low", "high"] = "high"
    temperature: float = 0.0


if __name__ == "__main__":
    from loguru import logger

    # 测试代码
    # config = LMMGrounderConfig()

    LMMGrounderConfig.log_self(logger)
