"""
File: lmmgroundercfg.py
Date: 2024/7/31
Author: yruns

Description: This file contains ...
"""

from typing import Dict, Any, Literal, List, Optional
from configs.baseconfig import BaseConfig

class LMMGrounderConfig(BaseConfig):

    lmm_name: Literal["llava:34b", "gpt-4o", "qwen-vl-max"] = "llava:34b"
    vote_nums: int = 1
    render_view_nums: int = 4
    render_quality: Literal["low", "high"] = "high"

    @classmethod
    def log_self(cls, logger):
        for attr in dir(cls):
            if not attr.startswith("__") and not callable(getattr(cls, attr)):
                logger.info(f"{attr}: {getattr(cls, attr)}")

if __name__ == "__main__":
    from loguru import logger

    # 测试代码
    # config = LMMGrounderConfig()

    LMMGrounderConfig.log_self(logger)
