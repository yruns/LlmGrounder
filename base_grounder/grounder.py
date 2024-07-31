"""
File: grounder.py
Date: 2024/7/28
Author: yruns

Description: This file contains ...
"""
import numpy as np
import os.path
import torch
import warnings
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
from typing import Literal

from utils import ground as ground_utils
from utils import openai
from utils.ground import GrounderBase


class LMMGrounder(object):

    def __init__(self,
                 lmm: Literal["llava-1.5-7b-hf", "gpt-4o"],
                 render_view_nums: int = 4,
                 render_quality: Literal["low", "high"] = "low"
                 ):
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

        # model = LlavaForConditionalGeneration.from_pretrained(
        #     model_name,
        #     torch_dtype=torch.float16,
        #     low_cpu_mem_usage=True,
        # ).to(0)
        #
        # processor = AutoProcessor.from_pretrained(model_name)

        if lmm == "llava-1.5-7b-hf":
            self.lmm: GrounderBase = None

        elif lmm == "gpt-4o":
            self.lmm: GrounderBase = openai.GPT(
                model="gpt-4o",
                api_key="sk-c1h5RWNyx6WxgiNEF2D01f2cF1A042De95F94b6b3b0cB5C3",
                base_url="https://api3.wlai.vip/v1/",
                pydantic_object=openai.Result
            )

        self.locator = ground_utils.Locator()
        self.picture_taker = ground_utils.PictureTaker(
            scan_root="/data3/ysh/Datasets/ScanNet/scans",
            view_nums=render_view_nums,
            quality=render_quality
        )

    def locate_bboxes(self, obj_name, scan_id):
        return self.locator.locate(obj_name, scan_id)

    def take_pictures(self, scan_id, uid, bboxes):
        return self.picture_taker.take_pictures(scan_id, uid=uid, bboxes=bboxes)

    def ask_lmm(self, sample):
        return self.lmm.invoke()

    def ask_gpt(self, sample):
        uid, scan_id, caption, target_landmark = (
            sample["uid"],
            sample["scan_id"],
            sample["caption"],
            sample["target_landmark"]
        )

        target_bboxes = np.load(os.path.join("rendered_views", scan_id, str(uid), "target_bboxes.npy"))

        result = self.lmm.invoke(
            description=caption,
            images_path=[os.path.join("rendered_views", scan_id, str(uid), f"view_{i}.png") for i in [0, 1]]
        )

        return result.index, target_bboxes[result.index]

    def ground(self, sample):
        uid, scan_id, caption, target_landmark = (
            sample["uid"],
            sample["scan_id"],
            sample["caption"],
            sample["target_landmark"]
        )

        target_obj_name, landmarks_obj_name = target_landmark["target"], target_landmark["landmark"]

        target_bboxes = self.locate_bboxes(target_obj_name, scan_id)
        assert len(target_bboxes) != 0, f"No target object found in the sample that uid={uid}"

        target_bboxes = np.stack(target_bboxes)
        # import os
        # np.save(os.path.join("rendered_views", scan_id, str(uid), "target_bboxes.npy"), target_bboxes)

        rendered_views_path = self.take_pictures(scan_id, uid=uid, bboxes=target_bboxes)

        return None
