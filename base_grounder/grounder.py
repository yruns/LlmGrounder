"""
File: grounder.py
Date: 2024/7/28
Author: yruns


"""
import os.path
from typing import Literal

import numpy as np

from .utils import ground as ground_utils
from .utils.ground import GrounderBase


class LMMGrounder(object):

    def __init__(
            self,
            lmm: Literal["llava:34b", "gpt-4o", "qwen-vl-max"],
            vote_nums=1,
            temperature: float = 0.8,
            render_view_nums: int = 4,
            render_quality: Literal["low", "high"] = "low"
    ):

        if "llava" in lmm:
            from .grounder_core import llava
            self.lmm: GrounderBase = llava.Llava(
                model=lmm,
                temperature=temperature,
            )
        elif "gpt" in lmm:
            from .grounder_core import openai
            self.lmm: GrounderBase = openai.GPT(
                model="gpt-4o",
                api_key="sk-c1h5RWNyx6WxgiNEF2D01f2cF1A042De95F94b6b3b0cB5C3",
                base_url="https://api3.wlai.vip/v1/",
            )
        elif "qwen" in lmm:
            from .grounder_core import qwen
            self.lmm: GrounderBase = qwen.Qwen(
                model=lmm,
                api_key="sk-024ef12d081e4a1da9434f7f15176357"
            )
        else:
            raise NotImplementedError(f"Unsupported LMM: {lmm}")

        assert vote_nums >= 1, "vote_nums must be greater than or equal to 1"
        self.vote_nums = vote_nums
        self.render_view_nums = render_view_nums
        self.render_quality = render_quality
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
        uid, scan_id, caption, target_landmark = (
            sample["uid"],
            sample["scan_id"],
            sample["caption"],
            sample["target_landmark"]
        )

        target_bboxes = np.load(os.path.join("rendered_views", scan_id, str(uid), "target_bboxes.npy"))

        vote_count = np.zeros((len(target_bboxes),), dtype=int)
        for _ in range(self.vote_nums):
            try:
                rendered_views_root = os.path.join("rendered_views", scan_id, str(uid))
                if self.render_quality == "high":
                    rendered_views_root = os.path.join(rendered_views_root, "high")
                result = self.lmm.invoke(
                    description=caption, candidate_bbox_nums=len(target_bboxes),
                    images_path=[
                        os.path.join(rendered_views_root, f"view_{i}.png")
                        for i in range(self.render_view_nums)
                    ]
                )
                vote_count[result] += 1
            except Exception:
                pass

        if np.sum(vote_count) == 0:
            raise ValueError("No valid votes from LMM")

        index = np.argmax(vote_count)
        return index, target_bboxes[index]

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
        os.makedirs(os.path.join("rendered_views", scan_id, str(uid)), exist_ok=True)
        np.save(os.path.join("rendered_views", scan_id, str(uid), "target_bboxes.npy"), target_bboxes)

        # rendered_views_path = self.take_pictures(scan_id, uid=uid, bboxes=target_bboxes)

        return None
