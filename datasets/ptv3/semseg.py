"""
ScanNet20 / ScanNet200 / ScanNet Data Efficient Dataset

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import glob
import os
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset

from datasets.scannet.scannet200_constants import (
    VALID_CLASS_IDS_20,
    VALID_CLASS_IDS_200,
)
from .transform import Compose


# @DATASETS.register_module()
class PTv3Dataset(Dataset):
    class2id = np.array(VALID_CLASS_IDS_20)

    def __init__(
            self,
            split="train",
            data_root="data/scannet",
            transform=None,
            lr_file=None,
            la_file=None,
            ignore_index=-1,
            test_mode=False,
            test_cfg=None,
            cache=False,
            loop=1,
    ):
        super(PTv3Dataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.transform = Compose(transform)
        self.cache = cache
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None

        if test_mode:
            self.test_voxelize = self.transform.transforms_builder(self.test_cfg.voxelize)
            self.test_crop = (
                self.transform.transforms_builder(self.test_cfg.crop) if self.test_cfg.crop else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        if lr_file:
            self.data_list = [
                os.path.join(data_root, "train", name + ".pth")
                for name in np.loadtxt(lr_file, dtype=str)
            ]
        else:
            self.data_list = self.get_data_list()
        self.la = torch.load(la_file) if la_file else None
        self.ignore_index = ignore_index
        print(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

        # Construct scan_id to scan_data mapping
        self.scan_id_to_data = {}
        for data_path in self.data_list:
            scan_id = os.path.basename(data_path).split(".")[0]
            self.scan_id_to_data[scan_id] = data_path


    def get_data_list(self):
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        return data_list

    def _get_ptv3_data(self, idx):
        data_path = self.data_list[idx % len(self.data_list)]
        data = torch.load(data_path)
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt20" in data.keys():
            segment = data["semantic_gt20"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            mask = np.ones_like(segment).astype(np.bool_)
            mask[sampled_index] = False
            segment[mask] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]


    def __getitem__(self, idx):
        raise RuntimeError("You should not call this function directly. Please use `get_ptv3_data` instead.")

    def __len__(self):
        raise RuntimeError("You should not call this function directly.")


# @DATASETS.register_module()
class ScanNet200Dataset(PTv3Dataset):
    class2id = np.array(VALID_CLASS_IDS_200)

    def get_data(self, idx):
        data = torch.load(self.data_list[idx % len(self.data_list)])
        coord = data["coord"]
        color = data["color"]
        normal = data["normal"]
        scene_id = data["scene_id"]
        if "semantic_gt200" in data.keys():
            segment = data["semantic_gt200"].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1
        if "instance_gt" in data.keys():
            instance = data["instance_gt"].reshape([-1])
        else:
            instance = np.ones(coord.shape[0]) * -1
        data_dict = dict(
            coord=coord,
            normal=normal,
            color=color,
            segment=segment,
            instance=instance,
            scene_id=scene_id,
        )
        if self.la:
            sampled_index = self.la[self.get_data_name(idx)]
            segment[sampled_index] = self.ignore_index
            data_dict["segment"] = segment
            data_dict["sampled_index"] = sampled_index
        return data_dict



def build_datasets(args, cfg, mode="all"):
    data_cfg_train = cfg.data.train
    data_cfg_val = cfg.data.val

    return_datasets = []
    if mode in ["train", "all"]:
        data_type = data_cfg_train.pop("type")
        builder = PTv3Dataset if data_type == "ScanNetDataset" else ScanNet200Dataset
        return_datasets.append(builder(**data_cfg_train))
    if mode in ["val", "all"]:
        data_type = data_cfg_val.pop("type")
        builder = PTv3Dataset if data_type == "ScanNetDataset" else ScanNet200Dataset
        return_datasets.append(builder(**data_cfg_val))

    return return_datasets



