"""
File: referit3d.py
Date: 2024/8/6
Author: yruns


"""
import json
import os.path as osp

import torch
from torch.utils.data import default_collate
from typing import *

from transformers import AutoTokenizer

from .scannet import ScanNetBaseDataset
from utils.prepare_input import assemble_instruction
from utils.tokenize import tokenize_scene_token
from utils.collator import DataCollatorBase
from staticvars.const import *


class ReferItDataset(ScanNetBaseDataset):
    def __init__(
        self,
        data_path: str,
        scannet_config: Dict = None,
        tokenizer: AutoTokenizer = None,
        grounding_granularity: Literal["seg", "reg"] = "reg",
        split: Literal["train", "val"] = "train",
    ):
        super().__init__(**scannet_config)
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.grounding_granularity = grounding_granularity
        self.split = split
        self.data = json.load(open(osp.join(self.data_path, f"nr3d_{self.split}.json"), "r"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        scan_id = data["scan_id"]
        scene_data_dict = self._get_scan_data(scan_id)

        utterance = data["utterance"]
        instruction = assemble_instruction(utterance, self.grounding_granularity)

        input_ids, target_ids = tokenize_scene_token(
            instruction,
            self.tokenizer,
        )

        return dict(
            input_ids=input_ids,
            labels=target_ids,
            scene_data_dict=scene_data_dict,
        )


class ReferIt3DCollator(DataCollatorBase):

    def collate(self, batch):
        input_ids, labels, scene_data_dict = tuple(
            [batch[key] for batch in batch]
            for key in ("input_ids", "labels", "images")
        )

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX
        )

        # Truncate if necessary
        max_len = self.tokenizer.model_input_names
        input_ids, labels = (
            input_ids[:, :max_len],
            labels[:, :max_len]
        )

        # Construct attention_mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Makesure every sample has <scene>
        assert all(scene_path is not None for scene_path in scene_data_dict), "Some samples do not have <scene>"
        scene_data_dict = default_collate(scene_data_dict)


        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            scene_data_dict=scene_data_dict,
        )







