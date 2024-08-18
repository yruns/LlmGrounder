"""
File: referit3d.py
Date: 2024/8/6
Author: yruns


"""
import json
import os.path as osp

import torch
from torch.utils.data import Dataset, DataLoader
from typing import *

from transformers import AutoTokenizer

from utils.prepare_input import assemble_instruction, prepare_for_llm
from utils.collator import DataCollatorBase
from staticvars.const import *


class ReferItDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        scan_root: str,
        tokenizer: AutoTokenizer = None,
        grounding_granularity: Literal["seg", "reg"] = "reg",
        mode: Literal["train", "val"] = "train",
    ):
        super().__init__()
        self.data_path = data_path
        self.scan_root = scan_root
        self.tokenizer = tokenizer
        self.grounding_granularity = grounding_granularity
        self.mode = mode
        self.data = json.load(open(osp.join(self.data_path, f"nr3d_{self.mode}.json"), "r"))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        scan_id = data["scan_id"]
        scan_path = osp.join(self.scan_root, scan_id, f"{scan_id}_vh_clean_2.ply")

        utterance = data["utterance"]
        instruction = assemble_instruction(utterance, self.grounding_granularity)

        input_ids, target_ids = prepare_for_llm(
            instruction,
            self.tokenizer,
            has_scene=True
        )

        return dict(
            input_ids=input_ids,
            labels=target_ids,
            scan_path=scan_path,
        )


class ReferIt3DCollator(DataCollatorBase):

    def __call__(self, batch):
        input_ids, labels, scene_paths = tuple(
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
        max_len = self.tokenizer.model_max_length
        input_ids, labels = (
            input_ids[:, :max_len],
            labels[:, :max_len]
        )

        # Construct attention_mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Makesure every sample has <scene>
        assert all(scene_path is not None for scene_path in scene_paths), "Some samples do not have <scene>"

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            scene_paths=scene_paths,
        )







