"""
File: referit3d.py
Date: 2024/8/6
Author: yruns


"""
import json
import os.path as osp
from typing import *

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase

from staticvars.const import *
from utils.collator import DataCollatorBase
from utils.prepare_input import assemble_instruction
from utils.tokenize import tokenize_scene_token
from .mask3d.semseg import Mask3DDataset


class Grounded3DDataset(Mask3DDataset):
    def __init__(
            self,
            data_path: str,
            mask3d_cfg: Dict = None,
            tokenizer: PreTrainedTokenizerBase = None,
            grounding_granularity: Literal["seg", "reg"] = "reg",
            split: Literal["train", "val"] = "train",
    ):
        super(Grounded3DDataset, self).__init__(**mask3d_cfg)
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.grounding_granularity = grounding_granularity
        self.split = split
        self.database = json.load(open(osp.join(self.data_path, f"nr3d_{self.split}.json"), "r"))

        self.tokenizer_copy = AutoTokenizer.from_pretrained(tokenizer.name_or_path)
        original_tokenizer_len = len(self.tokenizer_copy)
        added_tokens = []
        for idx in range(original_tokenizer_len, len(tokenizer)):
            added_tokens.append(tokenizer.decode([idx]))
        added_tokens.append(SCENE_TOKEN)

        self.tokenizer_copy.add_tokens(added_tokens, special_tokens=True)
        self.scene_token_id = self.tokenizer_copy(SCENE_TOKEN, add_special_tokens=False).input_ids[0]

    def __len__(self):
        return len(self.database)

    def __getitem__(self, idx):
        data = self.database[idx]
        scan_id = data["scan_id"]
        scene_data_dict = self._get_scan_data(scan_id)

        utterance = data["utterance"]
        instruction = assemble_instruction(utterance, self.grounding_granularity)

        input_ids, target_ids = tokenize_scene_token(
            instruction,
            self.tokenizer_copy,
            self.scene_token_id
        )

        return dict(
            input_ids=input_ids,
            labels=target_ids,
            scene_data_dict=scene_data_dict,
        )


class Grounded3DCollator(DataCollatorBase):

    def __init__(self, tokenizer: PreTrainedTokenizerBase, mask3d_collate: Callable):
        super(Grounded3DCollator, self).__init__()
        self.tokenizer = tokenizer
        self.mask3d_collate = mask3d_collate

    def collate(self, batch):
        input_ids, labels, scene_data_dict = tuple(
            [batch[key] for batch in batch]
            for key in ("input_ids", "labels", "scene_data_dict")
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
        model_max_length = self.tokenizer.model_max_length
        input_ids, labels = (
            input_ids[:, :model_max_length],
            labels[:, :model_max_length]
        )

        # Construct attention_mask
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)

        # Makesure every sample has <scene>
        assert all(scene_data is not None for scene_data in scene_data_dict), "Some samples do not have <scene>"
        scene_data_dict = self.mask3d_collate(scene_data_dict)

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            scene_data_dict=scene_data_dict,
        )


def build_dataloader(hparams, split: Literal["train", "val"]):
    mask3d_data_cfg = hparams.mask3d_cfg["data"]

    dataset = Grounded3DDataset(
        data_path=hparams.data_path,
        mask3d_cfg=mask3d_data_cfg[f"{split}_dataset"],
        tokenizer=hparams.tokenizer,
        grounding_granularity=hparams.grounding_granularity,
        split=split,
    )

    per_device_batch_size = hparams.per_device_train_batch_size \
        if split == "train" else hparams.per_device_eval_batch_size

    from .mask3d.utils import VoxelizeCollate

    return DataLoader(
        dataset=dataset,
        batch_size=per_device_batch_size,
        shuffle=(split == "train"),
        num_workers=hparams.num_workers,
        collate_fn=Grounded3DCollator(
            tokenizer=hparams.tokenizer,
            mask3d_collate=VoxelizeCollate(mask3d_data_cfg[f"{split}_collation"]),
        ).collate,
    )
