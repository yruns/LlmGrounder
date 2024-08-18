"""
File: tokenize.py
Date: 2024/8/17
Author: yruns
"""
from typing import Union, Tuple
import re

import torch
from transformers import AutoTokenizer

from staticvars.const import *

def tokenize_with_mask(text: str, tokenizer, wrapper: Union[Tuple, str]):
    assert len(wrapper) == 2, "wrapper must be a tuple of two elements(begin_token, end_token)"

    # Extract the text between the wrapper tokens
    pattern = re.compile(rf'{wrapper[0]}:\s*([\s\S]*?{wrapper[1]})')
    matches = pattern.finditer(text)

    indices_sets = []
    for match in matches:
        start_pos = match.start(1)
        end_pos = match.end(1)
        indices_sets.append(set(list(range(start_pos, end_pos))))

    encoding = tokenizer(text, return_offsets_mapping=True)
    input_ids, offset_mapping = encoding.input_ids, encoding.offset_mapping

    mask_ids = torch.zeros(len(input_ids), dtype=torch.long)

    def in_range(start, end, indices_sets):
        return any(indices_set.intersection(set(list(range(start, end)))) for indices_set in indices_sets)

    for i, (start, end) in enumerate(offset_mapping):
        if in_range(start, end, indices_sets):
            mask_ids[i] = 1

    # Double check
    assert input_ids == tokenizer.encode(text)

    return torch.tensor(input_ids, dtype=torch.long), mask_ids


def tokenize_scene_token(
        instruction: str,
        tokenizer,
        scene_token=SCENE_TOKEN,
        scene_token_index=SCENE_TOKEN_INDEX
) -> Tuple[torch.Tensor, torch.Tensor]:
    tokenizer_copy = AutoTokenizer.from_pretrained(tokenizer.name_or_path)
    tokenizer_copy.add_tokens([scene_token], special_tokens=True)
    temp_ids = tokenizer_copy(scene_token, add_special_tokens=False).input_ids[0]

    input_ids, mask_ids = tokenize_with_mask(instruction, tokenizer_copy, wrapper=(ROLES["reply"], REPLY_END_TOKEN))
    input_ids[input_ids == temp_ids] = scene_token_index

    target_ids = input_ids.clone()
    target_ids[mask_ids == 1] = IGNORE_INDEX

    return input_ids, target_ids



