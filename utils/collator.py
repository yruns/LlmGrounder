"""
File: collator.py
Date: 2024/8/18
Author: yruns
"""
from transformers import PreTrainedTokenizerBase
from dataclasses import dataclass


@dataclass
class DataCollatorBase(object):
    """
    Collator for data collation.
    """

    tokenizer: PreTrainedTokenizerBase

    def __call__(self, batch):
        pass




