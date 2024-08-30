"""
File: collator.py
Date: 2024/8/18
Author: yruns
"""
from abc import abstractmethod
from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase


@dataclass
class DataCollatorBase(object):
    """
    Collator for data collation.
    """

    @abstractmethod
    def collate(self, batch):
        pass
