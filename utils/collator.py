"""
File: collator.py
Date: 2024/8/18
Author: yruns
"""
from abc import abstractmethod
from dataclasses import dataclass


@dataclass
class DataCollatorBase(object):
    """
    Collator for data collation.
    """

    @abstractmethod
    def collate(self, batch):
        pass
