"""
File: arch.py
Date: 2024/8/17
Author: yruns
"""
from abc import ABC, abstractmethod
from typing import *

import torch
from torch import nn

from spatialreasoner.builder import build_mm_detector


class SpatialReasonerMetaModel:

    def __init__(self, config):
        super(SpatialReasonerMetaModel).__init__(config)

        if "mm_detector" in config:
            self.mm_detector = build_mm_detector()

    def get_detector(self):
        return self.mm_detector


class SpatialReasonerMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass




