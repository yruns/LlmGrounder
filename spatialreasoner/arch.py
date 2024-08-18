"""
File: arch.py
Date: 2024/8/17
Author: yruns
"""
from abc import ABC, abstractmethod
from typing import *

import torch
from torch import nn
from transformers import Cache

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


    def prepare_for_multimodal(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            labels: Optional[torch.LongTensor] = None,
            scene_data_dict: Dict = None
    ):
        pass




