"""
File: resoner.py
Date: 2024/8/17
Author: yruns

Modified from https://github.com/haotian-liu/LLaVA/blob/main/llava/model/language_model/llava_llama.py

"""
from abc import ABC
from typing import *

import torch
from torch import nn

from transformers import (
    AutoConfig, AutoModelForCausalLM,
    LlamaConfig, LlamaModel, LlamaForCausalLM
)

from spatialreasoner.builder import build_mm_detector
from spatialreasoner.arch import SpatialReasonerMetaModel, SpatialReasonerMetaForCausalLM

class SpatialReasonerConfig(LlamaConfig):
    model_type = "spatial_reasoner"


class SpatialReasonerModel(SpatialReasonerMetaModel, LlamaModel):
    config_class = SpatialReasonerConfig

    def __init__(self, config: LlamaConfig):
        super(SpatialReasonerModel, self).__init__(config)


class SpatialReasonerForCausalLM(SpatialReasonerMetaForCausalLM, LlamaForCausalLM):
    config_class = SpatialReasonerConfig

    def __init__(self, config: LlamaConfig):
        super(SpatialReasonerForCausalLM, self).__init__(config)

    def get_model(self):
        pass


AutoConfig.register("spatial_reasoner", SpatialReasonerConfig)
AutoModelForCausalLM.register(SpatialReasonerConfig, SpatialReasonerForCausalLM)