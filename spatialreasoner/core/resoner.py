"""
File: resoner.py
Date: 2024/8/17
Author: yruns

Modified from https://github.com/haotian-liu/LLaVA/blob/main/llava/model/language_model/llava_llama.py

"""
from typing import *

import torch
from torch import nn
from transformers import (
    AutoConfig, AutoModelForCausalLM,
    LlamaConfig, LlamaModel, LlamaForCausalLM, Cache
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from spatialreasoner.arch import SpatialReasonerMetaModel, SpatialReasonerMetaForCausalLM


class SpatialReasonerConfig(LlamaConfig):
    model_type = "spatial_reasoner"


class SpatialReasonerModel(SpatialReasonerMetaModel, LlamaModel):
    config_class = SpatialReasonerConfig

    def __init__(self, config: LlamaConfig):
        super(SpatialReasonerModel, self).__init__(config)


class SpatialReasonerForCausalLM(LlamaForCausalLM, SpatialReasonerMetaForCausalLM):
    config_class = SpatialReasonerConfig

    def __init__(self, config: LlamaConfig):
        super(SpatialReasonerForCausalLM, self).__init__(config)
        self.model = SpatialReasonerModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            scene_data_dict: Dict = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                inputs_embeds,
                labels
            ) = self.prepare_for_multimodal(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=labels,
                scene_data_dict=scene_data_dict
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )


AutoConfig.register("spatial_reasoner", SpatialReasonerConfig)
AutoModelForCausalLM.register(SpatialReasonerConfig, SpatialReasonerForCausalLM)
