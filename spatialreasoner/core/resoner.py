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
from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from spatialreasoner.arch import SpatialReasonerMetaModel, SpatialReasonerMetaForCausalLM
from staticvars.const import *


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

    def forward(self, **kwargs):
        if self.training:
            return self.training_forward(**kwargs)
        else:
            if "past_key_values" in kwargs:
                try:
                    return super().forward(**kwargs)
                except Exception as e:
                    exit(1)
            else:
                return self.inference_forward(**kwargs)

    def training_forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = True,
            return_dict: Optional[bool] = None,
            mask3d_data_dict: Dict = None,
            ptv3_data_dict: Dict = None
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        raw_input_ids = input_ids.clone()

        ## => Prepare for multimodal(insert scene feature)
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
                scene_data_dict=ptv3_data_dict
            )

        ## => Call original LLM's `forward()` method
        llm_output: CausalLMOutputWithPast = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict
        )
        llm_hidden_states = llm_output.hidden_states

        ## => Call grounding tower to generate segmentation
        grounding_loss = self.call_grounding_tower(raw_input_ids, mask3d_data_dict, llm_hidden_states)

        final_loss = llm_output.loss * getattr(self.config, "llm_loss_weight") + \
                     grounding_loss * getattr(self.config, "grounding_loss_weight")
        return final_loss, (llm_output.loss, grounding_loss)

    @torch.no_grad()
    def inference_forward(
            self,
            input_ids: torch.LongTensor = None,
            mask3d_data_dict: Dict = None,
            ptv3_data_dict: Dict = None,
            **kwargs
    ):
        assert input_ids.shape[0] == 1, "We only support batch size 1 for inference"
        kwargs["output_hidden_states"] = True
        output, multimodal = self.generate(
            inputs=input_ids, ptv3_data_dict=ptv3_data_dict, **kwargs,
            num_return_sequences=1, return_dict_in_generate=True,
        )

        output_ids = output.sequences
        hidden_states = output.hidden_states

        num_layers = len(hidden_states[0])
        hidden_states = tuple([
            torch.cat(
                [hidden_states[seq_idx][layer_idx] for seq_idx in range(len(hidden_states))],
                dim=1
            ) for layer_idx in range(num_layers)
        ])

        ## => Get the grounding tower and projector first
        grounding_tower = self.get_grounding_tower()
        grounding_cross_attn = self.get_grounding_cross_attn()

        ## => Call the grounding tower's `encode()` method
        encoded_state = grounding_tower.encode(mask3d_data_dict, is_eval=True, device=output_ids.device)
        raw_queries_pos = encoded_state[-1]

        ## => Construct queries from llm_hidden_states
        ref_embeddings = self.extract_ref_hidden_state(output_ids, hidden_states, inference=True)

        ## => Call the grounding tower's `decode()` method for each sample(maybe have different number of ref tokens)
        grounding_outputs = []
        for ref_embedding in ref_embeddings:
            ref_embedding = ref_embedding.unsqueeze(0).to(raw_queries_pos.dtype)
            queries_pos = grounding_cross_attn(ref_embedding, raw_queries_pos.permute(1, 0, 2))

            if queries_pos.shape[1] == 0:
                # => No ref tokens, so we just return None
                grounding_outputs.append(None)
            else:
                ## => Call the grounding tower's `decode()` method
                grounding_output = grounding_tower.decode(
                    encoded_state, mask3d_data_dict, queries_pos.permute(1, 0, 2), is_eval=True
                )
                if grounding_output["pred_bboxes"][0] == -1:
                    # no vaild points
                    grounding_outputs.append(None)
                else:
                    grounding_outputs.append(grounding_output)
        return output_ids, grounding_outputs

    @torch.no_grad()
    def generate(
            self,
            inputs: Optional[torch.Tensor] = None,
            ptv3_data_dict: Dict = None,
            max_new_tokens: int = MAX_NEW_TOKENS,
            **kwargs,
    ) -> Tuple[Union[GenerateOutput, torch.LongTensor], bool]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        kwargs.pop("labels", None)
        kwargs.pop("inputs_embeds", None)

        ## => Prepare for multimodal(insert scene feature)
        if ptv3_data_dict is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                input_embeds,
                _
            ) = self.prepare_for_multimodal(
                input_ids=inputs,
                position_ids=position_ids,
                attention_mask=attention_mask,
                labels=None,
                scene_data_dict=ptv3_data_dict
            )
        else:
            input_embeds = self.get_model().embed_tokens(inputs)

        multimodal = (ptv3_data_dict is not None)

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=input_embeds,
            max_new_tokens=max_new_tokens,
            **kwargs,
        ), multimodal


AutoConfig.register("spatial_reasoner", SpatialReasonerConfig)
AutoModelForCausalLM.register(SpatialReasonerConfig, SpatialReasonerForCausalLM)
