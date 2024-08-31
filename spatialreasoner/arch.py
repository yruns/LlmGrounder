"""
File: arch.py
Date: 2024/8/17
Author: yruns
"""
from abc import ABC, abstractmethod
from typing import *

import torch
from torch import nn

from spatialreasoner.builder import build_pointcloud_tower, build_grounding_tower
from staticvars.const import SCENE_TOKEN_INDEX, IGNORE_INDEX


class SpatialReasonerMetaModel(nn.Module):

    def __init__(self, config):
        super(SpatialReasonerMetaModel, self).__init__(config)
        self.config = config
        self.pointcloud_tower = None
        self.pointcloud_projector = None
        self.grounding_tower = None
        self.grounding_projector = None

    def initialize_pointcloud_tower(self, hparms):
        pointcloud_tower_cfg = getattr(hparms, "pointcloud_tower_cfg")
        self.pointcloud_tower = build_pointcloud_tower(pointcloud_tower_cfg)
        self.pointcloud_projector = nn.Linear(
            hparms.pointcloud_output_dim,
            self.config.hidden_size,
        )

    def initialize_grounding_tower(self, hparms):
        grounding_tower_cfg = getattr(hparms, "grounding_tower_cfg")
        self.grounding_tower = build_grounding_tower(grounding_tower_cfg)
        self.grounding_projector = nn.Linear(
            self.config.hidden_size,
            grounding_tower_cfg["model"]["hidden_dim"]
        )


    def get_pointcloud_tower(self):
        return getattr(self, "pointcloud_tower", None)

    def get_pointcloud_projector(self):
        return getattr(self, "pointcloud_projector", None)

    def reset_pointcloud_tower_precision(self, precision: torch.dtype):
        setattr(self, "pointcloud_tower", self.get_pointcloud_tower().to(dtype=precision))


class SpatialReasonerMetaForCausalLM(nn.Module):
    config: Dict

    @abstractmethod
    def get_model(self):
        pass

    def reset_detector_precision(self, precision: torch.dtype):
        self.get_model().reset_pointcloud_tower_precision(precision)

    def get_pointcloud_tower(self):
        return self.get_model().get_pointcloud_tower()

    def get_pointcloud_projector(self):
        return self.get_model().get_pointcloud_projector()

    def encode_scene(self, scene_data_dict: Dict, device):
        dtype = getattr(self.config, "compute_dtype")
        scene_features = (
             self.get_pointcloud_tower()
             .encode_scene(scene_data_dict)
             .to(dtype)
        )
        return self.get_pointcloud_projector()(scene_features)

    def prepare_for_multimodal(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            scene_data_dict: Dict = None
    ):
        # pointcloud_tower = self.get_detector()
        # assert pointcloud_tower is not None, "Please provide a pointcloud_tower"

        device = input_ids.device
        embed_tokens_fn = self.get_model().embed_tokens

        ### => Encode Scene
        scene_enc_features = self.encode_scene(scene_data_dict, device)

        # if position_ids is None:
        #     position_ids = torch.range(0, input_ids.shape[1], device=device)

        ### => Remove the padding by attention_mask
        mask_tensor = attention_mask.bool()
        input_ids: List[torch.Tensor] = [cur_input_ids[mask] for cur_input_ids, mask in zip(input_ids, mask_tensor)]
        labels: List[torch.Tensor] = [cur_labels[mask] for cur_labels, mask in zip(labels, mask_tensor)]

        ### => Insert the scene features into input_embeds
        multimodal_input_embeds = []
        multimodal_labels = []
        scene_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            cur_labels = labels[batch_idx]
            num_scenes = torch.sum(cur_input_ids == SCENE_TOKEN_INDEX)

            if num_scenes == 0:
                cur_input_embeds = embed_tokens_fn(cur_input_ids)
                multimodal_input_embeds.append(cur_input_embeds)
                multimodal_labels.append(cur_labels)
                continue

            scene_token_indices = [-1] + torch.where(cur_input_ids == SCENE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_no_scene_snippets = []
            cur_labels_no_scene_snippets = []

            for i in range(len(scene_token_indices) - 1):
                cur_input_ids_no_scene_snippets.append(
                    cur_input_ids[scene_token_indices[i] + 1: scene_token_indices[i + 1]]
                )
                cur_labels_no_scene_snippets.append(
                    cur_labels[scene_token_indices[i] + 1: scene_token_indices[i + 1]]
                )

            cur_input_embeds_no_scene_snippets = [
                embed_tokens_fn(cur_input_ids_no_scene_snippets[i]) for i in range(len(cur_input_ids_no_scene_snippets))
            ]
            cur_multimodal_embeds = []
            cur_multimodal_labels = []

            for i in range(num_scenes + 1):  # `(num_scenes + 1)` is equal to `len(cur_input_embeds_no_scene_snippets)`
                cur_multimodal_embeds.append(cur_input_embeds_no_scene_snippets[i])
                cur_multimodal_labels.append(cur_labels_no_scene_snippets[i])

                if i < num_scenes:
                    cur_scene_features = scene_enc_features[scene_idx]
                    cur_multimodal_embeds.append(cur_scene_features)
                    cur_multimodal_labels.append(torch.full((cur_scene_features.shape[0],), IGNORE_INDEX, device=device, dtype=torch.long))
                    scene_idx += 1

            cur_multimodal_embeds = torch.cat(cur_multimodal_embeds, dim=0)
            cur_multimodal_labels = torch.cat(cur_multimodal_labels, dim=0)
            multimodal_input_embeds.append(cur_multimodal_embeds)
            multimodal_labels.append(cur_multimodal_labels)

        ### => Truncate the input_embeds
        model_max_len = getattr(self.config, 'tokenizer_model_max_length')
        batch_max_len, batch_size = 0, len(multimodal_input_embeds)
        multimodal_input_embeds_truncated, multimodal_labels_truncated = [], []
        for i in range(batch_size):
            multimodal_input_embeds_truncated.append(multimodal_input_embeds[i][:model_max_len])
            multimodal_labels_truncated.append(multimodal_labels[i][:model_max_len])
            batch_max_len = max(batch_max_len, multimodal_input_embeds[i].shape[0])

        ### => Combine the input_embeds and labels
        embedding_dim = multimodal_input_embeds[0].shape[1]
        multimodal_input_embeds_padded = torch.zeros((batch_size, batch_max_len, embedding_dim),
                                                    dtype=multimodal_input_embeds_truncated[0].dtype, device=device)
        multimodal_labels_padded = torch.ones((batch_size, batch_max_len), dtype=torch.long, device=device) * IGNORE_INDEX
        multimodal_attention_mask = torch.zeros((batch_size, batch_max_len), dtype=torch.bool, device=device)

        padding_strategy = getattr(self.config, 'tokenizer_padding_side', 'right')
        for i in range(batch_size):
            if padding_strategy == 'left':
                multimodal_input_embeds_padded[i, -multimodal_input_embeds_truncated[i].shape[0]:] = \
                    multimodal_input_embeds_truncated[i]
                multimodal_labels_padded[i, -multimodal_labels_truncated[i].shape[0]:] = \
                    multimodal_labels_truncated[i]
                multimodal_attention_mask[i, -multimodal_input_embeds_truncated[i].shape[0]:] = True
            else:
                multimodal_input_embeds_padded[i, :multimodal_input_embeds_truncated[i].shape[0]] = \
                    multimodal_input_embeds_truncated[i]
                multimodal_labels_padded[i, :multimodal_labels_truncated[i].shape[0]] = \
                    multimodal_labels_truncated[i]
                multimodal_attention_mask[i, :multimodal_input_embeds_truncated[i].shape[0]] = True

        ### => input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels
        return None, position_ids, attention_mask, multimodal_input_embeds_padded, multimodal_labels_padded
