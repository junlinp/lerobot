from collections import deque
from copy import deepcopy
from functools import partial
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from lerobot.common.constants import OBS_ENV, OBS_ROBOT
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.dm.configuration_dm import DMConfig
from lerobot.common.policies.utils import get_device_from_parameters, get_output_shape, populate_queues
from transformers import AutoImageProcessor, ViTModel
import torchvision.transforms as tv_transforms

class DMPolicy(PreTrainedPolicy):
    config_class = DMConfig
    name = "dm"
    def __init__(self, config: DMConfig, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        super().__init__(config)
        self.model = DM(config = config)

        self.loss_fn = torch.nn.MSELoss()

        self.image_transform = tv_transforms.Compose([
            tv_transforms.Resize((224, 224)),
            tv_transforms.Normalize(mean = [0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def reset(self):
        pass

    def get_optim_params(self) -> dict:
        return [
            {
                "params": [
                    p for n, p in self.named_parameters() if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params" :[
                    p for n, p in self.named_parameters() if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            }

        ]
    @torch.no_grad()
    def select_action(self, batch:dict[str, Tensor]) -> Tensor:
        for key in batch.keys():
            if key.find("image") >= 0:
                image_observation = batch[key]
                #for index in range(image_observation.size(0)):
                    #print(f"image observation : {image_observation[index, :, :, :].shape}")
                images = [self.image_transform(image_observation[index, :, :, :]).unsqueeze(0) for index in range(image_observation.size(0))]
                #print(f" concatenate : {torch.concatenate(images, axis = 0).shape}")
                batch[key] = torch.concatenate(images, axis = 0)
        predicted_actions = self.model.forward(batch)
        return predicted_actions.squeeze(1)


    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:

        for key in batch.keys():
            if key.find("image") >= 0:
                image_observation = batch[key]
                #for index in range(image_observation.size(0)):
                    #print(f"image observation : {image_observation[index, :, :, :].shape}")
                images = [self.image_transform(image_observation[index, :, :, :]).unsqueeze(0) for index in range(image_observation.size(0))]
                #print(f" concatenate : {torch.concatenate(images, axis = 0).shape}")
                batch[key] = torch.concatenate(images, axis = 0)

        predicted_actions = self.model.forward(batch)
        loss = self.loss_fn(predicted_actions, batch['action'])
        lost_dict = {}
        return Tensor(loss), lost_dict


class DM(nn.Module):
    def __init__(self, config:DMConfig):
        super().__init__()
        self.config = config

        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.visual_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

        self.condition_action_policy = ConditionMultipleHead(14, 768, 7)

        self.project = torch.nn.Linear(768 * 196, 14)

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        #print(f"forward batch.keys: {batch.keys()}")

        image_observation = []
        for key in batch.keys():
            if key.find("image") >= 0:
                input = batch[key]
                #print(f"image input : {input.shape}")
                #resize_image = self.image_processor(input, return_tensors="pt")
                input_dict = {
                    "pixel_values" : batch[key],
                    "output_hidden_states" : True
                }
                visual_embed = self.visual_encoder(**input_dict)
                # shape (batch, 196, embed_dim)
                #print(f"hidden_states : {visual_embed.hidden_states}") 
                image_observation.append(visual_embed.last_hidden_state[:, 1:, :])

        if len(image_observation) > 1:
            image_embed = torch.concatenate(image_observation, axis = 1) 
        else:
            image_embed = image_observation[0]

        #image_embed = einops.rearrange(image_embed, "batch patch_num d_dim -> batch (patch_num d_dim)", patch_num = 196, d_dim = 768)
        batch_size = image_embed.size(0)
        noised_action = torch.randn((batch_size, self.config.chunk_size, 14)).to(image_embed.device)
        actions = self.condition_action_policy.forward(noised_action, image_embed)
        return actions


class ConditionMAPLayer(nn.Module):

    def __init__(self, query_dim : int, num_heads = 8, dropout_rate:float = 0.1, feedforward_dim:int = 1024):
        super().__init__()
        self.attn_layer = torch.nn.MultiheadAttention(query_dim, num_heads=num_heads, dropout=dropout_rate, batch_first=True)
        self.layer_norm = torch.nn.LayerNorm(query_dim)

        self.feedforward = torch.nn.Sequential(
            torch.nn.Linear(query_dim, feedforward_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(feedforward_dim, query_dim)
        )
        self.feedforward_norm = torch.nn.LayerNorm(query_dim)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        attn_output, _ = self.attn_layer(query, context, context)
        attn_output = self.layer_norm(attn_output + query)

        feedforwad_output = self.feedforward(attn_output)

        return self.feedforward_norm(feedforwad_output + query)


        
class ConditionMultipleHead(nn.Module):
    def __init__(self, query_dim : int, context_dim:int, num_heads = 8, layer_num:int = 6, dropout_rate:float = 0.1):
        super().__init__()

        self.model = torch.nn.ModuleList(
            [ConditionMAPLayer(query_dim, num_heads, dropout_rate) for _ in range(layer_num)]
        )

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.context_project = torch.nn.Linear(context_dim, query_dim)

    
    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        projected_context = self.context_project(context)
        for lyr in self.model:
            query = lyr.forward(query, projected_context)
        return query

