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
from diffusers import StableDiffusionInstructPix2PixPipeline
import torchvision.transforms as tv_transforms
import math

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

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        self.image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        self.visual_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        #self.future_image_predictor = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        #    "timbrooks/instruct-pix2pix", torch_dtype=torch.float16)

        self.state_project_to_embed = nn.Linear(config.max_action_dim, 768)

    def reset(self):
        pass

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_vision_encoder:
            self.visual_encoder.eval()

    def get_optim_params(self) -> dict:
        return self.parameters()

    def prepare_future_image(self, batch) -> dict:
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch input. At least one expected.(batch: {batch.keys()}) (image_features needed : {self.config.image_features})"
            )

        images = []
        for key in present_img_keys:
            img = batch[key]
            #print(f"img.shape : {img.shape}")
            img = [self.image_transform(img[index, 1, :, :, :]).unsqueeze(0) for index in range(img.size(0))]
            images.append(torch.concatenate(img, axis = 0))
        return images
        pass 


    def pixelValueNormal(self, image: Tensor) -> dict:
        return self.image_transform(image)

    def prepare_state(self, batch:dict) -> Tensor:
        state = batch['observation.state']
        return self.state_project_to_embed(pad_vector(state, self.config.max_state_dim))

    def prepare_images(self, batch:dict) -> list[Tensor]:
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch input. At least one expected.(batch: {batch.keys()}) (image_features needed : {self.config.image_features})"
            )

        images = []
        for key in present_img_keys:
            img = batch[key]
            #print(f"img.shape : {img.shape}")
            img = [self.image_transform(img[index,:, :, :]).unsqueeze(0) for index in range(img.size(0))]
            images.append(torch.concatenate(img, axis = 0))
        return images

    def embed_image(self, image:Tensor) -> Tensor:
            input_dict = {
                "pixel_values" : image,
                "output_hidden_states" : True
            }
            visual_embed = self.visual_encoder(**input_dict)
            # shape (batch, 196, embed_dim)
            return visual_embed.last_hidden_state[:, 1:, :]
    def predict_future_image(self, prompts: list[str], images:list[Tensor]) -> list[Tensor]:
        predict_images = []
        for batch_image in images:
            batch_image = batch_image * 0.5 + 0.5
            list_image = [batch_image[index, :, : , :].squeeze(0) for index in range(batch_image.size(0))]
            torch.concatenate(self.future_image_predictor(prompt = prompts, image = list_image,output_type="pt"), axis = 0)
        return 2.0 * (predict_images - 0.5)


    @torch.no_grad()
    def select_action(self, batch:dict[str, Tensor]) -> Tensor:
        images = self.prepare_images(batch)
        image_observation = [self.embed_image(image) for image in images]
        if len(image_observation) > 1:
            image_embed = torch.concatenate(image_observation, axis = 1) 
        else:
            image_embed = image_observation[0]
        state_embed = self.prepare_state(batch).unsqueeze(1)
        

        image_and_state_embed = torch.concatenate([image_embed, state_embed], axis = 1)
        predicted_actions = self.model.sample_actions(image_and_state_embed=image_and_state_embed)

        original_action_dim = self.config.action_feature.shape[0]
        predicted_actions = predicted_actions[:, :, :original_action_dim]
        predicted_actions = self.unnormalize_outputs({"action" : predicted_actions})["action"]
        return predicted_actions[:, 0, :].squeeze(1)


    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        #print(f"batch.keys() : {batch.keys()}")
        instruction = batch['task']
        #print(f"instruction : {instruction}")
        images = self.prepare_images(batch)
        #future_images = self.prepare_future_image(batch)
        #predict_future_images = self.predict_future_image(instruction,images)

        #reconstruction_loss = 0         

        #for future_image, predict_future_image in zip(future_images, predict_future_images):
        #   reconstruction_loss += torch.mean(future_image - predict_future_image)

        image_observation = [self.embed_image(image) for image in images]

        if len(image_observation) > 1:
            image_embed = torch.concatenate(image_observation, axis = 1) 
        else:
            image_embed = image_observation[0]

        state_embed = self.prepare_state(batch).unsqueeze(1)

        #batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        normalized_actions = batch['action']

        actions_is_pad = batch.get("actions_is_pad")

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound"] = losses.clone()

        image_and_state_embed = torch.concatenate([image_embed, state_embed], axis = 1)
        losses = self.model.forward(image_and_state_embed= image_and_state_embed, actions=pad_vector(normalized_actions, self.config.max_action_dim))

        loss_dict = {"losses_after_forward" : losses.clone()}

        #print(f"losses.shape : {losses.shape}")
        losses = losses[:, :, :normalized_actions.shape[-1]]
        #print(f"unpad_losses.shape : {losses.shape}")
        loss = losses.mean() 
        return loss, loss_dict


def pad_vector(vector:torch.Tensor, new_dim:int) -> torch.Tensor:
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector

def create_sinusoidal_pos_embedding(time:torch.Tensor, dimension:int, min_period: float, max_period: float, device = "cpu") -> Tensor:

        # PE(batch_index, dimension_index) = sin(time[batch_index] / period**(dimension_index / dimension))

        if dimension % 2 != 0:
            raise ValueError(f"dimension ({dimension}) must be divisible by 2")

        if time.ndim != 1:
            raise ValueError("The time tensor is expected to be of shape `(batch_size, )`")

        dtype = torch.float64 
        #
        fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
        period = min_period * (max_period / min_period) ** fraction
        scaling_factor = 1.0 / period * 2 * math.pi

        sin_input = einops.repeat(fraction, "dimension_div_2 -> batch dimension_div_2", batch = time.size(0)) * einops.repeat(time, "batch -> batch dimension_div_2", dimension_div_2 = dimension // 2)
        pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim = 1)
        # shape (batch, dimension)
        return pos_emb

def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)

class DM(nn.Module):
    def __init__(self, config:DMConfig):
        super().__init__()
        self.config = config
        self.condition_action_policy = ConditionMultipleHead(self.config.action_time_proj_width, 768, 8, layer_num=128)
        self.project = torch.nn.Linear(768 * 196, 14)
        self.action_proj_action_time_embed = torch.nn.Linear(self.config.max_action_dim, self.config.action_time_proj_width)
        self.action_time_mlp_in = torch.nn.Linear(self.config.action_time_proj_width * 2, self.config.action_time_proj_width)
        self.action_time_mlp_out = torch.nn.Linear(self.config.action_time_proj_width, self.config.action_time_proj_width)
        self.action_out_proj = torch.nn.Linear(self.config.action_time_proj_width, self.config.max_action_dim)

    def train(self, mode: bool = True):
        super().train(mode)

    def sample_noise(self, shape, device):
        noise =torch.normal(
            mean = 0.0,
            std = 1.0,
            size = shape,
            dtype=torch.float32,
            device= device,
        )
        return noise
    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def forward(self, image_and_state_embed: Tensor, actions: Tensor) -> Tensor:

        timestep = self.sample_time(actions.shape[0], actions.device)
        time_expanded = einops.repeat(timestep, "batch -> batch seq_len action_dim", seq_len = actions.size(1), action_dim = actions.size(2))
        noise = self.sample_noise(actions.shape, actions.device)
        x_t = time_expanded * noise + ( 1 - time_expanded) * actions
        u_t = noise - actions
        batch_size = image_and_state_embed.size(0)

        action_time_embed = self.create_action_time_emb(x_t, timestep)
        diffusion_out = self.condition_action_policy.forward(action_time_embed, image_and_state_embed)
        v_t = self.action_out_proj(diffusion_out)
        losses = torch.nn.functional.mse_loss(u_t, v_t, reduction="none")
        return losses
    
    def sample_actions(self, image_and_state_embed:Tensor) -> Tensor:
        batch_size = image_and_state_embed.shape[0]
        device = image_and_state_embed.device

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        actions_shape = (batch_size, self.config.n_action_steps, self.config.max_action_dim)
        noise = self.sample_noise(actions_shape, device)
        x_t = noise

        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(batch_size)
            #expanded_time = einops.repeat(expanded_time, "batch_size -> batch_size action_seq action_dim", batch_size = batch_size, action_seq = self.config.n_action_steps, action_dim = self.config.max_action_dim)
            v_t = self.denoise_step(
                image_and_state_embed,
                x_t,
                expanded_time
            )

            x_t += dt * v_t
            time += dt
        return x_t


    def create_action_time_emb(self, actions:Tensor, timestep:Tensor) -> Tensor:
        device = actions.device
        dtype = actions.dtype
        time_emb = create_sinusoidal_pos_embedding(timestep, self.config.action_time_proj_width, min_period=4e-3, max_period=4.0, device = device) 
        time_emb = time_emb.type(dtype=dtype)
        time_emb = einops.repeat(time_emb, "batch dim -> batch seq dim", seq = actions.size(1))
        action_emb = self.action_proj_action_time_embed(actions)
        action_time_emb = torch.cat([action_emb, time_emb], dim = 2)
        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = torch.nn.functional.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)
        return action_time_emb

    def denoise_step(self, image_and_state_embed:Tensor, x_t:Tensor, timestep:Tensor)->Tensor:
        action_time_emb = self.create_action_time_emb(x_t, timestep)
        v_t = self.condition_action_policy.forward(action_time_emb, image_and_state_embed) 
        return self.action_out_proj(v_t)





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

