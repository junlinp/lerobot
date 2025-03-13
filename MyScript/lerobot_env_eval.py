# example.py
import imageio
import gymnasium as gym
import numpy as np
import gym_aloha
import einops
from MyScript.vit import ViTPolicy
import torch
import tqdm

#env = gym.make("gym_aloha/AlohaInsertion-v0")
env = gym.make("gym_aloha/AlohaTransferCube-v0")

observation, info = env.reset()
frames = []

print(f"action.space : {env.action_space}")
print(f"observation.space : {env.observation_space}")

def ObservationTransform(img:np.array):
    return einops.repeat(einops.rearrange(img, "h w c ->c h w"), "c h w -> b c h w", b = 1) / 255.0


policy = ViTPolicy(14 * 50)
policy.load_state_dict(torch.load("./vit_policy.pth", weights_only=True))
policy.eval()

device = torch.device("cpu")
policy.to(device)
for step in tqdm.tqdm(range(1000)):
    x = {
        "imgs":torch.tensor(ObservationTransform(observation['top']), dtype=torch.float32).to(device)
    }
    action = policy.predict_action(x)
    
    #action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action.cpu().squeeze(0).detach().numpy())
    image = env.render()
    frames.append(image)

    if terminated or truncated:
        observation, info = env.reset()
        break

env.close()
imageio.mimsave("example.mp4", np.stack(frames), fps=25)