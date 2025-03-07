
#import simpler_env
#from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
#task_name = "widowx_carrot_on_plate"  # @param ["google_robot_pick_coke_can", "google_robot_move_near", "google_robot_open_drawer", "google_robot_close_drawer", "widowx_spoon_on_towel", "widowx_carrot_on_plate", "widowx_stack_cube", "widowx_put_eggplant_in_basket"]

#import simpler_env
#from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
#import mediapy as media

from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy, PI0Config
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode

import numpy as np
import torch

input_features = {}
input_features["observation.image"] = PolicyFeature(FeatureType.VISUAL, (512, 640, 3))
output_features = {}
output_features["action"] = PolicyFeature(FeatureType.ACTION, (7,))
normalize_mapping = {}
normalize_mapping["action"] = NormalizationMode.MIN_MAX
data_state = {}

pi0_config = PI0Config(input_features=input_features, output_features=output_features, normalization_mapping=normalize_mapping)
print(f"cuda available : {torch.cuda.is_available()}")
device = torch.device("cpu")

policy  = PI0Policy.from_pretrained("/home/junlinp/checkpoint", config=pi0_config)
policy.to(device)


#print(f"policy input features : {policy.config.input_features}")
#env = simpler_env.make('google_robot_pick_coke_can')
#obs, reset_info = env.reset()
#instruction = env.get_language_instruction()
##print(f"Obs : {obs}")
##print(f"reset_info : {reset_info}")
##print("Reset info", reset_info)
#print("Instruction", instruction)
#print(f"Action space: {env.action_space}")
#print(f"action space : {env.action_space.low}")
#print(f"action space : {env.action_space.high}")
#print(f"observation space : {env.observation_space}")

#data_state["action"] = {
   #"min": torch.tensor(env.action_space.low),
   #"max" : torch.tensor(env.action_space.high),
#}
#done, truncated = False, False

#image = get_image_from_maniskill2_obs_dict(env, obs)
#images = [image]
##print(f"images : {image} shape : {image.shape}")

#while not (done or truncated):
   ## action[:3]: delta xyz; action[3:6]: delta rotation in axis-angle representation;
   ## action[6:7]: gripper (the meaning of open / close depends on robot URDF)
   #image = get_image_from_maniskill2_obs_dict(env, obs)
   #state = np.concatenate((obs['agent']['qpos'], obs['agent']['qvel'], obs['agent']['controller']['gripper']['target_qpos']))
   #input_data = {}
   #input_data["task"] = [instruction]
   ## convert to CHW
   #input_data["observation.image"] = torch.tensor(np.asarray([image]), dtype=torch.float32).permute((0, 3, 1, 2))
   #input_data["observation.state"] = torch.tensor(state, dtype=torch.float32).view((1, -1))
   #policy_action = policy.select_action(input_data)
   #obs, reward, done, truncated, info = env.step(policy_action.detach().numpy().squeeze()) # for long horizon tasks, you can call env.advance_to_next_subtask() to advance to the next subtask; the environment might also autoadvance if env._elapsed_steps is larger than a threshold

   ##action = env.action_space.sample() # replace this with your policy inference
   ##obs, reward, done, truncated, info = env.step(action) # for long horizon tasks, you can call env.advance_to_next_subtask() to advance to the next subtask; the environment might also autoadvance if env._elapsed_steps is larger than a threshold
   #new_instruction = env.get_language_instruction()
   #images.append(image)
   #if new_instruction != instruction:
      ## for long horizon tasks, we get a new instruction when robot proceeds to the next subtask
      #instruction = new_instruction
      #print("New Instruction", instruction)

#episode_stats = info.get('episode_stats', {})
#print("Episode stats", episode_stats)
#media.write_video(f"./output.mp4", images, fps = 5)
