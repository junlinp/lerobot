
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch
from lerobot.common.datasets.utils import cycle
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from typing import Iterator

repo_id = "lerobot/aloha_sim_transfer_cube_scripted"
repo_id = "dragon-95/so100_sorting_3"
root=None

class EpisodeSampler(torch.utils.data.Sampler):
    def __init__(self, dataset: LeRobotDataset, episode_index: int):
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        self.frame_ids = range(from_idx, to_idx)

    def __iter__(self) -> Iterator:
        return iter(self.frame_ids)

    def __len__(self) -> int:
        return len(self.frame_ids)

dataset = LeRobotDataset(repo_id, root=root)
#
dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=EpisodeSampler(dataset, 1),
        pin_memory=True,
        drop_last=False,
    )

#batch = next(dl_iter)
#print(f"batch : {batch}")

#policy_path="/mnt/ml-experiment-data/junlinp/outputs/so100_sorting_3/checkpoints/080000/pretrained_model"
policy_path="/home/junlinp/checkpoint"

kwargs ={}
kwargs["pretrained_name_or_path"] = policy_path

print(f"Load Policy")
policy = PI0Policy.from_pretrained(**kwargs)



index = 0;
for batch in dataloader:
    action = policy.select_action(batch)
    print(f"index {index} Action: {action}")
    index+=1
