
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch
from lerobot.common.datasets.utils import cycle
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from typing import Iterator
from MyScript.vit import ViTPolicy
import tqdm

repo_id = "lerobot/aloha_sim_transfer_cube_scripted"
#repo_id = "dragon-95/so100_sorting_3"
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

#print(f"Load Policy")
#policy = PI0Policy.from_pretrained(**kwargs)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

index = 0;
policy = ViTPolicy(output_dim=14)
policy = policy.to(device)
loss = 0
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)


for epoch in range(128):
    mean_loss = 0
    for batch in tqdm.tqdm(dataloader):
        #print(f"batch {batch}")
        #print(f"batch {batch['observation.images.top'].shape}")

        opt.zero_grad()
        x = {
            "imgs": batch['observation.images.top'].to(device)
        }
        action = batch['action'].to(device)
        predict_action = policy.forward(x)
        #print(f"predict_action shape {predict_action.shape}")
        #print(f"action shape {action.shape}")
        loss = torch.mean((action - predict_action)**2)
        mean_loss += loss.item()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch} loss : {mean_loss / len(dataset)}")

torch.save(policy.state_dict(), "vit_policy.pth")

    
