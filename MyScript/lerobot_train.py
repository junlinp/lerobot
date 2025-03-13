
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
import torch
from lerobot.common.datasets.utils import cycle
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy
from typing import Iterator
from MyScript.vit import ViTPolicy
import tqdm
import einops

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
ds_meta = LeRobotDatasetMetadata(repo_id, root=root)

chunk_size = 50
delta_timestamps = {"action": [i / ds_meta.fps for i in range(chunk_size)]}

dataset = LeRobotDataset(repo_id, root=root, delta_timestamps=delta_timestamps)
#
dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        #sampler=EpisodeSampler(dataset, 1),
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
policy = ViTPolicy(chunk_size=chunk_size,action_dim=14, image_width=640, image_height=480)
policy = policy.to(device)
loss = 0
opt = torch.optim.Adam(policy.parameters(), lr=1e-3)


def continuous_loss(pred_value: torch.Tensor, ground_truth_value: torch.Tensor):
    
    assert pred_value.size(1) == chunk_size * 14
    assert ground_truth_value.size(1) == chunk_size * 14

    loss = torch.square(pred_value - ground_truth_value)
    return torch.mean(loss)

for epoch in range(24):
    mean_loss = 0
    for batch in tqdm.tqdm(dataloader):
        #print(f"batch {batch}")
        #print(f"batch {batch['observation.images.top'].shape}")
        #print(f"episode_index {batch['episode_index']}, index {batch['index']}, frame_index {batch['frame_index']}")
        #print(f"action.shape {batch['action'].shape}")
        opt.zero_grad()
        batch_data = {
            "imgs": batch['observation.images.top'].to(device),
            "action" : batch['action'].to(device),
        }
        #action = batch['action'].to(device)
        #predict_action = policy.forward(x)
        #print(f"predict_action shape {predict_action.shape}")
        #print(f"action shape {action.shape}")
        #loss = continuous_loss(predict_action, action)
        loss = policy.compute_loss(batch_data)
        mean_loss += loss.item()
        loss.backward()
        opt.step()
    print(f"Epoch {epoch} loss : {mean_loss / len(dataset)}")

torch.save(policy.state_dict(), "vit_policy.pth")

    
