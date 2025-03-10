
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import torch
from lerobot.common.datasets.utils import cycle
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

repo_id = "lerobot/aloha_sim_transfer_cube_scripted"
repo_id = "dragon-95/so100_sorting_3"
root=None
dataset = LeRobotDataset(repo_id, root=root)
#
dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        pin_memory=True,
        drop_last=False,
    )

dl_iter = cycle(dataloader)
batch = next(dl_iter)
print(f"batch : {batch}")

policy_path="/mnt/ml-experiment-data/junlinp/outputs/pi0_cube_scripted/checkpoints/080000/pretrained_model"

kwargs ={}
kwargs["pretrained_name_or_path"] = policy_path

policy = PI0Policy()
print(f"Load Policy")
policy.from_pretrained(**kwargs)
