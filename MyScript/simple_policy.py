from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

from lerobot.configs.types import FeatureType
from huggingface_hub import login
#login()

def main():
    output_directory = Path("outputs/train/example_simple_policy")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    device = torch.device("cpu")
    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.

    training_steps = 5000
    log_freq = 1

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata("lerobot/aloha_sim_transfer_cube_human")
    features = dataset_to_policy_features(dataset_metadata.features)
    print(f"features : {features}")
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    print(f"output_features : {output_features}")
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    print(f"input_features : {input_features}")

    config = PI0Config(input_features= input_features, output_features=output_features)
    #config.resize_imgs_with_padding = (480, 640)
    #config.adapt_to_pi_aloha = True
    #print(f"stats : {dataset_metadata.stats}")
    policy = PI0Policy(config=config, dataset_stats=dataset_metadata.stats)
    #policy = PI0Policy.from_pretrained("lerobot/pi0")

    policy.train()
    policy.to(device)
    dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")
    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            #batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            pre_batch = {}
            for k, v in batch.items():
                if type(v) is torch.tensor:
                    pre_batch[k] = v.to(device)
                else:
                    pre_batch[k] = v
            batch = pre_batch
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)





if __name__ == "__main__":
    main()

    