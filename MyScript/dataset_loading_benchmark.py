import time
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def main():
    batch_size = 64
    # Create dataset and dataloader
    dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True
    )

    # Time iteration over full dataset
    start_time = time.perf_counter()
    num_batches = 0
    from tqdm import tqdm
    for _ in tqdm(dataloader, desc="Processing batches", unit="batch"):
        num_batches += 1
    print() # New line after progress bar
    elapsed = time.perf_counter() - start_time

    print(f"Dataset size: {len(dataset)} samples")
    print(f"Number of batches: {num_batches}")
    print(f"Time to iterate: {elapsed:.2f} seconds")
    print(f"Average time per batch: {(elapsed/num_batches)*1000:.2f} ms")
    print(f"Average samples per second: {(len(dataset)/elapsed):.2f}")

if __name__ == "__main__":
    main()
