import time
import torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="lerobot/aloha_sim_transfer_cube_human")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--create_video_cache", type=bool, default=False)
    args = parser.parse_args()
    # Create dataset and dataloader
    dataset = LeRobotDataset(args.repo_id, create_video_cache=args.create_video_cache)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
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
