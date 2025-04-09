from __future__ import annotations
import argparse
import os
import json
from functools import reduce
import shutil
import datasets
import numpy as np
import tqdm

def load_jsonl(path:str) -> list:
    with open(path, "r") as f:
        return [json.loads(line) for line in f.readlines()]

def save_jsonl(data: list, path: str) -> None:
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


class LerobotDatasetDirectory():

    def __init__(self, directories_path: str, is_empty_dataset = False) -> None:

        if not os.path.exists(directories_path):
            raise FileNotFoundError(f"Directory path '{directories_path}' does not exist")

        self.data_dir_path = os.path.join(directories_path, "data")
        if not os.path.exists(self.data_dir_path):
            raise FileNotFoundError(f"Data directory path '{self.data_dir_path}' does not exist")

        self.meta_dir_path = os.path.join(directories_path, "meta")
        if not os.path.exists(self.meta_dir_path):
            raise FileNotFoundError(f"Meta directory path '{self.meta_dir_path}' does not exist")

        self.videos_dir_path = os.path.join(directories_path, "videos")
        if not os.path.exists(self.videos_dir_path):
            raise FileNotFoundError(f"Videos directory path '{self.videos_dir_path}' does not exist")

        self.root_dir = directories_path
        self.load_meta()
        self.is_empty_dataset = is_empty_dataset
        if is_empty_dataset:
            self.hf_dataset = None
        else:
            self.hf_dataset = self.load_hf_dataset()

    def get_video_keys(self) -> list:
        features = self.info['features']
        return [key for key, value in features.items() if value['dtype'] == "video"]

    def merge_jsonl_data_and_video(self, other:LerobotDatasetDirectory):
        episode_index_offset = self.info['total_episodes']
        chunks_size = self.info['chunks_size']

        total_add_episode_index = other.info['total_episodes']
        other_chunks_size = self.info['chunks_size']

        video_pattern = self.info['video_path']
        other_video_pattern = other.info['video_path']


        other_episodes_map = { 
            episode_lines['episode_index'] : episode_lines
            for episode_lines in other.episodes
        }

        other_episodes_stats_map = {
             stat_line['episode_index'] : stat_line
            for stat_line in other.episodes_stats
        }

        if self.hf_dataset is None:
            # Create empty dataset with same features as other dataset
            empty_dict = {
                feature_name: [] for feature_name in other.hf_dataset.features
            }
            self.hf_dataset = datasets.Dataset.from_dict(empty_dict, features=other.hf_dataset.features)
            max_episode_index = -1
            max_index = -1
        else:
            max_episode_index = max(self.hf_dataset["episode_index"])
            max_index = max(self.hf_dataset["index"])

        # Create modified versions of episode_index and timestamp for second dataset
        modified_episode_indices = [idx + max_episode_index + 1 for idx in other.hf_dataset["episode_index"]]
        modified_indices = [idx + max_index + 1 for idx in other.hf_dataset['index']]

        # Create new dataset with modified values
        hf_dataset = other.hf_dataset.remove_columns(["episode_index", "index"])
        hf_dataset = hf_dataset.add_column("episode_index", modified_episode_indices)
        hf_dataset = hf_dataset.add_column("index", modified_indices)

        self.hf_dataset = datasets.concatenate_datasets([self.hf_dataset, hf_dataset])
        # copy video
        for other_episode_index in tqdm.tqdm(range(total_add_episode_index), desc="Merging episodes"):
            episode_index = episode_index_offset + other_episode_index
            episode_chunk = episode_index // chunks_size
            other_episode_chunk = other_episode_index // other_chunks_size
            for video_key in other.get_video_keys():
                dest_path = os.path.join(self.root_dir, video_pattern.format(episode_chunk = episode_chunk, video_key = video_key, episode_index = episode_index))
                sour_path = os.path.join(other.root_dir, other_video_pattern.format(episode_chunk = other_episode_chunk, video_key = video_key, episode_index = other_episode_index))
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(sour_path, dest_path)

            item = other_episodes_map[other_episode_index]
            item['episode_index'] += episode_index_offset
            self.episodes.append(item)

            stat_item = other_episodes_stats_map[other_episode_index]
            stat_item['episode_index'] += episode_index_offset

            stat_item['stats']['episode_index']['min'][0] += episode_index_offset
            stat_item['stats']['episode_index']['max'][0] += episode_index_offset
            stat_item['stats']['episode_index']['mean'][0] += episode_index_offset
            self.episodes_stats.append(stat_item)

        self.info['total_episodes'] += other.info['total_episodes']

        total_task_size = self.info['total_tasks']
        other_total_task_size = other.info['total_tasks']

        other_task_map = {
            task_line['task_index'] : task_line
            for task_line in other.tasks
        }

        for other_task_index in range(other_total_task_size):
            item = other_task_map[other_task_index]
            item['task_index'] += total_task_size
            self.tasks.append(item)

        self.info['total_tasks'] += other.info['total_tasks']
        self.info['total_chunks'] = self.info['total_episodes'] // self.info['chunks_size'] + int(self.info['total_episodes'] % self.info['chunks_size'] != 0)
        self.info['splits'] = {
            "train": f"0:{self.info['total_episodes']}"
        }
        self.info['total_videos'] += other.info['total_videos']
            
    def validate_features(self, other:LerobotDatasetDirectory):

        keys = self.info['features'].keys()
        other_keys = self.info['features'].keys()
        len(keys) == len(other_keys)
        for key in other_keys:
            if key not in keys:
                raise ValueError(f"Feature {key} in {other.root_dir} not found in dataset {self.root_dir}")

            if self.info['features'][key]['shape'] != other.info['features'][key]['shape']:
                raise ValueError(f"Feature {key} shape mismatch: {self.info['features'][key]['shape']} != {other.info['features'][key]['shape']}")

            if self.info['features'][key]['dtype'] != other.info['features'][key]['dtype']:
                raise ValueError(f"Feature {key} dtype mismatch: {self.info['features'][key]['dtype']} != {other.info['features'][key]['dtype']}")
        


    def merge(self, other:LerobotDatasetDirectory):
        if self.is_empty_dataset and not other.is_empty_dataset:
            if self.info['codebase_version'] != other.info['codebase_version']:
                raise ValueError(f"Codebase versions do not match: {self.info['codebase_version']} != {other.info['codebase_version']}")
            self.info['features'] = other.info['features']
            self.info['robot_type'] = other.info['robot_type']
            self.info['total_frames'] += other.info['total_frames']
            self.info['fps'] = other.info['fps']
            self.is_empty_dataset = False
        elif not self.is_empty_dataset and not other.is_empty_dataset:
            if self.info['codebase_version'] != other.info['codebase_version']:
                raise ValueError(f"Codebase versions do not match: {self.info['codebase_version']} != {other.info['codebase_version']}")
            if self.info['robot_type'] != other.info['robot_type']:
                raise ValueError(f"Robot types do not match: {self.info['robot_type']} != {other.info['robot_type']}")
            if self.info['fps'] != other.info['fps']:
                raise ValueError(f"FPS values do not match: {self.info['fps']} != {other.info['fps']}")


        self.validate_features(other= other)
        self.merge_jsonl_data_and_video(other=other)
        

            

    def load_meta(self):
        with open(os.path.join(self.meta_dir_path, "info.json"), "r") as f:
            self.info = json.load(f)
        self.episodes = load_jsonl(os.path.join(self.meta_dir_path, "episodes.jsonl"))
        self.episodes_stats = load_jsonl(os.path.join(self.meta_dir_path, "episodes_stats.jsonl"))
        self.tasks = load_jsonl(os.path.join(self.meta_dir_path, "tasks.jsonl"))

    def save(self):
        with open(os.path.join(self.meta_dir_path, "info.json"), "w") as f:
            json.dump(self.info, f)
        save_jsonl(self.episodes, os.path.join(self.meta_dir_path, "episodes.jsonl"))
        save_jsonl(self.episodes_stats, os.path.join(self.meta_dir_path, "episodes_stats.jsonl"))
        save_jsonl(self.tasks, os.path.join(self.meta_dir_path, "tasks.jsonl"))
            # Calculate chunk size and total chunks needed
        chunk_size = self.info['chunks_size']
        total_episodes = self.info['total_episodes']
        total_chunks = self.info['total_chunks']
        # Save data in chunks
        for chunk in range(total_chunks):
            chunk_start = chunk * chunk_size
            chunk_end = min((chunk + 1) * chunk_size, total_episodes)
            # Filter episodes for this chunk
            chunk_mask = [idx >= chunk_start and idx < chunk_end for idx in self.hf_dataset["episode_index"]]
            chunk_dataset = self.hf_dataset.filter(lambda _, idx: chunk_mask[idx], with_indices=True)

            # Save each episode in chunk to separate parquet file
            for episode_idx in range(chunk_start, chunk_end):
                episode_mask = [idx == episode_idx for idx in chunk_dataset["episode_index"]]
                episode_dataset = chunk_dataset.filter(lambda _, idx: episode_mask[idx], with_indices=True)
                if len(episode_dataset) > 0:
                    episode_path = os.path.join(self.root_dir, self.info['data_path'].format(episode_chunk=chunk,episode_index = episode_idx))
                    # Save chunk to parquet file
                    chunk_path = os.path.dirname(episode_path)
                    os.makedirs(chunk_path, exist_ok=True)

                    episode_dataset.to_parquet(episode_path)

        #print(f"self.hf_dataset : {self.hf_dataset['episode_index'][23468]}")

    def load_hf_dataset(self):
        return datasets.load_dataset("parquet", data_dir=self.data_dir_path, split="train")
    
    @classmethod
    def create_empty_dataset(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        data_dir = os.path.join(output_dir, "data")
        meta_dir = os.path.join(output_dir, "meta") 
        videos_dir = os.path.join(output_dir, "videos")

        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True) 
        os.makedirs(videos_dir, exist_ok=True)

        # Create empty meta files
        info = {
                "codebase_version": "v2.1",
                "robot_type": "null",
                "total_episodes": 0,
                "total_frames": 0,
                "total_tasks": 0,
                "total_videos": 0,
                "total_chunks": 0,
                "chunks_size": 1000,
                "fps": 0,
                "splits": {
                    "train": "0:50"
                },
                "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
                "features": {}
        }
        
        with open(os.path.join(meta_dir, "info.json"), "w") as f:
            json.dump(info, f)

        save_jsonl([], os.path.join(meta_dir, "episodes.jsonl"))
        save_jsonl([], os.path.join(meta_dir, "episodes_stats.jsonl"))
        save_jsonl([], os.path.join(meta_dir, "tasks.jsonl"))
        
        return  LerobotDatasetDirectory(output_dir, True)

def test():
    hf_dataset = datasets.load_dataset("parquet", data_dir="/mnt/nas/share-all/junlinp/lerobot_dataset_bak/pick_up_pen_50_episodes/data", split="train")
    print(hf_dataset['timestamp'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge multiple datasets into one")
    parser.add_argument("--source-dirs", nargs="+", required=True, help="List of source dataset directories to merge")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged dataset")
    args = parser.parse_args()
    source_dirs = args.source_dirs
    output_dir = args.output_dir
    lerobot_datasets = [LerobotDatasetDirectory(directory) for directory in source_dirs]
    merged_dataset = LerobotDatasetDirectory.create_empty_dataset(output_dir=output_dir)    
    for dataset in lerobot_datasets:
        merged_dataset.merge(dataset)
    merged_dataset.save()