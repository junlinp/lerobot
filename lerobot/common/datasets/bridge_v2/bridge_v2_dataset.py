import pickle
import os
import cv2
import numpy as np
import tqdm
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

global_timestamp_index = 0

def check_bridge_files(traj_dir:str):
    file_or_dir = ["obs_dict.pkl", "policy_out.pkl", "lang.txt", "images0"]
    exist = [os.path.exists(os.path.join(traj_dir, item)) for item in file_or_dir]

    for index, value in enumerate(exist):
        if value is False:
            print(f"{os.path.join(traj_dir, file_or_dir[index])} don't exists.")
    return all(exist)

def load_bridge_traj(traj_dir:str):
    # control frequency is 5 Hz
    with open(os.path.join(traj_dir, "obs_dict.pkl"), "rb") as f:
        obs_dict = pickle.load(f)
    with open(os.path.join(traj_dir, "policy_out.pkl"), "rb") as f:
        policy_out = pickle.load(f)
    # lack of ROS package sensor_msgs
    #with open(os.path.join(traj_dir, "agent_data.pkl"), "rb") as f:
    #    agent_data = pickle.load(f)
    with open(os.path.join(traj_dir, "lang.txt"), "r") as f:
        lang_txt = f.readline()[:-1]
    image_dir_path = os.path.join(traj_dir,"images0")
    img_size = len(
        [i for i in filter(lambda path: path.endswith("jpg") ,os.listdir(image_dir_path))]
        )

    imgs = []
    for index in range(img_size):
        img_name  = f"im_{index}.jpg"
        img_path = os.path.join(image_dir_path, img_name)
        # height, width, channel
        # 480, 640, 3
        imgs.append(cv2.imread(img_path))
    return obs_dict, policy_out, lang_txt, imgs

def append_episode(dataset: LeRobotDataset, bridge_traj_path:str):
        global global_timestamp_index
        obs, policy, lang, imgs = load_bridge_traj(bridge_traj_path)

        assert len(obs['time_stamp']) == len(imgs)
        obs_size = len(imgs)
        assert obs_size - 1 == len(policy)
        base_timestamp = 0
        for index in range(obs_size):
            if index == 0:
                # pass this index since the delta time more then 200ms 
                # and the size of policy is obs_size - 1
                base_timestamp = obs['time_stamp'][1]
                continue
            #print(f"obs state shape : {np.array(obs['state'][0], dtype = np.float32).shape}")
            #"episode_index" : 0,
            #"index": 0,
            #"frame_index" : 0,
            #"task_index" : 0,
            #print(f"timestamp shape : {np.array((obs['time_stamp'][0],), dtype=np.float32).shape}")
            #"timestamp" : np.array((obs['time_stamp'][index] - base_timestamp,), dtype=np.float32)
            #print(f"state:{obs['state'][index]}, full_state:{obs['full_state'][index]} desired_state:{obs['desired_state'][index]}, action:{policy[index-1]['actions']}")
            observation = {
                "observation.state": np.array(obs['full_state'][index], dtype = np.float32),
                "img" : imgs[index],
                "timestamp" : np.array((0.2 * global_timestamp_index,), dtype=np.float32)
            }

            action = {
                "action" : np.array(policy[index - 1]['actions'], dtype = np.float32)
            }

            frame = {**observation, **action, "task": lang}
            dataset.add_frame(frame)
            global_timestamp_index += 1
        dataset.save_episode()


if __name__ == "__main__":

    obs, policy, lang, imgs = load_bridge_traj("/mnt/nas/share-all/junlinp/PublicDataSet/bridge/raw/bridge_data_v2/datacol2_folding_table_white_tray/sweep_granular/03/2023-05-23_11-57-28/raw/traj_group0/traj0")

    print(f"obs : {obs.keys()}")
    print(f"obs.joint_effort : {obs['joint_effort'].shape}")
    print(f"obs.qpos : {obs['qpos'].shape}")
    print(f"obs.qvel : {obs['qvel'].shape}")
    print(f"obs.state: {obs['state'].shape}")
    print(f"obs.full_state : {obs['full_state'].shape}")
    #print(f"obs.t_get_obs : {obs['t_get_obs'].shape}")
    print(f"obs.desired_state: {obs['desired_state'].shape}")
    print(f"obs.timestamp : {obs['time_stamp']}")
    print(f"obs.env_done : {obs['env_done']}")
    print(f"obs.eef_transform : {obs['eef_transform'].shape}")

    print(f"policy_out keys : {len(policy)}")
    print(f"lang txt : {lang}")

    timestamp_list = obs['time_stamp']

    base_time = timestamp_list[0]

    #for index, time in enumerate(timestamp_list):
        #print(f"index {index} delta_time : {time - base_time}")
    #print(f"t_get_obs : {obs['t_get_obs']}")
    #print(f"agent_data : {agent_data}")
    #print(f"policy_out keys : {policy[0]}")

    # convert test
    #root="./",

        #"observation.state": 
    features = {
        "observation.state" : {"dtype": "float32", "shape": (7,), "names" :None},
        "timestamp": {"dtype": "float32", "shape": (1,), "names": None},
        "frame_index": {"dtype": "int64", "shape": (1,), "names": None},
        "episode_index": {"dtype": "int64", "shape": (1,), "names": None},
        "index": {"dtype": "int64", "shape": (1,), "names": None},
        "task_index": {"dtype": "int64", "shape": (1,), "names": None},
        "img" : {"dtype" : "image", "shape":(480, 640, 3), "names":["height", "width", "channel"]},
        "action" : {"dtype" :"float32", "shape": (7,), "names" : None}
    }
    dataset = LeRobotDataset.create(
            repo_id="test",
            root="/mnt/nas/share-all/junlinp/PublicDataSet/bridge/bridge_lerobot_dataset_all",
            fps=5,
            use_videos=True,
            features=features,
            tolerance_s=2e-3
    )


    bridge_dataset_root = "/mnt/nas/share-all/junlinp/PublicDataSet/bridge/raw/bridge_data_v2" 

    scenes = os.listdir(bridge_dataset_root)
    #traj_count = 0
    all_traj_path = []
    for scene in scenes:
        scene_path = os.path.join(bridge_dataset_root, scene)
        tasks = os.listdir(scene_path)
        for task in tasks:
            task_path = os.path.join(scene_path, task)
            task_nums = os.listdir(task_path)

            for task_num in task_nums:
                task_num_path = os.path.join(task_path, task_num)
                task_num_dates = os.listdir(task_num_path)

                for task_num_date in task_num_dates:
                    task_num_date_path = os.path.join(task_num_path, task_num_date)
                    task_descriptor_and_raw_data = os.listdir(task_num_date_path)
                    if "raw" in task_descriptor_and_raw_data:
                        raw_path = os.path.join(task_num_date_path, "raw")
                        traj_groups = os.listdir(raw_path)

                        for traj_group in traj_groups:
                            traj_groups_path = os.path.join(raw_path, traj_group)
                            trajs = os.listdir(traj_groups_path)
                            for traj in trajs:
                                traj_path = os.path.join(traj_groups_path, traj)
                                all_traj_path.append(traj_path)
                        

    append_episode_count = 0
    for traj_path in tqdm.tqdm(all_traj_path):
        if check_bridge_files(traj_dir=traj_path):
            append_episode(dataset=dataset, bridge_traj_path=traj_path)
            append_episode_count += 1
    print(f"Convert {append_episode_count}/{len(all_traj_path)} into lerobot dataset")


    

