from lerobot.common.datasets import lerobot_dataset

LOCAL_PATH="/mnt/nas/share-all/junlinp/PublicDataSet/OXE"

#
# PI0 subset OXE mixture dataset.
# referenced as OXE magic soup
# details: https://arxiv.org/pdf/2406.09246
#
# lerobot don't contains all of OXE magic soup dataset
# 
#
# OXE repo-id
OXE_REPO = [
"lerobot/taco_play",
"lerobot/jaco_play",
"lerobot/berkeley_cable_routing",
"lerobot/roboturk",
"lerobot/viola",
"lerobot/berkeley_autolab_ur5",
"lerobot/toto",
"lerobot/stanford_hydra_dataset",
"lerobot/austin_buds_dataset",
"lerobot/nyu_franka_play_dataset",
"lerobot/ucsd_kitchen_dataset",
"lerobot/austin_sailor_dataset",
"lerobot/austin_sirius_dataset",
"lerobot/dlr_edan_shared_control",
"lerobot/iamlab_cmu_pickup_insert",
"lerobot/utaustin_mutex",
"lerobot/berkeley_fanuc_manipulation",
"lerobot/cmu_stretch",
"lerobot/fmb",
"lerobot/droid_100",
]

# OXE dataset not included by lerobot
# Fractal
# Kuka
# Bridge
# Language Table
# Furniture Bench Dataset
# BC-Z
# DobbE

if __name__ == "__main__":
    for repo_id in OXE_REPO:
        lerobot_dataset.LeRobotDataset(repo_id=repo_id, root=LOCAL_PATH)