
DATA_SET="
lerobot/aloha_sim_transfer_cube_human
lerobot/aloha_sim_transfer_cube_scripted
"


python lerobot/scripts/train.py --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --policy.path=lerobot/act_aloha_sim_transfer_cube_human

# for transfer task
python lerobot/scripts/eval.py --policy.path=lerobot/act_aloha_sim_transfer_cube_human --env.type=aloha --eval.batch_size=10 --eval.n_episodes=10 --use_amp=false --device=cuda --env.task=AlohaTransferCube-v0