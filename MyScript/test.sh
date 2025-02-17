
DATA_SET="
lerobot/aloha_sim_transfer_cube_human
lerobot/aloha_sim_transfer_cube_scripted
"


#python lerobot/scripts/train.py --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human --policy.path=lerobot/act_aloha_sim_transfer_cube_human --steps 10000

#python lerobot/scripts/train.py \
    #--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    #--policy.type=diffusion \
    #--steps 10000

# square image required.
#python lerobot/scripts/train.py \
    #--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    #--policy.type=tdmpc \
    #--steps 10000

#python lerobot/scripts/train.py \
    #--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    #--policy.type=vqbet \
    #--steps=10000

# for transfer task
#python lerobot/scripts/eval.py \
    #--policy.path=/home/junlinp/test/lerobot/outputs/train/2025-02-14/15-30-00_vqbet/checkpoints/last/pretrained_model \
    #--env.type=aloha \
    #--eval.batch_size=8 \
    #--eval.n_episodes=128 \
    #--use_amp=false \
    #--device=cuda \
    #--env.task=AlohaTransferCube-v0


python lerobot/scripts/eval.py \
    --policy.path=lerobot/pi0 \
    --env.type=aloha \
    --eval.batch_size=8 \
    --eval.n_episodes=64 \
    --use_amp=false \
    --device=cuda \
    --env.task=AlohaTransferCube-v0


#python lerobot/scripts/eval.py \
    #--policy.path=lerobot/pi0 \
    #--env.type=aloha \
    #--eval.batch_size=8 \
    #--eval.n_episodes=128 \
    #--use_amp=false \
    #--device=cuda \
    #--env.task=AlohaInsertion-v0

