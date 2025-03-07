
DATA_SET="
lerobot/aloha_sim_transfer_cube_human
lerobot/aloha_sim_transfer_cube_scripted
"
#xvfb-run python lerobot/scripts/train.py \
    #--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
    #--policy.type=pi0 \
    #--device cuda \
    #--steps 10000

rm -r /mnt/data/junlinp/outputs/pi0_cube_scripted

xvfb-run python lerobot/scripts/train.py \
    --dataset.repo_id=lerobot/aloha_sim_transfer_cube_scripted \
    --policy.type=pi0 \
    --device cuda \
    --output_dir=/mnt/data/junlinp/outputs/pi0_cube_scripted \
    --steps 80000

xvfb-run python lerobot/scripts/eval.py \
    --policy.path=/mnt/data/junlinp/outputs/pi0_cube_scripted/checkpoints/080000/pretrained_model \
    --env.type=aloha \
    --eval.batch_size=16 \
    --eval.n_episodes=128 \
    --use_amp=false \
    --device=cuda \
    --output_dir=/mnt/data/junlinp/outputs/evals \
    --env.task=AlohaTransferCube-v0


    #--policy.path=lerobot/act_aloha_sim_transfer_cube_human \
#python lerobot/scripts/eval.py \
    #--policy.path=/mnt/ml-experiment-data/junlinp/outputs/act_cube_scripted/checkpoints/080000/pretrained_model \
    #--eval.batch_size=8 \
    #--eval.n_episodes=64 \
    #--use_amp=false \
    #--device=cuda \
    #--env.type=aloha \
    #--env.task=AlohaTransferCube-v0

#python lerobot/scripts/eval.py \
    #--policy.path=lerobot/pi0 \
    #--env.type=aloha \
    #--eval.batch_size=8 \
    #--eval.n_episodes=64 \
    #--use_amp=false \
    #--device=cuda \
    #--output_dir=/mnt/data/junlinp/outputs/pi0_cube_scripted 
    #--env.task=AlohaTransferCube-v0


#python lerobot/scripts/eval.py \
    #--policy.path=lerobot/pi0 \
    #--env.type=aloha \
    #--eval.batch_size=8 \
    #--eval.n_episodes=128 \
    #--use_amp=false \
    #--device=cuda \
    #--env.task=AlohaInsertion-v0

#xvfb-run python lerobot/scripts/train.py \
    #--dataset.repo_id=dragon-95/so100_sorting_3 \
    #--policy.type=pi0 \
    #--device cuda \
    #--output_dir=/mnt/data/junlinp/outputs/pi0_so100 \
    #--steps 80000
