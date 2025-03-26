  #--robot.leader_arms={} \
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.display_cameras=false \
  --control.fps=30 \
  --control.single_task="pick the pen into basket" \
  --control.repo_id=junlinp/eval_pi0_so100_200_episodes \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=10 \
  --control.episode_time_s=40 \
  --control.reset_time_s=30 \
  --control.num_episodes=5 \
  --control.push_to_hub=false \
  --control.policy.path=/home/dm/checkpoint/so100_200_episodes_60k

