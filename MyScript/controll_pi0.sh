rm -r /home/junlinp/.cache/huggingface/lerobot/junlinp/eval_act_so100_test
python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.fps=30 \
  --control.single_task="Put the object in box A into box B" \
  --control.repo_id=junlinp/eval_act_so100_test \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=5 \
  --robot.leader_arms={} \
  --control.episode_time_s=30 \
  --control.reset_time_s=30 \
  --control.num_episodes=10 \
  --control.push_to_hub=false \
  --control.policy.path=/home/junlinp/checkpoint

