  #--robot.leader_arms={} \
  #--control.repo_id=junlinp/eval_pi0_20k_all_objects_merged_8 \

  #--control.policy.path=/mnt/nas/share-all/junlinp/lerobot_checkoutpoints/pi0_20k_all_objects_merged/checkpoints/200000/pretrained_model

python lerobot/scripts/control_robot.py \
  --robot.type=so100 \
  --control.type=record \
  --control.display_cameras=false \
  --control.fps=30 \
  --control.single_task="pick the all objects from right box into left box" \
  --control.repo_id=junlinp/eval_pi0_slower_operations_30_episodes_modified_PID_merged_2B_step_40k_checkpoint \
  --control.tags='["tutorial"]' \
  --control.warmup_time_s=10 \
  --control.episode_time_s=40 \
  --control.reset_time_s=10 \
  --control.num_episodes=10 \
  --control.push_to_hub=false \
  --control.policy.path=/mnt/nas/share-all/junlinp/lerobot_checkoutpoints/pi0_slower_operations_30_episodes_modified_PID_merged_2B_step/checkpoints/0040000/pretrained_model

