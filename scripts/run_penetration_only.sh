#!/bin/bash
# Quick test: run iARCS with non-penetration reward only for a few stages.
# Verifies that the reward increases over training.

python train.py \
    exp_name="test_penetration_only" \
    seed=42 \
    sample.batch_size=16 \
    train.batch_size=16 \
    sample.num_batches_per_epoch=4 \
    pipeline.stage_cnt=5 \
    threed_scene_layout=true \
    universal_rewards=true \
    universal_reward_components=[penetration] \
    wandb.enabled=false \
    train.incremental_training=true \
    train.only_train_steps=15 \
    sample.num_steps=20
