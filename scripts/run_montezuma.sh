#! /bin/bash

# モンテズマの逆襲用のトレーニングスクリプト
task=montezuma
name=$1
device=$2
seed=$3

shift
shift
shift

export CUDA_VISIBLE_DEVICES=$device; python dynalang/train.py \
  --run.script train \
  --run.log_keys_video log_image \
  --logdir /workspace/logdir/atari/${task}_${name} \
  --use_wandb False \
  --task gym_ALE/MontezumaRevenge-v5 \
  --envs.amount 4 \
  --dataset_excluded_keys info \
  --seed $seed \
  --decoder.image_dist mse \
  --encoder.cnn_keys image \
  --decoder.cnn_keys image \
  --batch_size 32 \
  --batch_length 128 \
  --run.train_ratio 64 \
  "$@"
