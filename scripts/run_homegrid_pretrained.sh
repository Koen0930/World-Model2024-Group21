#! /bin/bash

task=$1
name=$2
device=$3
seed=$4

shift
shift
shift
shift

export CUDA_VISIBLE_DEVICES=$device; python dynalang/train.py \
  --run.script train \
  --logdir logdir/homegrid/$name \
  --use_wandb True \
  --task $task \
  --envs.amount 1 \
  --seed $seed \
  --encoder.mlp_keys token$ \
  --decoder.mlp_keys token$ \
  --decoder.vector_dist onehot \
  --batch_size 16 \
  --batch_length 256 \
  --run.train_ratio 32 \
  --load_wm_ckpt True \
  --run.from_checkpoint /World-Model2024-Group21/logdir/textpt/exp2_42/checkpoint.pkl \
  "$@"
