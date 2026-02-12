#!/bin/bash
# FAIL-PG (Policy Gradient) training script for FLUX
set -x

WORLD_SIZE=${WORLD_SIZE:-1}
RANK=${RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-7189}

NGPUS=$(python3 -c "import torch; print(torch.cuda.device_count())")

export PYTHONPATH="$PYTHONPATH:$(pwd)"
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

torchrun --nnodes=$WORLD_SIZE --nproc_per_node=$NGPUS --node_rank=$RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    fastvideo/train_fail_pg_flux.py \
    --project_name 'fail' \
    --experiment_name 'flux_fail_pg' \
    --seed 42 \
    --pretrained_model_name_or_path ./data/flux \
    --data_path ./data/train_flux_rl_embeddings/videos2caption.json \
    --teacher_image_root ./data/teacher_images \
    --output_dir ./output/flux_fail_pg \
    --discriminator_model_size 2B \
    --qwen3vl_model_path ./data/Qwen3-VL-2B-Instruct \
    --discriminator_learning_rate 2e-6 \
    --discriminator_weight_decay 0. \
    --discriminator_warmup_steps 25 \
    --train_batch_size 1 \
    --sp_size 1 \
    --image_log_interval 10 \
    --learning_rate 1e-5 \
    --max_train_steps 1000 \
    --max_grad_norm 1.0 \
    --rollout_n 3 \
    --teacher_n 1 \
    --num_mc_samples 8 \
    --clip_range 1e-5 \
    --num_epochs 1 \
    --rollout_mini_batch_size 3 \
    --num_sampling_steps 28 \
    --guidance_scale 1.0 \
    --height 512 \
    --width 512 \
    --checkpointing_steps 50 \
    --include_teacher_in_policy \
    --use_kl_regularization \
    --kl_beta 0.05 \
    --reference_model_offload
