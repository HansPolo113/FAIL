#!/bin/bash
# FAIL-PD (Pathwise Derivative) training script for FLUX
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
    fastvideo/train_fail_pd_flux.py \
    --project_name 'fail' \
    --experiment_name 'flux_fail_pd' \
    --seed 42 \
    --pretrained_model_name_or_path ./data/flux \
    --data_path ./data/train_flux_rl_embeddings/videos2caption.json \
    --teacher_image_root ./data/teacher_images \
    --output_dir ./output/flux_fail_pd \
    --discriminator_model_size 2B \
    --qwen3vl_model_path ./data/Qwen3-VL-2B-Instruct \
    --discriminator_freeze_backbone \
    --discriminator_learning_rate 2e-6 \
    --discriminator_weight_decay 0. \
    --discriminator_warmup_steps 25 \
    --train_batch_size 1 \
    --sp_size 1 \
    --image_log_interval 10 \
    --learning_rate 1e-5 \
    --max_train_steps 2000 \
    --max_grad_norm 1.0 \
    --rollout_n 3 \
    --teacher_n 1 \
    --rollout_mini_batch_size 3 \
    --num_mc_samples 2 \
    --noise_type original \
    --timestep_type scheduler \
    --num_sampling_steps 28 \
    --guidance_scale 1.0 \
    --include_teacher_in_dis_update \
    --height 512 \
    --width 512 \
    --checkpointing_steps 50
