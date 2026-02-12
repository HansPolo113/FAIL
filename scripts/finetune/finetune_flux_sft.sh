#!/bin/bash
# SFT (Supervised Fine-Tuning) training script for FLUX
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
    fastvideo/train_sft_flux.py \
    --wandb_project_name 'fail' \
    --wandb_experiment_name 'flux_sft' \
    --seed 42 \
    --pretrained_model_name_or_path ./data/flux \
    --data_json_path ./data/train_flux_rl_embeddings/videos2caption.json \
    --teacher_image_root ./data/teacher_images \
    --output_dir ./output/flux_sft \
    --dataloader_num_workers 1 \
    --train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --image_size 512 \
    --max_images_per_prompt 1 \
    --checkpointing_steps 50 \
    --max_train_steps 200 \
    --num_epochs 3 \
    --learning_rate 1e-5 \
    --lr_warmup_steps 0 \
    --gradient_checkpointing \
    --weight_decay 0 \
    --guidance_scale 1.0 \
    --cfg 0.1
