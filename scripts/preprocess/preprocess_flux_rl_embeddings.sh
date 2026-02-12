#!/bin/bash
# Preprocess FLUX RL embeddings from parquet file
GPU_NUM=8
MODEL_PATH="./data/flux"
OUTPUT_DIR="./data/train_flux_rl_embeddings"

torchrun --nproc_per_node=$GPU_NUM --master_port 19002 \
    fastvideo/data_preprocess/preprocess_flux_embedding_parquet.py \
    --model_path $MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --parquet_path "data/gemini_13k.parquet"
