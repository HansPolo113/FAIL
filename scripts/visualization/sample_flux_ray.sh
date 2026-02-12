#!/bin/bash
# Ray-based distributed FLUX image generation script
set -x

DATASET_TYPE=${DATASET_TYPE:-alchemist}  # alchemist, dpg, unigenbench, hpsv3
PROMPT_VERSION=${PROMPT_VERSION:-normal}  # normal, long (for unigenbench)

export PYTHONPATH="$PYTHONPATH:$(pwd)"

# Dataset Configuration
if [ "$DATASET_TYPE" = "alchemist" ]; then
    DATA_PATH="data/alchemist_3k_final.csv"
    OUTPUT_DIR="./output/alchemist_vis"
    RESOLUTION=512
    NUM_IMAGES=1
elif [ "$DATASET_TYPE" = "dpg" ]; then
    DATA_PATH="data/dpg_prompt/prompts"
    OUTPUT_DIR="./output/dpg_vis"
    RESOLUTION=512
    NUM_IMAGES=4
elif [ "$DATASET_TYPE" = "unigenbench" ]; then
    DATA_PATH="data/unigenbench_prompt"
    OUTPUT_DIR="./output/unigenbench_vis"
    RESOLUTION=512
    NUM_IMAGES=4
elif [ "$DATASET_TYPE" = "hpsv3" ]; then
    DATA_PATH="data/hpsv3/benchmark"
    OUTPUT_DIR="./output/hpsv3_vis"
    RESOLUTION=512
    NUM_IMAGES=1
else
    echo "Unknown DATASET_TYPE: $DATASET_TYPE"
    exit 1
fi

# Model Paths
FLUX_MODEL="./data/flux"
CHECKPOINT_PATH=""  # Set to checkpoint path if loading trained model

CMD="python scripts/visualization/sample_flux_ray.py \
    --dataset_type $DATASET_TYPE \
    --data_path $DATA_PATH \
    --pretrained_model_name_or_path $FLUX_MODEL \
    --output_dir $OUTPUT_DIR \
    --guidance_scale 1.0 \
    --true_cfg_scale 1.0 \
    --resolution $RESOLUTION \
    --num_images_per_prompt $NUM_IMAGES \
    --num_inference_steps 28 \
    --seed 42 \
    --save_format png"

if [ "$DATASET_TYPE" = "unigenbench" ]; then
    CMD="$CMD --prompt_version $PROMPT_VERSION"
fi

if [ -n "$CHECKPOINT_PATH" ]; then
    CMD="$CMD --checkpoint_path $CHECKPOINT_PATH --load_checkpoint"
fi

echo "================================"
echo "Ray-based FLUX Visualization"
echo "================================"
echo "Dataset: $DATASET_TYPE"
echo "Data path: $DATA_PATH"
echo "Output dir: $OUTPUT_DIR"
echo "================================"

eval $CMD
