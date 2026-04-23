#!/bin/bash

PORT=29501
MODEL="runwayml/stable-diffusion-v1-5"
DATASET="./dataset/BPO-Bench"

run_exp () {
  SCRIPT=$1
  OUTDIR=$2
  LR=$3
  BETA=$4
  BATCH=$5
  WORKERS=$6
  EXTRA_ARGS=$7

  echo "Running: $SCRIPT"

  accelerate launch --main_process_port $PORT $SCRIPT \
    --pretrained_model_name_or_path=$MODEL \
    --output_dir="$OUTDIR" \
    --dataset_name=$DATASET \
    --resolution=512 \
    --mixed_precision="fp16" \
    --train_batch_size=$BATCH \
    --dataloader_num_workers=$WORKERS \
    --learning_rate=$LR \
    --beta_dpo=$BETA \
    --report_to="tensorboard" \
    --checkpointing_steps=500 \
    $EXTRA_ARGS
}

# ========================
# EXP-1
# ========================
run_exp "train_pos.py" \
        "real-outputs/pos" \
        "1e-7" \
        "5000" \
        6 \
        6 \
        "--gradient_accumulation_steps=4 \
         --scale_lr \
         --lr_scheduler=constant_with_warmup \
         --lr_warmup_steps=100 \
         --max_train_steps=1000"

# ========================
# EXP-2
# ========================
run_exp "train_neg.py" \
        "real-outputs/neg" \
        "5e-6" \
        "500" \
        8 \
        8 \
        "--gradient_accumulation_steps=4 \
        --gradient_checkpointing \
        --use_8bit_adam \
        --rank=8 \
        --lr_scheduler=constant \
        --lr_warmup_steps=0 \
        --enable_xformers_memory_efficient_attention \
        --max_train_steps=2000"

echo "Done."