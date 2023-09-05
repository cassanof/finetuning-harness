#!/bin/bash
# need to give deepspeed config file as argument and language as second argument


if [ -z "$1" ]
  then
    echo "No language supplied"
    exit 1
fi
T_LANG=$1
# check if env is set, otherwise default to 24000
MASTER_PORT=${MASTER_PORT:-24000}
python3 -m torch.distributed.launch \
        --master_port $MASTER_PORT \
        --nproc_per_node 8 \
        train.py \
        --model_path="smallcloudai/Refact-1_6B-fim" \
        --no_custom_tokenizer \
        --dataset_name="nuprl/MultiPL-T" \
        --split="$T_LANG" \
        --no_approx_tokens \
        --output_dir="./model_refact_multiplt_$T_LANG" \
        --seq_length 4096 \
        --epochs 5 \
        --batch_size 1 \
        --gradient_accumulation_steps 4 \
        --learning_rate 2e-5 \
        --num_warmup_steps 10 \
        --num_workers=$(expr $(nproc --all) - 4) \
        --no_fp16 \
        --bf16 \
        --perc_valid_set 0.0025 \
        --save_total_limit 20
