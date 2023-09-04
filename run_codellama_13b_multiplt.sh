#!/bin/bash
# need to give deepspeed config file as argument and language as second argument


# throw error if deepspeed and language not given as argument
if [ -z "$1" ]
  then
    echo "No deepspeed config file supplied"
    exit 1
fi
if [ -z "$2" ]
  then
    echo "No language supplied"
    exit 1
fi
T_LANG=$2
python3 -m torch.distributed.launch \
        --nproc_per_node 8 \
        train.py \
        --deepspeed="$1" \
        --model_path="codellama/CodeLlama-13b-hf" \
        --no_custom_tokenizer \
        --dataset_name="nuprl/MultiPL-T" \
        --split="$T_LANG" \
        --no_approx_tokens \
        --output_dir="./model_codellama_13b_multiplt_$T_LANG" \
        --seq_length 2048 \
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
