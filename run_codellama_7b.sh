# need to give deepspeed config file as argument
if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Please give deepspeed config file as argument"
    exit 1
fi
python3 -m torch.distributed.launch \
        --nproc_per_node 8 \
        train.py \
        --deepspeed="$1" \
        --model_path="codellama/CodeLlama-7b-hf" \
        --no_custom_tokenizer \
        --dataset_name="nuprl/stack_dedup_lua_codegen_full" \
        --no_approx_tokens \
        --output_dir="./model_codellama_7b_multiplt" \
        --seq_length 2048 \
        --epochs 5 \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --num_warmup_steps 10 \
        --num_workers=$(expr $(nproc --all) - 4) \
        --no_fp16 \
        --bf16 \
        --perc_valid_set 0.0025 \
        --save_total_limit 20
