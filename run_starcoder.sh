#CUDA_VISIBLE_DEVICES=... python3 -m torch.distributed.launch \
torchrun \
        --nproc_per_node 4 \
        train.py \
        --deepspeed="./deepspeed_z3_config_bf16.json" \
        --model_path="bigcode/starcoderbase" \
        --no_custom_tokenizer \
        --dataset_name="nuprl/stack_dedup_lua_codegen" \
        --output_dir="./model_starcoder_lua50k" \
        --seq_length 2048 \
        --epochs 10 \
        --batch_size 1 \
        --gradient_accumulation_steps 8 \
        --learning_rate 2e-5 \
        --num_warmup_steps 10 \
        --num_workers=$(expr $(nproc --all) - 4) \
        --no_fp16 \
        --bf16 \
        --perc_valid_set 0.01 \
        --save_total_limit 20
