#CUDA_VISIBLE_DEVICES=... python3 -m torch.distributed.launch \
python3 train.py \
        --model_path="/home/federico/starcoderbase-1b/" \
        --no_custom_tokenizer \
        --model_revision="main" \
        --dataset_name="nuprl/stack_dedup_lua_codegen" \
        --lang="lua" \
        --output_dir="./model_starcoder1b_edu5" \
        --seq_length 2048 \
        --batch_size 16 \
        --gradient_accumulation_steps 1 \
        --save_total_limit 20 \
        --learning_rate 5e-5 \
        --epochs 20 \
        --num_warmup_steps 15 \
        --num_workers=$(expr $(nproc --all) - 4) \
        --no_fp16 \
        --bf16 \
        --perc_valid_set 0.05
