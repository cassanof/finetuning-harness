#CUDA_VISIBLE_DEVICES=... python3 -m torch.distributed.launch \
python3 -m torch.distributed.launch \
        --nproc_per_node 8 main.py \
        --model_path="/home/arjun/models/starcoderbase-1b/" \
        --no_custom_tokenizer \
        --model_revision="main" \
        --dataset_name="nuprl/stack-dedup-python-testgen-starcoder-format" \
        --subset="data" \
        --lang="py" \
        --data_column "content" \
        --split="train" \
        --output_dir="./model_starcoder1b_lora" \
        --seq_length 2048 \
        --batch_size 16 \
        --gradient_accumulation_steps 1 \
        --learning_rate 1e-4 \
        --max_steps 70 \
        --num_warmup_steps 5 \
        --eval_freq 5 \
        --save_freq 5 \
        --log_freq 1 \
        --num_workers=$(expr $(nproc --all) - 4) \
        --no_fp16 \
        --bf16 \
        --lora \
        --lora_r 32 \
        --lora_alpha 32 \
        --lora_dropout 0.1 \
        --humaneval_eval_loss \
        --eval_reruns 30 
