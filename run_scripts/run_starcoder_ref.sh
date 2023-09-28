# need to give deepspeed config file as argument
if [ $# -eq 0 ]
  then
    echo "No arguments supplied. Please give deepspeed config file as argument"
    exit 1
fi
# check if env is set, otherwise default to 24000
MASTER_PORT=${MASTER_PORT:-24000}
python3 -m torch.distributed.launch \
        --nproc_per_node 4 \
        --master_port $MASTER_PORT \
        main.py \
        --deepspeed="$1" \
        --model_path="bigcode/starcoderbase" \
        --dataset_name="nuprl-staging/py_reflection_bootstrap_2500_train" \
        --output_dir="./model_starcoder_bootstrap_ref" \
        --seq_length 4096 \
        --epochs 10 \
        --batch_size 1 \
        --gradient_accumulation_steps 1 \
        --learning_rate 2e-5 \
        --num_warmup_steps 10 \
        --num_workers=$(expr $(nproc --all) - 4) \
        --no_fp16 \
        --bf16 \
        --perc_valid_set 0.05 \
        --save_total_limit 20
