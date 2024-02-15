# unlike multipl_e_eval.sh, this script does not do parallelism, but shards the model across multiple GPUs,
# for very large models that do not fit in a single GPU
#!/bin/bash
if [ $# -lt 4 ]; then
  echo "Usage: multipl_e_eval_shard.sh <lang> <MultiPL-E dir> <checkpoint dir> <dataset> <num_gpus>"
    exit 1
fi

LANG=$1
ROOT=$(realpath $2)
CHECKPOINT_DIR=$(realpath $3)
DATASET=$4
NUM_GPUS=$5

if [[ -z "${BATCH_SIZE}" ]]; then
  V_BATCH_SIZE=20
else
  V_BATCH_SIZE="$BATCH_SIZE"
fi

if [[ -z "${COMP_LIMIT}" ]]; then
  V_COMP_LIMIT=20
else
  V_COMP_LIMIT="$COMP_LIMIT"
fi


IS_LOCAL=1
# if the DATASET is "humaneval" or "mbpp", set IS_LOCAL to 0
if [ $DATASET == "humaneval" ] || [ $DATASET == "mbpp" ]; then
  IS_LOCAL=0
else
  DATASET=$(realpath $DATASET)
fi
echo "Using dataset: $DATASET - is local: $IS_LOCAL"

# put in a list all the directories that contain the checkpoints
CHECKPOINT_DIRS=()
for dir in $CHECKPOINT_DIR/*; do
    CHECKPOINT_DIRS+=($dir)
done

pushd $ROOT

for dir in "${CHECKPOINT_DIRS[@]}"; do
  echo "Evaluating model in $dir"
  BASEDIR=$(basename $dir)
  OUT_DIR="$dir/eval_${LANG}_${BASEDIR}"
  mkdir -p $OUT_DIR
  if [ $IS_LOCAL -eq 0 ]; then
    python3 automodel_vllm.py \
        --name $dir \
        --root-dataset $DATASET \
        --lang $LANG \
        --completion-limit $V_COMP_LIMIT \
        --batch-size $V_BATCH_SIZE \
        --temperature 0.2 \
        --output-dir $OUT_DIR \
        --num_gpus $NUM_GPUS
  else
    python3 automodel_vllm.py \
        --name $dir \
        --use-local \
        --dataset $DATASET \
        --lang $LANG \
        --completion-limit $V_COMP_LIMIT \
        --batch-size $V_BATCH_SIZE \
        --temperature 0.2 \
        --output-dir $OUT_DIR \
        --num_gpus $NUM_GPUS
  fi
  echo "Running docker eval in background for $OUT_DIR"
  docker run --rm -d --network none --volume $OUT_DIR:/inputs:ro --volume $OUT_DIR:/outputs:rw multipl-e-evaluation --dir /inputs --output-dir /outputs
done

popd
