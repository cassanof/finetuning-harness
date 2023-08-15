#!/bin/bash
if [ $# -lt 4 ]; then
  echo "Usage: multipl_e_eval.sh <lang> <MultiPL-E dir> <checkpoint dir> <dataset> <num_gpus>"
    exit 1
fi

LANG=$1
ROOT=$(realpath $2)
CHECKPOINT_DIR=$(realpath $3)
DATASET=$4
NUM_GPUS=$5

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

# group the checkpoints in lists of size NUM_GPUS or less, separated by '|'
CHECKPOINT_GROUPS=()
temp_group=""
for (( i=0; i<${#CHECKPOINT_DIRS[@]}; i++ )); do
    if (( (i % NUM_GPUS == 0 && i != 0) || i == ${#CHECKPOINT_DIRS[@]}-1 )); then
        CHECKPOINT_GROUPS+=("$temp_group")
        temp_group="${CHECKPOINT_DIRS[$i]}"
    else
        temp_group+="${CHECKPOINT_DIRS[$i]}|"
    fi
done

pushd $ROOT
for (( gi=0; gi<${#CHECKPOINT_GROUPS[@]}; gi++ )); do
  IFS='|' read -ra ADDR <<< "${CHECKPOINT_GROUPS[$gi]}"
  echo "Firing off gpu-group $gi"
  PIDS=()

  for (( i=0; i<${#ADDR[@]}; i++ )); 
  do 
      BASEDIR=$(basename ${ADDR[$i]})
      OUT_DIR="${ADDR[$i]}/eval_${LANG}_${BASEDIR}"
      echo "Starting process $i with checkpoint ${ADDR[$i]} - output dir: $OUT_DIR"
      mkdir -p $OUT_DIR
      if [ $IS_LOCAL -eq 0 ]; then
        CUDA_VISIBLE_DEVICES=$i python3 automodel.py \
            --name ${ADDR[$i]} \
            --root-dataset $DATASET \
            --lang $LANG \
            --completion-limit 20 \
            --batch-size 20 \
            --temperature 0.2 \
            --output-dir $OUT_DIR &
      else
        CUDA_VISIBLE_DEVICES=$i python3 automodel.py \
            --name ${ADDR[$i]} \
            --use-local \
            --dataset $DATASET \
            --lang $LANG \
            --completion-limit 20 \
            --batch-size 20 \
            --temperature 0.2 \
            --output-dir $OUT_DIR &
      fi
      PIDS+=($!)
  done

  echo "Waiting for all processes to finish... Pids: ${PIDS[@]}"

  # capture a ctrl-c and kill all processes
  function ctrl_c() {
      echo "Trapped CTRL-C, killing all processes..."
      for pid in ${PIDS[@]}; do
          kill $pid
      done
      exit 1
  }

  trap ctrl_c INT

  wait # wait for all background processes to finish

  # run docker eval
  for (( i=0; i<${#ADDR[@]}; i++ ));
  do
    BASEDIR=$(basename ${ADDR[$i]})
    EVAL_DIR="${ADDR[$i]}/eval_${LANG}_${BASEDIR}"
    echo "Running docker eval in background for $EVAL_DIR"
    docker run --rm -d --network none --volume $EVAL_DIR:/inputs:ro --volume $EVAL_DIR:/outputs:rw multipl-e-evaluation --dir /inputs --output-dir /outputs
  done
done
popd
