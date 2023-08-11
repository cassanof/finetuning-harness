#!/bin/bash
if [ $# -lt 4 ]; then
  echo "Usage: multipl_e_eval.sh <lang> <MultiPL-E dir> <checkpoint dir> <num_gpus>"
    exit 1
fi

LANG=$1
ROOT=$(realpath $2)
CHECKPOINT_DIR=$(realpath $3)
NUM_GPUS=$4

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
  echo "Group $gi"
  PIDS=()

  for (( i=0; i<${#ADDR[@]}; i++ )); 
  do 
      echo "Starting process $i with checkpoint ${ADDR[$i]}"
      OUT_DIR="${ADDR[$i]}/eval"
      mkdir -p $OUT_DIR
      CUDA_VISIBLE_DEVICES=$i python3 automodel.py \
          --name ${ADDR[$i]} \
          --root-dataset humaneval \
          --lang $LANG \
          --completion-limit 20 \
          --batch-size 20 \
          --temperature 0.2 \
          --output-dir $OUT 
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
done
popd
