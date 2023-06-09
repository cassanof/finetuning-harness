#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: ./sync_pod.sh <pod name> <path to project on pod>"
    exit 1
fi

pod_name=$1
proj_path=$2

task() {
  # copy file to pod
  file=$1
  echo "Copying $file to $pod_name:$proj_path/$file"
  kubectl cp $file $pod_name:$proj_path/$file
}

N=5
(
# for file that is not a directory in this directory
for file in $(ls -ap | grep -v /); do
   ((i=i%N)); ((i++==0)) && wait
   task "$file" &
done
)
echo "Done copying files to pod"
