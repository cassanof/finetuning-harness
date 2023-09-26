#!/bin/bash


# given two directories, it will copy all the contents of the directory $1 to all the subdirectories of $2
# $1: directory to copy from
# $2: directory with subdirectories to copy to

if [ $# -ne 2 ]; then
    echo "Usage: $0 <directory to copy from> <directory with subdirectories to copy to>"
    exit 1
fi

for dir in $2/*; do
    cp -r $1/* $dir
done
