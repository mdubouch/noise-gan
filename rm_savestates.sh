#!/bin/bash

DIR=$1

if [ -z $DIR ]; then
    echo Usage: $0 DIRECTORY_LIST
    exit 1
fi

for dir in "$@"; do

    SAVE_FILES=$(ls -t $dir/*.pt | awk '{if(NR>1)print}')
    echo "Removing $(wc -w <<< $SAVE_FILES) save states"
    read -p "Press enter to confirm"

    rm -f $SAVE_FILES
done
