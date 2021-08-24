#!/bin/bash

JOB_SERIES=$1

if [[ -z $JOB_SERIES ]]; then
    echo Usage: $0 JOB_SERIES
    echo where JOB_SERIES is the first digit in the job indices with 7 digits
    exit 1
fi

FILE_LIST=$(ls output_* -d | grep -E "${JOB_SERIES}[0-9]{6}")
echo $FILE_LIST
du -hc $FILE_LIST

for TGT in $FILE_LIST; do
    COUNT_IRODS=$(ils /ccin2p3/home/comet/permanent/mdubouch/gan_archive/$TGT | grep "\.pt" | wc -l)
    COUNT_SPS=$(ls $TGT | grep "\.pt" | wc -l)

    if [[ $COUNT_IRODS == $COUNT_SPS ]]; then
        #read -p "Removing $COUNT_SPS pt files in $TGT. Press enter to confirm"
        echo $TGT/*.pt
        rm $TGT/*.pt
    else
        echo "Didn't find the right number of state files on iRODS. Skipping $TGT"
    fi
done
