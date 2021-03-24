#!/bin/bash

START_IDX=$1
END_IDX=$2

if [[ -z $START_IDX || -z $END_IDX ]]; then
    echo Usage: $0 START_IDX END_IDX
    exit 1
fi

for i in $(seq $START_IDX $END_IDX); do
    TGT="output_$i"
    MISSING=$(stat $TGT &> /dev/null; echo $?)
    if [ $MISSING == 0 ]; then
        #echo "Checking states are on iRODS"
        COUNT_IRODS=$(ils gan_archive/$TGT | grep "\.pt" | wc -l)
        COUNT_SPS=$(ls $TGT | grep "\.pt" | wc -l)
        if [[ $COUNT_IRODS == $COUNT_SPS ]]; then
            #read -p "Removing $COUNT_SPS pt files in $TGT. Press enter to confirm"
            echo $TGT/*.pt
            rm $TGT/*.pt
        else
            echo "Didn't find the right number of state files on iRODS. Skipping $TGT"
        fi
    else
        echo "Skip $TGT"
    fi

done
