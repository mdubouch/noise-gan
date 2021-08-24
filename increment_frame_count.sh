#!/bin/bash

JOB_ID=$1
OFFSET=$2

if [[ -z $JOB_ID || -z $OFFSET ]]; then
    echo Usage: $0 JOB_ID OFFSET
    exit 1
fi

cd output_$JOB_ID/anim

mkdir -p tmp

for f in *.png; do
    INDEX=$(echo $f | cut -d'_' -f 2 | cut -d'.' -f 1 | sed 's/^0*//')
    NEW_INDEX=$(printf '%03d' $(($INDEX + $OFFSET)))
    NEW_NAME=frame_${NEW_INDEX}.png
    echo $NEW_NAME
    mv $f tmp/$NEW_NAME
done

mv tmp/* ./
rm tmp -r

