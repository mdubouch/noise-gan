#!/bin/bash

if [[ -z $JOB_ID ]]; then
    JOB_ID=$1
fi

python3 eval.py --job-id=$JOB_ID
