#!/bin/bash
#$ -P P_comet
#$ -j y
#$ -cwd
#$ -q mc_gpu_long
#$ -pe multicores_gpu 4
#$ -l sps=1,GPU=1,GPUtype=V100

if [[ -z $JOB_ID ]]; then
    JOB_ID=$1
fi

if [[ ! -z $2 ]]; then
    N_EPOCHS=$2
else
    N_EPOCHS=1
fi

python3 train.py --n-epochs=$N_EPOCHS --job-id=$JOB_ID
