#!/bin/bash
#$ -P P_comet
#$ -j y
#$ -cwd
#$ -q mc_gpu_long
#$ -pe multicores_gpu 4
#$ -l sps=1,GPU=1,GPUtype=V100

python3 ./batch_gpu_test.py
