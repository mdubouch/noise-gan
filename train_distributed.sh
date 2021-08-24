#!/bin/bash
#$ -P P_comet
#$ -j y
#$ -cwd
#$ -M m.dubouchet18@imperial.ac.uk
#$ -m be
#$ -q mc_gpu_long
#$ -pe multicores_gpu 4
#$ -l sps=1,GPU=4,GPUtype=V100

python3 -m torch.distributed.launch --nproc_per_node=4 "$@"
