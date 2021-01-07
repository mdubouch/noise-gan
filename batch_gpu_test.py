#!/usr/bin/python3
#$ -P P_comet
#$ -j y
#$ -cwd
#$ -q mc_gpu_long
#$ -pe multicores_gpu 4
#$ -l sps=1,GPU=1,GPUtype=V100

import torch

print('cuda is available:', torch.cuda.is_available())

if (torch.cuda.is_available()):
    print('cuda device name:', torch.cuda.get_device_name())
