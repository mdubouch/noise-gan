#!/bin/bash
#$ -P P_comet
#$ -j y
#$ -cwd
#$ -M m.dubouchet18@imperial.ac.uk
#$ -l sps=1,irods=1

# Store output files from GAN training to iRODS collection

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
        echo "Archiving states from $TGT"
        imkdir gan_archive/$TGT 
        iput -v $TGT/*.pt gan_archive/$TGT/
    else
        echo "Skip $TGT"
    fi
done
