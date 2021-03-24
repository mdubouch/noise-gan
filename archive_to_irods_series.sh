#!/bin/bash
#$ -P P_comet
#$ -j y
#$ -cwd
#$ -M m.dubouchet18@imperial.ac.uk
#$ -l sps=1,irods=1

# Store output files from GAN training to iRODS collection

JOB_SERIES=$1

if [[ -z $JOB_SERIES ]]; then
    echo Usage: $0 JOB_SERIES
    echo where JOB_SERIES is the first digit in the job indices with 7 digits
    exit 1
fi

FILE_LIST=$(ls output_* -d | grep -E "${JOB_SERIES}[0-9]{6}")
echo $FILE_LIST

for TGT in $FILE_LIST; do
    echo "Archiving states from $TGT to /ccin2p3/home/comet/permanent/mdubouch/gan_archive/$TGT"
    imkdir /ccin2p3/home/comet/permanent/mdubouch/gan_archive/$TGT 
    iput -v $TGT/*.pt /ccin2p3/home/comet/permanent/mdubouch/gan_archive/$TGT/
done

