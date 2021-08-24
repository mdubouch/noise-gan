#!/bin/bash

JOB_ID=$1
FRAMERATE=$2

if [[ -z $JOB_ID || -z $FRAMERATE ]]; then
    echo Usage: $0 JOB_ID FRAMERATE
    exit 1
fi

cd output_$JOB_ID/latent_anim
ffmpeg -framerate $FRAMERATE -i frame_%03d.png -c:v libvpx-vp9 -crf 30 -b:v 0 -pix_fmt yuva420p anim.webm

echo Done rendering job $JOB_ID at framerate $FRAMERATE
