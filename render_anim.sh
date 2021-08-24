#!/bin/bash

JOB_ID=$1

if [[ -z $JOB_ID ]]; then
    echo Usage: $0 JOB_ID
    exit 1
fi

cd output_$JOB_ID/anim
ffmpeg -framerate 25 -i frame_%03d.png -c:v libvpx-vp9 -crf 30 -b:v 0 -pix_fmt yuva420p anim.webm
