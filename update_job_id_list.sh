#!/bin/bash

OUT="var job_id_list = ["

LIST=$(ls output_* -dt)
for f in $LIST; do
    SPLIT=(${f//_/ })
    ID=${SPLIT[1]}
    OUT="$OUT '$ID',"
done
OUT="$OUT ];"
echo $OUT > job_id_list.js
