#!/bin/bash

rsync -at --prune-empty-dirs --include '*/' --include '*.png' --include '*.webm' --exclude '*' ./ md618@lx00.hep.ph.ic.ac.uk:~/public_html/job_eval
