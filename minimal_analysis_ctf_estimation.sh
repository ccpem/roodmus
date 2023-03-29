#!/usr/bin/bash

SCRIPT=$"/mnt/parakeet_storage4/roodmus/minimal_analysis.py"

# VERBOSE toggle on
# ANALYSE_CTF_ESTIMATION toggle on

nohup python $SCRIPT --verbose --analyse_ctf_estimation > analyse_ctf_estimation.log 2>&1 &