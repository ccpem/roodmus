#!/usr/bin/bash

SCRIPT=$"/mnt/parakeet_storage4/roodmus/minimal_analysis.py"

# VERBOSE toggle on
# PARTICLE_LABELLING toggle on

nohup python $SCRIPT --verbose --particle_labelling > particle_labelling.log 2>&1 &