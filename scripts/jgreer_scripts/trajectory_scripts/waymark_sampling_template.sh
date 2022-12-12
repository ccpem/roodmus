#!/usr/bin/bash

WAYMARK_SCRIPT_PATH='/mnt/parakeet_storage/trajectories/trajic/waymarking/waymarking.py '
N_CONFORMATIONS=1
DEBUG='True'
SAMPLING_METHOD='waymark'
OUTPUT_DIR=$SAMPLING_METHOD$'_'$N_CONFORMATIONS
TRAJFILES_DIR_PATH='/mnt/parakeet_storage/trajectories/DESRES-Trajectory_sarscov2-13795965-no-water/sarscov2-13795965-no-water'
TOPFILE_PATH='/mnt/parakeet_storage/trajectories/DESRES-Trajectory_sarscov2-13795965-no-water/sarscov2-13795965-no-water/DESRES-Trajectory_sarscov2-13795965-no-water.pdb'
LOGFILE=$SAMPLING_METHOD$'_'$N_CONFORMATIONS$'.log'

# DESRES-Trajectory_sarscov2-13795965-no-water 1 conformations
python $WAYMARK_SCRIPT_PATH --output_dir $OUTPUT_DIR --debug $DEBUG --sampling_method $SAMPLING_METHOD --trajfiles_dir_path $TRAJFILES_DIR_PATH --topfile_path $TOPFILE_PATH --n_conformations $N_CONFORMATIONS > $LOGFILE