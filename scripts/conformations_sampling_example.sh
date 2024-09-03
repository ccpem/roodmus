#!/usr/bin/bash

EXECUTABLE=$"roodmus"
SUBROUTINE=$" conformations_sampling"

TRAJFILES_DIR_PATH="/path/to/trajfiles/dir"
TOPFILE_PATH="/path/to/topfile"
SAMPLING_METHOD=$"even_sampling"
N_CONFORMATIONS=5
#LIMIT_N_TRAJ_SUBFILES=1
TRAJ_EXTENSION=$".dcd"
OUTPUT_DIR=$"conformations_sampling_example"
RANDOM_STARTPOINT_SEED=99
# RND_START toggled on if --rnd_start is passed as arg
# USE_CONTIGUOUS_CONFORMATIONS= toggled on if use_contiguous_conformations is pased as arg
# INVESTIGATE_TRAJECTORY_FILES toggled on if --investigate_trajectory_files is passed as arg
#RMSD=0.5
LOGFILE=$OUTPUT_DIR$"/conformation_sampling.log"

nohup $EXECUTABLE $SUBROUTINE --trajfiles_dir_path $TRAJFILES_DIR_PATH --topfile_path $TOPFILE_PATH --verbose --sampling_method $SAMPLING_METHOD --n_conformations $N_CONFORMATIONS --traj_extension $TRAJ_EXTENSION --output_dir $OUTPUT_DIR --random_startpoint_seed  $RANDOM_STARTPOINT_SEED --rnd_start --use_contiguous_conformations > $LOGFILE 2>&1 &
