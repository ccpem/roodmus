#!/bin/bash

EXECUTABLE=$"roodmus"
SUBROUTINE=$" plot_picking"

CONFIG_DIR=path/to/config/files
NUM_UGRAPHS=1
META_FILE="path/to/meta/file1 \
    path/to/meta/file2 "
JOB_TYPES="job012_extraction job020_selection"
PLOT_DIR=path/to/plot/directory
PLOT_TYPES="label_truth label_picked label_truth_and_picked label_matched_and_unmatched precision boundary overlap"
BOX_WIDTH=16
BOX_HEIGHT=16
PARTICLE_DIAMETER=100
DPI=300

$EXECUTABLE $SUBROUTINE \
    --verbose \
    --tqdm \
    --config_dir $CONFIG_DIR \
    --num_ugraphs $NUM_UGRAPHS \
    --meta_file $META_FILE \
    --job_types $JOB_TYPES \
    --plot_dir $PLOT_DIR \
    --plot_types $PLOT_TYPES \
    --dpi $DPI
