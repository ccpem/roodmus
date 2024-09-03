#!/bin/bash

EXECUTABLE=$"roodmus"
SUBROUTINE=$" plot_ctf"

CONFIG_DIR=path/to/config/files
META_FILE=path/to/meta/file
JOB_TYPES="job019_extraction"
PLOT_DIR=path/to/plot/directory
PLOT_TYPES="scatter" # "scatter" or "per-particle-scatter"
DPI=300

$EXECUTABLE $SUBROUTINE \
    --verbose \
    --tqdm \
    --config_dir $CONFIG_DIR \
    --meta_file $META_FILE \
    --job_types $JOB_TYPES \
    --plot_dir $PLOT_DIR \
    --plot_types $PLOT_TYPES \
    --dpi $DPI

