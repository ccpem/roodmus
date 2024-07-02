#!/bin/bash

EXECUTABLE=$"roodmus"
SUBROUTINE=$" plot_ctf"

CONFIG_DIR=/path/to/config/files/dir
NUM_UGRAPHS=5
META_FILE=/path/to/metafile
PLOT_DIR=/path/to/output/files/dir
PLOT_TYPES="scatter" # "scatter" or "per-particle-scatter"
DPI=300

$EXECUTABLE $SUBROUTINE \
    --verbose \
    --tqdm \
    --config_dir $CONFIG_DIR \
    --num_ugraphs $NUM_UGRAPHS \
    --meta_file $META_FILE \
    --plot_dir $PLOT_DIR \
    --plot_types $PLOT_TYPES \
    --dpi $DPI

