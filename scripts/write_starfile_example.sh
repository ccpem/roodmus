#!/bin/bash

EXECUTABLE=$"roodmus"
SUBROUTINE=$" write_starfile"

INPUT_CSV=path/to/input/csv/file
TYPE=coordinate_star
OUTPUT_DIR=path/to/output/directory
UGRAPH_DIR=Micrographs
PIXEL_SIZE=1

$EXECUTABLE $SUBROUTINE \
    --verbose \
    --tqdm \
    --input_csv $INPUT_CSV \
    --type $TYPE \
    --output_dir $OUTPUT_DIR \
    --ugraph_dir $UGRAPH_DIR \
    --pixel_size $PIXEL_SIZE

INPUT_CSV=path/to/input/csv/file
TYPE=data_star
OUTPUT_DIR=path/to/output/directory
UGRAPH_DIR=Micrographs
PIXEL_SIZE=1
OPTICS_GROUP_NAME=opticsGroup1
OPTICS_GROUP=1
MTF_FILENAME=mtf_300kV.star
MICROGRAPH_ORIGINAL_PIXEL_SIZE=1.0
VOLTAGE=300.0
SPHERICAL_ABERRATION=2.7
AMPLITUDE_CONTRAST=0.1
IMAGE_PIXEL_SIZE=1.0
IMAGE_SIZE=128
IMAGE_DIMENSIONALITY=2
CTF_DATA_ARE_CTF_PREMULTIPLIED=0

$EXECUTABLE $SUBROUTINE \
    --verbose \
    --tqdm \
    --input_csv $INPUT_CSV \
    --type $TYPE \
    --output_dir $OUTPUT_DIR \
    --ugraph_dir $UGRAPH_DIR \
    --pixel_size $PIXEL_SIZE \
    --optics_group_name $OPTICS_GROUP_NAME \
    --optics_group $OPTICS_GROUP \
    --mtf_filename $MTF_FILENAME \
    --micrograph_original_pixel_size $MICROGRAPH_ORIGINAL_PIXEL_SIZE \
    --voltage $VOLTAGE \
    --spherical_aberration $SPHERICAL_ABERRATION \
    --amplitude_contrast $AMPLITUDE_CONTRAST \
    --image_pixel_size $IMAGE_PIXEL_SIZE \
    --image_size $IMAGE_SIZE \
    --image_dimensionality $IMAGE_DIMENSIONALITY \
    --ctf_data_are_ctf_premultiplied $CTF_DATA_ARE_CTF_PREMULTIPLIED


