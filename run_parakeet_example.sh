#!/usr/bin/bash

PDB_DIR=$""
MRC_DIR=$""
N_IMAGES=2
N_MOLECULES=300

ELECTRONS_PER_ANGSTROM=45.0
ENERGY=300.0
#DQE=
NX=4000
NY=4000
PIXEL_SIZE=1.0
C_10=-20000.0
C_10_STDDEV=5000.0
C_C=2.7
BOX_X=4000.0
BOX_Y=4000.0
BOX_Z=500.0
CENTRE_X=2000.0
CENTRE_Y=2000.0
CENTRE_Z=250.
TYPE=$"cuboid"
CUBOID_LENGTH_X=4000.0
CUBOID_LENGTH_Y=4000.0
CUBOID_LENGTH_Z=500.0
#FAST_ICE=
SIMULATION_MARGIN=0
SIMULATION_PADDING=1

LOGFILE=$"run_parakeet_example.log"

nohup python roodmus.py run_parakeet --dqe --fast_ice --pdb_dir $PDB_DIR --mrc_dir $MRC_DIR --n_images $N_IMAGES --n_molecules $N_MOLECULES --electrons_per_angstrom=$ELECTRONS_PER_ANGSTROM --energy $ENERGY --nx $NX --ny $NY --pixel_size $PIXEL_SIZE --c_10 $C_10 --c_10_stddev $C_10_STDDEV --c_c $C_C --box_x $BOX_X --box_y $BOX_Y --box_z $BOX_Z --centre_x $CENTRE_X --centre_y $CENTRE_Y --centre_z $CENTRE_Z --type $TYPE --cuboid_length_x $CUBOID_LENGTH_X --cuboid_length_y $CUBOID_LENGTH_Y --cuboid_length_z $CUBOID_LENGTH_Z --simulation_margin $SIMULATION_MARGIN --simulation_padding $SIMULATION_PADDING > $LOGFILE 2>&1 &