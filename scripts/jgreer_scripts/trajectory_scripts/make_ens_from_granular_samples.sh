#!/usr/bin/bash

DOUBLING_SPACING_DATASETS=( '1' '2' '4' '8' '16' '32' '64' )
CONFORMATIONS_PATH=$'/mnt/parakeet_storage3/ConformationSampling/DESRES-Trajectory_sarscov2-13795965-no-water/'

for n in ${DOUBLING_SPACING_DATASETS[@]}; do
  LOGFILE=$'CreateEnsembleEvery'$n$'Steps.log'
  SAMPLED_CONFORMATIONS_DIR=$CONFORMATIONS_PATH'SampleGranularity'$n$'/'
  nohup python /mnt/parakeet_storage/trajectories/trajic/waymarking/ens_tools.py --structures_dir $SAMPLED_CONFORMATIONS_DIR --ensemble_filename $CONFORMATIONS_PATH$'EnsembleGranularity'$n --output_dir $CONFORMATIONS_PATH > $LOGFILE 2>&1 &
  wait $!
done