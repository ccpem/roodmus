#!/usr/bin/bash

CONFORMATIONS_DIR=$'/mnt/parakeet_storage3/ConformationSampling/DESRES-Trajectory_sarscov2-13795965-no-water/consecutive_conformations_from_start/even_sampling_5000'

DEBUG=True

N_CONFS_PER_ENSEMBLE=50
DOUBLING_SPACING_DATASETS=( '1' '2' '4' '8' '16' '32' '64' )

for n in ${DOUBLING_SPACING_DATASETS[@]}; do
  N_CONFS_TO_SAMPLE_FROM=$((n*$N_CONFS_PER_ENSEMBLE))
  LOGFILE=$'SampleEvery'$n$'Confs.log'
  SAMPLED_CONFORMATIONS_DIR=$'/mnt/parakeet_storage3/ConformationSampling/DESRES-Trajectory_sarscov2-13795965-no-water/SampleGranularity'$n$'/'
  nohup python /mnt/parakeet_storage/trajectories/trajic/waymarking/investigate_dynamics.py --conformations_dir $CONFORMATIONS_DIR --limit_contiguous_conformations $N_CONFS_TO_SAMPLE_FROM --sampling_granularity $n --sampled_conformations_dir $SAMPLED_CONFORMATIONS_DIR --debug $DEBUG > $LOGFILE 2>&1 &
  wait $!
done

echo $'Finished sampling all conformation granularities'