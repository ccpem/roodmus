#!/usr/bin/bash

# REMEMBER TO RUN VIA FOLLOWING USING NOHUP!
# conda activate strato
# nohup run_create_N_conf_datasets.sh

image_name=$'image'
particles_per_ugraph=200
particles_per_dataset=60000
n_confs=( '10000' )
conformationspath=$'/mnt/parakeet_storage3/ConformationSampling/DESRES-Trajectory_sarscov2-13795965-no-water/'
sampling_type=$'even_sampling'

for n in ${n_confs[@]}; do
  outputdir=$'DESRES-Trajectory_sarscov2-13795965-no-water_'$n$'_conf'
  logfile=$outputdir$'.log'
  echo $'Running dataset of '$particles_per_dataset$' particles for '$n$' conformations into directory '$outputdir
  time bash create_N_conf_dataset_randomise_confs.sh $n $particles_per_dataset $outputdir $image_name $conformationspath $sampling_type > $logfile
  echo $'Finished running dataset of '$particles_per_dataset$' particles for '$n$' conformations into directory '$outputdir
done