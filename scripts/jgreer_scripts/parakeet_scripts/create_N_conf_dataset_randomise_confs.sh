#!/usr/bin/bash

start=`date +%s`

# SET UP VARIABLES
# number of conformations
CONFORMATIONS=$1
# number of particles to put in micrographs
PARTICLES=$2
#output folder
OUTPUTFOLDER=$3
mkdir $OUTPUTFOLDER
#outputfile
OUTPUTFILENAME=$4
# path to directory with subfolders that correspond to even_sampling and/or waymark_sampling
# Include trailing and leading /'s please!
# Example: /mnt/parakeet_storage2/ConformationSampling/DESRES-Trajectory_sarscov2-13795965-no-water/
CONFORMATIONSFILEPATH=$5
# sampling type (no leading /)!
# Example: even_sampling
SAMPLING_TYPE=$6

# config details
CONFIG_TEMPLATE='config_template.yaml'
CONFIG_WORKING_NAME='config_working_name.yaml'
CONFIG_WRITER_SCRIPT='/mnt/parakeet_storage/trajectories/trajic/parakeet_scripts/parakeet_config_writer.py'

## path to script to create yaml defining the particles to put into each image
MOLECULES_CONFIG_WRITE_SCRIPT='/mnt/parakeet_storage/trajectories/trajic/parakeet_scripts/parakeet_molecules_config_writer.py'

# conformation file prefix and suffix
CONFORMATIONS_PATH=$CONFORMATIONSFILEPATH$SAMPLING_TYPE$'/'
CONFORMATIONS_FILE=$'conformation_'
CONFORMATIONS_FILE_EXT='.pdb'

ENERGY=300.
EPA=45.
c_10=-20000
c_30=2.7
c_c=2.7
PIXEL_SIZE=1.
# will always use 200 particles per micrograph
INSTANCES=200

# set energy, epa, c_10, c_30, c_c, pixel_size in config file
python $CONFIG_WRITER_SCRIPT -i $CONFIG_TEMPLATE -o $CONFIG_WORKING_NAME -e $EPA -c_c $c_c -c_30 $c_30 -E $ENERGY -d $c_10 -p $PIXEL_SIZE
# note have already changed nx,ny, cuboid-lengthxyz parameters
# as well as sample.type to cuboid
# and simulation.ice to True

# need to create a dataset of PARTICLES from a set of conformations

# work out how many images are from each conformation
IMAGESREQ=$(($PARTICLES/$INSTANCES))
echo $IMAGESREQ$' images will be generated using dataset of '$CONFORMATIONS$' conformations'

## create the yaml file (dict of list of dicts indexed by 6-digit image index)
MOLECULES_LOGFILE='local_mols.log'
MOLECULES_YAML='local_mols.yaml'
## also set the yaml file name for the defoci
DEFOCI_YAML='defoci.yaml'
DEFOCI_LOGFILE='defoci.log'
DEFOCI_SPREAD=5000
DEFOCI_SPREAD_TYPE='gaussian'
python $MOLECULES_CONFIG_WRITE_SCRIPT -i $IMAGESREQ -c $CONFORMATIONS -np $INSTANCES -cp $CONFORMATIONS_PATH -st $SAMPLING_TYPE -cf $CONFORMATIONS_FILE -cfe $CONFORMATIONS_FILE_EXT -f $MOLECULES_YAML -l $MOLECULES_LOGFILE -fd $DEFOCI_YAML -d $DEFOCI_LOGFILE -md $c_10 -sd $DEFOCI_SPREAD -sdt $DEFOCI_SPREAD_TYPE 


# create an image counter
IMAGE_N=0

# begin loop over images required
for c in $(seq 0 $(($IMAGESREQ-1))); do
  
  # set the conformation and number of instances in config
  python $CONFIG_WRITER_SCRIPT -i $CONFIG_WORKING_NAME -o $CONFIG_WORKING_NAME -in $(printf "%06d" $c) -c $MOLECULES_YAML -vd $DEFOCI_YAML
  cp $CONFIG_WORKING_NAME $OUTPUTFOLDER$'/'

  # run the parakeet simulation
  pushd $OUTPUTFOLDER
  echo $'Doing subjob: '$(printf "%06d" $IMAGE_N)$' from config: '$CONFIG_WORKING_NAME
  OUTPUTIMAGEH5=$OUTPUTFILENAME$'.h5'
  parakeet.run -c $CONFIG_WORKING_NAME -i $OUTPUTIMAGEH5
  #convert image from h5 to mrc
  #OUTPUTFILE=$OUTPUTFOLDER$'/'$OUTPUTFILENAME$(printf "%06d" $IMAGE_N)$'.mrc'
  OUTPUTFILE=$OUTPUTFILENAME$(printf "%06d" $IMAGE_N)$'.mrc'

  parakeet.export $OUTPUTIMAGEH5 -o $OUTPUTFILE
  # remove intermediate work files
  rm *$'.h5'
  echo 'Finished subjob: '$(printf "%06d" $IMAGE_N)
  printf '\n'
  popd
  IMAGE_N=$(($IMAGE_N+1))
done

end=`date +%s`

runtime=$((end-start))
echo $runtime$' seconds taken to simulate '$IMAGE_N$' micrographs'