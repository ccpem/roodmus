# take imported micrgraphs and make a starfile with the ground truth coordinates as well as the other optics
# group information necessary to allow particles to be extracted into particle stacks straight from the
# micrographs

# in this case we need no motion correction!
# and can/should use ground truth per-particle defoci?
# WILL USE PER-UGRAPH DEFOCUS, simply because already coded in. MAYBE REVISE THIS?

INPUT_CSV="truth.csv"
TYPE="data_star" #? check
OUTPUT_DIR="GT_starfile"
UGRAPH_DIR="Micrographs" # the dir of imported micrographs
PIXEL_SIZE=1.0

# optics group arguments
OPTICS_GROUP_NAME="opticsGroup1"
OPTICS_GROUP="1"
MTF_FILENAME="Micrographs/relion/mtf_300kV.star"
MICROGRAPH_ORIGINAL_PIXEL_SIZE="1.0"
VOLTAGE="300"
SPHERICAL_ABERRATION="2.7"
AMPLITUDE_CONTRAST="0.1"
IMAGE_PIXEL_SIZE=1.0
IMAGE_SIZE=256
IMAGE_DIMENSIONALITY=2
CTF_DATA_ARE_CTF_PREMULTIPLIED=0

# EXTRACT_DIR="" not used in this case

roodmus write_starfile --pp_defoci --tqdm --verbose --input_csv $INPUT_CSV --type $TYPE --output_dir $OUTPUT_DIR --ugraph_dir $UGRAPH_DIR --pixel_size $PIXEL_SIZE --optics_group_name $OPTICS_GROUP_NAME --optics_group $OPTICS_GROUP --mtf_filename $MTF_FILENAME --micrograph_original_pixel_size $MICROGRAPH_ORIGINAL_PIXEL_SIZE --voltage $VOLTAGE --spherical_aberration $SPHERICAL_ABERRATION --amplitude_contrast $AMPLITUDE_CONTRAST --image_pixel_size $IMAGE_PIXEL_SIZE --image_size $IMAGE_SIZE --image_dimensionality $IMAGE_DIMENSIONALITY --ctf_data_are_ctf_premultiplied $CTF_DATA_ARE_CTF_PREMULTIPLIED 