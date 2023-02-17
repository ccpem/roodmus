TRAJFILES_DIR_PATH=$"/mnt/parakeet_storage/trajectories/DESRES-Trajectory_sarscov2-13795965-no-water/sarscov2-13795965-no-water"
TOPFILE_PATH=$"/mnt/parakeet_storage/trajectories/DESRES-Trajectory_sarscov2-13795965-no-water/sarscov2-13795965-no-water/DESRES-Trajectory_sarscov2-13795965-no-water.pdb"
#DEBUG is toggled on by use of --debug
SAMPLING_METHOD=$"even_sampling"
N_CONFORMATIONS=5
LIMIT_N_TRAJ_SUBFILES=1
TRAJ_EXTENSION=$".dcd"
OUTPUT_DIR=$"waymark_example"
RANDOM_STARTPOINT_SEED=99
# RND_START toggled on if --rnd_start is passed as arg
# USE_CONTIGUOUS_CONFORMATIONS= toggled on if use_contiguous_conformations is pased as arg
# INVESTIGATE_TRAJECTORY_FILES toggled on if --investigate_trajectory_files is passed as arg
#RMSD=0.5

nohup python roodmus.py waymarking --trajfiles_dir_path $TRAJFILES_DIR_PATH --topfile_path $TOPFILE_PATH --debug --sampling_method $SAMPLING_METHOD --n_conformations $N_CONFORMATIONS --limit_n_traj_subfiles $LIMIT_N_TRAJ_SUBFILES --traj_extension $TRAJ_EXTENSION --output_dir $OUTPUT_DIR --random_startpoint_seed $RANDOM_STARTPOINT_SEED --rnd_start --use_contiguous_conformations > waymark_example.log 2>&1 &
