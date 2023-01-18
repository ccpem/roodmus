
# Description: extract particle stacks from an mrc file

### arguments
import argparse
parser = argparse.ArgumentParser(description='extract particle stacks from an mrc file')
parser.add_argument("--mrc-dir", type=str, help="path to the directory containing the mrc files")
parser.add_argument("--out-dir", type=str, help="path to the directory in which to save the particle stacks")
parser.add_argument("-L", "--box-size", type=int, help="box size in pixels", default=128)
args = parser.parse_args()

### imports
# general
import os
import mrcfile
import yaml
import numpy as np
from tqdm import tqdm

### functions
def get_mrc_files(mrc_dir):
    mrc_files = []
    for file in os.listdir(mrc_dir):
        if file.endswith(".mrc"):
            mrc_files.append(os.path.join(mrc_dir, file))
    return mrc_files

def get_num_molecules(metadata):
    num_molecules = 0
    for molecule in metadata["sample"]["molecules"]["local"]:
        if type(molecule["instances"]) == int:
            num_molecules += int(molecule["instances"])
        elif type(molecule["instances"]) == list:
            num_molecules += len(molecule["instances"])
    return num_molecules

def get_particle(micrograph, position, box_size):
    particle = micrograph[0,
                          position[1]-box_size//2:position[1]+box_size//2,
                          position[0]-box_size//2:position[0]+box_size//2]
    return particle

### main
def main(args):
    # make output dir if it does not exist
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    # get the mrc files
    mrc_files = get_mrc_files(args.mrc_dir)
    
    # loop over the mrc files and load the micrographs
    for mrc_file in mrc_files:
        micrograph = mrcfile.open(mrc_file).data
        metadata_file = mrc_file.replace(".mrc", ".yaml")
        metadata = yaml.load(open(metadata_file, "r"), Loader=yaml.FullLoader)
        
        # extract the particles from the micrograph
        particle_list = []
        progressbar = tqdm(total=get_num_molecules(metadata))
        for molecule in metadata["sample"]["molecules"]["local"]:
            for instance in molecule["instances"]:
                position = instance["position"]
                position = [int(float(r)) for r in position]
                
                particle = get_particle(micrograph, position, args.box_size)
                
                # if the particle is not the right size, skip it
                if particle.shape != (args.box_size, args.box_size):
                    _ = progressbar.update(1)
                    continue
                
                particle_list.append(particle)
                _ = progressbar.update(1)

        progressbar.close()            
        particle_stack = np.array(particle_list)
        print(f"particle stack shape: {particle_stack.shape}")
        
        # save the particle stack
        particle_file = os.path.join(args.out_dir, os.path.basename(mrc_file.replace(".mrc", "_particles.mrc")))
        with mrcfile.new(particle_file, overwrite=True) as mrc:
            mrc.set_data(np.float32(particle_stack))

if __name__ == "__main__":
    main(args)