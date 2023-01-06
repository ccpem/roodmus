
# script to run parakeet from the command line

### arguments
import argparse
parser = argparse.ArgumentParser(description='Run parakeet')
parser.add_argument("--pdb-dir", type=str, help="path to the directory containing the pdb files")
parser.add_argument("--mrc-dir", type=str, help="path to the directory in which to save the mrc files")
parser.add_argument("-n", "--n-images", type=int, help="number of images to generate", default=1)
parser.add_argument("-m", "--n-molecules", type=int, help="number of molecules to generate in each image", default=1)
cfg = parser.add_argument_group("configuration")
cfg.add_argument("--exposure", help="exposure [e/A^2]", type=float, default=40)
cfg.add_argument("--voltage", help="voltage [kV]", type=float, default=300)
cfg.add_argument("--box_xy", help="box size in xy direction [A]", type=float, default=4000)
cfg.add_argument("--box_z", help="box size in z direction [A]", type=float, default=200)
cfg.add_argument("--pixel_size", help="pixel size [A]", type=float, default=1)
cfg.add_argument("--defocus", help="defocus", type=float, default=[5000], nargs="+")
args = parser.parse_args()

### imports
# general
import os
import numpy as np
from tqdm import tqdm

# parakeet
import parakeet

# roodmus
from configuration import configuration
from orientation_generator import orientation_generator

### functions
def sample_defocus(defocus):
    return np.random.normal(defocus, 200)# value for sigma chosen arbitrarily, could be a parameter

def get_pdb_files(pdb_dir):
    pdb_dir = os.path.abspath(pdb_dir)
    pdb_files = []
    for file in os.listdir(pdb_dir):
        if file.endswith(".pdb"):
            pdb_files.append(os.path.join(pdb_dir, file))
    return pdb_files

def get_instances(pdb_files, n_molecules):
    num_structures = len(pdb_files)
        
    # if the number of structures is less than the number of molecules, we need to repeat some of the structures
    if num_structures < n_molecules:
        # start by adding each structure the same number of times to get as close to the number of molecules as possible
        n_instances = [n_molecules//num_structures]*num_structures
        # then randomly add one to some of the structures until we reach the number of molecules
        for n in range(n_molecules - sum(n_instances)):
            n_instances[np.random.randint(num_structures)] += 1
            
    # if the number of structures is greater than or equal to the number of molecules, the number of instances is 1 and we may need to remove some of the structures
    else:
        # sample the pdb files without replacement
        pdb_files = np.random.choice(pdb_files, n_molecules, replace=False)
        n_instances = [1]*n_molecules
        
    return pdb_files, n_instances
        
### main
def main(args):
    
    ## the main function loops over the number of images to generate. For each image, parakeet is configured and a number of .pdb files is selected. 
    ## The number of instances gets determined based on the number of .pdb files. If there are less .pdb files than molecules to generate, some of the
    ## .pdb files are repeated. If there are more .pdb files than molecules to generate, some of the .pdb files are removed. 
    ## The orientations can be sampled, or left to parakeet to generate.
    ## Then we run parakeet and save the mrc files. From the Sample object we can pull the orientations and positions of the molecules. 
    ## These, along with the optical parameters, are saved in a .yaml file.
    
    ## nomenclature
    ## frame: .pdb file containing the structure of a molecule from a single frame in the MD trajectory
    ## instance: multiplicity of a frame in the simulated image. if there are 10 frames and 100 molecules, each frame is an instance 10 times
    ## molecule: a single particle in the the simulated image
    
    sample_filename = "sample.h5"
    exit_wave_filename = "exit_wave.h5"
    optics_filename = "optics.h5"
    image_filename = "image.h5"
    
    ## loop over the number of images to generate
    progressbar = tqdm(range(args.n_images))
    defocus_idx = 0
    for n_image in range(args.n_images):
        
        mrc_filename = f"{n_image}".zfill(6) + ".mrc"
        mrc_path = os.path.join(args.mrc_dir, mrc_filename) # full path to where the current image will be saved
        config_filename = f"{n_image}".zfill(6) + ".yaml"
        config_path = os.path.join(args.mrc_dir, config_filename) # full path to where the current configuration will be saved
    
        # sample the defocus around the specified value
        defocus = sample_defocus(args.defocus[defocus_idx])
        defocus_idx = (defocus_idx + 1) % len(args.defocus) # in case there are multiple defocus values, we loop over them

        # initialise the configuration
        Config = configuration(
            config_filename=config_path,
            exposure=args.exposure,
            voltage=args.voltage,
            box_xy=args.box_xy,
            box_z=args.box_z,
            pixel_size=args.pixel_size,
            defocus = defocus        
        )
        
        # get the pdb files and determine the number of instances we need of each structure
        frames = get_pdb_files(args.pdb_dir)
        frames, n_instances = get_instances(frames, args.n_molecules)
        
        # add the molecules to the configuration
        Config.add_molecules(frames, n_instances)
            
        # run parakeet
        Sample = parakeet.sample.new(config_path, sample_file=sample_filename)
        Sample = parakeet.sample.add_molecules(config_path, sample_file=sample_filename)
        parakeet.simulate.exit_wave(config_path, sample_filename, exit_wave_file=exit_wave_filename)
        parakeet.simulate.optics(config_path, exit_wave_file=exit_wave_filename, optics_file=optics_filename)
        parakeet.simulate.image(config_path, optics_file=optics_filename, image_file=image_filename)

        # save the image
        os.system(f"parakeet.export image.h5 -o {mrc_path}")
        
        # remove the intermediate files
        os.system(f"rm {sample_filename} {exit_wave_filename} {optics_filename} {image_filename}")
        
        # update the config from the sample
        Config.update_config(Sample)
            
        progressbar.update(1)
        
    progressbar.close()

if __name__ == "__main__":
    main(args)