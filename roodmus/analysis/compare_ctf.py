"""script to compare estimated CTF values from RELION or CryoSPARC to the ground-truth CTF values used in Parakeet data generation"""

### arguments
def add_arguments(parser):
    parser.add_argument("--mrc-dir", help="directory with .mrc files and .yaml config files", type=str)
    parser.add_argument("--meta-file", help="particle metadata file. Can be .star (RELION) or .cs (CryoSPARC)", type=str)
    parser.add_argument("--plot-file", help="output file name", type=str, default="ctf.png")
    return parser

def get_name():
    return "compare_ctf"

### imports
# general
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# roodmus
from roodmus.analysis.utils import IO

### functions
def load_metadata(meta_file):
    if meta_file.endswith(".star"):
        raise NotImplementedError
        # metadata = IO.load_relion_star(meta_file)
    elif meta_file.endswith(".cs"):
        metadata = IO.load_cs(meta_file)
        file_type = "cs"
    else:
        raise ValueError(f"unknown metadata file type: {meta_file}")
    return metadata, file_type

def extract_from_metadata(metadata, file_type):
    if file_type == "cs":
        ugraph_paths = IO.get_ugraph_cs(metadata) # a list of all microraps in the metadata file
        ctf = IO.get_ctf_cs(metadata) # an array of all the defocus values in the metadata file
    elif file_type == "star":
        raise NotImplementedError
    else:
        raise ValueError(f"unknown metadata file type: {file_type}")
    return ugraph_paths, ctf

def extract_from_config(config):
    defocus = config["microscope"]["lens"]["c_10"] # in Angstroms
    # the defous in Parakeet is a single value that is used for all particles in the micrograph
    return defocus
    

### main
def main(args):
    ## the script loads the metadata file and extracts the ctf parameters for each particle. It is possible that the CTF values for each particle are the same
    ## if the CTF estimation was done on the entire micrograph. In that case, the script will plot the CTF values for each micrograph. 
    ## Next, the script loads the config parakeet file and gets the CTF values for each micrograph. 
    ## for each micrograph that is present in the metadata, the script will plot the estimated CTF values against the ground-truth CTF values.
    
    metadata, file_type = load_metadata(args.meta_file)
    print("metadata loaded. extracting ctf values from metadata file")
    ugraph_paths, ctf = extract_from_metadata(metadata, file_type) # get the micrograph paths and the ctf values for each particle
    gt_ctf = [] # list of ground-truth ctf values
    progressbar = tqdm(total=len(np.unique(ugraph_paths)), desc="loading ground-truth ctf values")
    for ugraph_path in np.unique(ugraph_paths):
        num_particles_in_ugraph = np.sum(np.array(ugraph_paths) == ugraph_path)
        config = IO.load_config(os.path.join(args.mrc_dir, ugraph_path.replace(".mrc", ".yaml")))
        gt_defocus = extract_from_config(config) # single value
        gt_defocus = [gt_defocus] * num_particles_in_ugraph # convert to list
        gt_ctf += gt_defocus # add to list
        _ = progressbar.update(1)
    progressbar.close()
        
    ## plot the ground truth defocus values against the estimated defocus values
    fig, ax = plt.subplots(1, 2, figsize=(10,5), sharey=True)
    ax[0].scatter(gt_ctf, ctf[:,0], label="defocusU")
    ax[0].plot([min(gt_ctf)*0.9, max(gt_ctf)*1.1], [min(gt_ctf)*0.9, max(gt_ctf)*1.1], color="black", linestyle="--", alpha=0.5)
    ax[1].scatter(gt_ctf, ctf[:,1], label="defocusV")
    ax[1].plot([min(gt_ctf)*0.9, max(gt_ctf)*1.1], [min(gt_ctf)*0.9, max(gt_ctf)*1.1], color="black", linestyle="--", alpha=0.5)
    ax[0].grid()
    ax[1].grid()
    ax[0].set(xlabel="ground-truth defocus (A)", ylabel="estimated defocus (A)", title="defocusU")
    ax[1].set(xlabel="ground-truth defocus (A)", title="defocusV")
    fig.savefig(args.plot_file)
    plt.close(fig)
    
if __name__ == "__main__":
    main(args)

