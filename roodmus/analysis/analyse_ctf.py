"""script to compare estimated CTF values from RELION or CryoSPARC to the ground-truth CTF values used in Parakeet data generation"""

### arguments
def add_arguments(parser):
    parser.add_argument("--mrc-dir", help="directory with .mrc files and .yaml config files", type=str)
    parser.add_argument("--meta-file", help="particle metadata file. Can be .star (RELION) or .cs (CryoSPARC)", type=str)
    parser.add_argument("--plot-dir", help="output file name", type=str, default="ctf.png")
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    return parser

def get_name():
    return "analyse_ctf"

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
        metadata = IO.load_star(meta_file)
        file_type = "star"
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
        ugraph_paths = IO.get_ugraph_star(metadata)
        ctf = IO.get_ctf_star(metadata)
    else:
        raise ValueError(f"unknown metadata file type: {file_type}")
    return ugraph_paths, ctf

def extract_from_config(config):
    defocusU = config["microscope"]["lens"]["c_10"] # in Angstroms
    defocusV = defocusU
    kV = config["microscope"]["beam"]["energy"] # in kV
    Cs = config["microscope"]["lens"]["c_30"] # in mm
    amp = 0 # not implemented in Parakeet as far as I know
    Bfac = 0 # not implemented in Parakeet as far as I know
    return np.array([defocusU, defocusV, kV, Cs, amp, Bfac])
    
### plotting functions
def plot_defocus_scatter(gt_ctf, ctf):
    fig, ax = plt.subplots(1, 2, figsize=(10,5), sharey=True)
    ax[0].scatter(gt_ctf[:,0], ctf[:,0], label="defocusU")
    ax[0].plot([min(gt_ctf[:,0])*0.9, max(gt_ctf[:,0])*1.1], [min(gt_ctf[:,0])*0.9, max(gt_ctf[:,0])*1.1], color="black", linestyle="--", alpha=0.5)
    ax[1].scatter(gt_ctf[:,1], ctf[:,1], label="defocusV")
    ax[1].plot([min(gt_ctf[:,1])*0.9, max(gt_ctf[:,1])*1.1], [min(gt_ctf[:,1])*0.9, max(gt_ctf[:,1])*1.1], color="black", linestyle="--", alpha=0.5)
    ax[0].grid()
    ax[1].grid()
    ax[0].set(xlabel="ground-truth defocus (A)", ylabel="estimated defocus (A)", title="defocusU")
    ax[1].set(xlabel="ground-truth defocus (A)", title="defocusV")
    return fig, ax

### main
def main(args):
    ## the script loads the metadata file and extracts the ctf parameters for each particle. It is possible that the CTF values for each particle are the same
    ## if the CTF estimation was done on the entire micrograph. In that case, the script will plot the CTF values for each micrograph. 
    ## Next, the script loads the config parakeet file and gets the CTF values for each micrograph. 
    ## for each micrograph that is present in the metadata, the script will plot the estimated CTF values against the ground-truth CTF values.
    
    metadata, file_type = load_metadata(args.meta_file)
    print("metadata loaded. extracting ctf values from metadata file")
    ugraph_paths, ctf = extract_from_metadata(metadata, file_type) # get the micrograph paths and the ctf values for each particle
    progressbar = tqdm(total=len(np.unique(ugraph_paths)), desc="loading ground-truth ctf values")
    gt_ctf = []
    for ugraph_path in np.unique(ugraph_paths):
        num_particles_in_ugraph = np.sum(np.array(ugraph_paths) == ugraph_path)
        config = IO.load_config(os.path.join(args.mrc_dir, ugraph_path.replace(".mrc", ".yaml")))
        gt_ctf_ugraph = extract_from_config(config) # single value for the entire micrograph
        _ = [gt_ctf.append(gt_ctf_ugraph) for _ in range(num_particles_in_ugraph)]
        _ = progressbar.update(1)
    progressbar.close()
    gt_ctf = np.array(gt_ctf)  
    gt_ctf = np.abs(gt_ctf) # the defocus values are negative in the config file, but positive in the metadata file 
    print(f"ground-truth ctf values loaded. {gt_ctf.shape} ctf values found")
    print(f"estimated ctf values loaded. {ctf.shape} ctf values found")
        
    ## plot the ground truth defocus values against the estimated defocus values
    fig, ax = plot_defocus_scatter(gt_ctf, ctf)
    fig.savefig(os.path.join(args.plot_dir, f"defocus_scatter_{file_type}.png"))
    plt.close(fig)
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print('{}, {}'.format(arg, getattr(args, arg)))
    main(args)

