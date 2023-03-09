"""API containing functions to process the ctf parameters from a .cs or .star file and from the Parakeet config files"""

### imports
# general
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# roodmus
from roodmus.analysis.utils import IO

### CTF estimation
class ctf_estimation(object):
    def __init__(self, meta_file: str, config_dir: str, results: dict=None, verbose: bool=False):
        self.meta_file = meta_file
        self.config_dir = config_dir
        self.verbose = verbose
        if results is not None:
            self.results = results
        else:
            self.results = {
                "ugraph_filename": [], # the path to the micrograph in which the particle was found
                "defocusU": [], # the estimated defocusU
                "defocusV": [], # the estimated defocusV
                "kV": [], # the estimated kV
                "Cs": [], # the estimated Cs
                "amp": [], # the estimated amp
                "Bfac": [], # the estimated Bfac
                "defocus_truth": [], # the true defocus
                "kV_truth": [], # the true kV
                "Cs_truth": [], # the true Cs
            }
        self.compute()

    def compute(self, meta_file: str=None, config_dir: str=None, verbose: bool=None):
        """processing of the estimated and true ctf parameters from a .cs or .star file and from the Parakeet config files. 
        The ctf parameter values for each particle are read from the metadata file and the config files and stored in a dictionary.
        If the user has provided new arguments, the class attributes are updated and the computation is performed again.
        
        Args:
            meta_file (str, optional): path to the metadata file. Defaults to None.
            config_dir (str, optional): path to the directory containing the config files. Defaults to None.
            verbose (bool, optional): whether to print the progress. Defaults to None.
        
        Returns:
            dict: dictionary containing the ctf parameters for each particle
        """	
        
        # if the user has provided new arguments, use them
        if meta_file is not None:
            self.meta_file = meta_file
        if config_dir is not None:
            self.config_dir = config_dir
        if verbose is not None:
            self.verbose = verbose

        # load the metadata
        if self.verbose:
            print(f"loading metadata from {self.meta_file}...")
        metadata, file_type = self._load_metadata(self.meta_file)
        ugraph_paths, ctf = self._extract_from_metadata(metadata, file_type)
        if self.verbose:
            print(f"found {len(ctf)} particles in {len(np.unique(ugraph_paths))} micrographs")

        # load the config files    
        progressbar = tqdm(total=len(np.unique(ugraph_paths)), desc="loading ground-truth ctf parameters")
        gt_ctf = []
        for ugraph_path in np.unique(ugraph_paths):
            num_particles_in_ugraph = np.sum(np.array(ugraph_paths) == ugraph_path)
            config = IO.load_config(os.path.join(self.config_dir, ugraph_path.replace(".mrc", ".yaml")))
            gt_ctf_ugraph = self._extract_from_config(config) # single value for the entire micrograph
            _ = [gt_ctf.append(gt_ctf_ugraph) for _ in range(num_particles_in_ugraph)]
            _ = progressbar.update(1)
        progressbar.close()
        gt_ctf = np.array(gt_ctf)  
        gt_ctf = np.abs(gt_ctf) # the defocus values are negative in the config file, but positive in the metadata file 
        if self.verbose:
            print(f"ground-truth ctf values loaded. {gt_ctf.shape} ctf values found")
            print(f"estimated ctf values loaded. {ctf.shape} ctf values found")

        for i in range(len(ugraph_paths)):
            self.results["ugraph_filename"].append(ugraph_paths[i])
            self.results["defocusU"].append(ctf[i,0])
            self.results["defocusV"].append(ctf[i,1])
            self.results["kV"].append(ctf[i,2])
            self.results["Cs"].append(ctf[i,3])
            self.results["amp"].append(ctf[i,4])
            self.results["Bfac"].append(ctf[i,5])
            self.results["defocus_truth"].append(gt_ctf[i,0])
            self.results["kV_truth"].append(gt_ctf[i,1])
            self.results["Cs_truth"].append(gt_ctf[i,2])
    
    def _extract_from_metadata(self, metadata, file_type):
        if file_type == "cs":
            ugraph_paths = IO.get_ugraph_cs(metadata) # a list of all microraps in the metadata file
            ctf = IO.get_ctf_cs(metadata) # an array of all the defocus values in the metadata file
        elif file_type == "star":
            ugraph_paths = IO.get_ugraph_star(metadata)
            ctf = IO.get_ctf_star(metadata)
        else:
            raise ValueError(f"unknown metadata file type: {file_type}")
        return ugraph_paths, ctf

    def _extract_from_config(self, config):
        defocus = config["microscope"]["lens"]["c_10"] # in Angstroms
        kV = config["microscope"]["beam"]["energy"] # in kV
        Cs = config["microscope"]["lens"]["c_30"] # in mm
        amp = 0 # not implemented in Parakeet as far as I know
        Bfac = 0 # not implemented in Parakeet as far as I know
        return np.array([defocus, kV, Cs, amp, Bfac])
    
    def _load_metadata(self, meta_file):
        if meta_file.endswith(".star"):
            metadata = IO.load_star(meta_file)
            file_type = "star"
        elif meta_file.endswith(".cs"):
            metadata = IO.load_cs(meta_file)
            file_type = "cs"
        else:
            raise ValueError(f"unknown metadata file type: {meta_file}")
        return metadata, file_type

