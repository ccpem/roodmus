
### imports
# general
import os
import yaml
import numpy as np
# pipeliner
from pipeliner.jobstar_reader import RelionStarFile

### i/o functions
class IO(object):
    ## class containing several functions to load metadata from .star (RELION) and .cs (CryoSPARC) files 
    ## and the config file generated during the Parakeet simulation
    
    
    ## loading .cs files and parsing the ctf parameters, the particle positions and orientations
    @classmethod
    def load_cs(self, cs_path):
        metadata = np.load(cs_path)
        return metadata

    @classmethod
    def get_ugraph_cs(self, metadata_cs):
        ugraph_paths = metadata_cs["location/micrograph_path"]
        # convert to basename and remove index
        ugraph_paths = [os.path.basename(path).decode("utf-8").split("_")[-1] for path in ugraph_paths]
        return ugraph_paths

    @classmethod
    def get_ctf_cs(self, metadata_cs):
        defocusU = metadata_cs["ctf/df1_A"]
        defocusV = metadata_cs["ctf/df2_A"]
        kV = metadata_cs["ctf/accel_kv"]
        Cs = metadata_cs["ctf/cs_mm"]
        amp = metadata_cs["ctf/amp_contrast"]
        Bfac = metadata_cs["ctf/bfactor"]
        return np.stack([defocusU, defocusV, kV, Cs, amp, Bfac], axis=1)

    @classmethod
    def get_position_cs(self, metadata_cs):
        ugraph_shape = metadata_cs["location/micrograph_shape"]
        print(type(ugraph_shape), len(ugraph_shape), ugraph_shape.shape)
        x = metadata_cs["location/center_x_frac"]
        y = metadata_cs["location/center_y_frac"]
        # convert to absolute coordinates
        x_abs = x * ugraph_shape[0,0] # assuming all micrographs have the same shape
        y_abs = y * ugraph_shape[0,1]
        # conver to single array
        pos = np.stack([x_abs, y_abs], axis=1)    
        return pos

    # loading .star files and parsing the ctf parameters, the particle positions and orientations
    def load_star(self, star_path):
         return RelionStarFile(star_path)
        
    def get_ugraph_star(self, metadata_star):
        particles = metadata_star.get_block("particles")
        ugraph_paths = particles.column_as_list("_rlnMicrographName")
        # convert to basename and remove index
        ugraph_paths = [os.path.basename(path).decode("utf-8").split("_")[-1] for path in ugraph_paths]
        return ugraph_paths
    
    def get_ctf_star(self, metadata_star):
        optics = metadata_star.get_block("optics")
        kV = optics.column_as_list("_rlnVoltage")
        Cs = optics.column_as_list("_rlnSphericalAberration")
        amp = optics.column_as_list("_rlnAmplitudeContrast")

        particles = metadata_star.get_block("particles")
        defocusU = particles.column_as_list("_rlnDefocusU")
        defocusV = particles.column_as_list("_rlnDefocusV")
        return np.stack([defocusU, defocusV, kV, Cs, amp], axis=1)

    # loading the config file
    @classmethod
    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

