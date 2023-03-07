
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
        if "ctf/df1_A" in metadata_cs.dtype.names:
            defocusU = metadata_cs["ctf/df1_A"]
            defocusV = metadata_cs["ctf/df2_A"]
            kV = metadata_cs["ctf/accel_kv"]
            Cs = metadata_cs["ctf/cs_mm"]
            amp = metadata_cs["ctf/amp_contrast"]
            Bfac = metadata_cs["ctf/bfactor"]
            return np.stack([defocusU, defocusV, kV, Cs, amp, Bfac], axis=1)
        else:
            return None
 
    @classmethod
    def get_positions_cs(self, metadata_cs):
        ugraph_shape = metadata_cs["location/micrograph_shape"]
        # print(type(ugraph_shape), len(ugraph_shape), ugraph_shape.shape)
        x = metadata_cs["location/center_x_frac"]
        y = metadata_cs["location/center_y_frac"]
        # convert to absolute coordinates
        x_abs = x * ugraph_shape[0,0] # assuming all micrographs have the same shape
        y_abs = y * ugraph_shape[0,1]
        # conver to single array
        pos = np.stack([x_abs, y_abs], axis=1)    
        return pos
    
    @classmethod
    def get_ugraph_shape_cs(self, metadata_cs):
        ugraph_shape = metadata_cs["location/micrograph_shape"]
        return ugraph_shape

    # loading .star files and parsing the ctf parameters, the particle positions and orientations
    @classmethod
    def load_star(self, star_path):
         return RelionStarFile(star_path)
    
    @classmethod
    def get_ugraph_star(self, metadata_star):
        particles = metadata_star.get_block("particles")
        ugraph_paths = metadata_star.column_as_list("particles", "_rlnMicrographName")
        # convert to basename and remove index
        ugraph_paths = [os.path.basename(path).split("_")[-1] for path in ugraph_paths]
        return ugraph_paths

    @classmethod
    def get_ctf_star(self, metadata_star):
        kV = [float(r) for r in metadata_star.column_as_list("optics", "_rlnVoltage")]
        Cs = [float(r) for r in metadata_star.column_as_list("optics", "_rlnSphericalAberration")]
        amp = [float(r) for r in metadata_star.column_as_list("optics", "_rlnAmplitudeContrast")]

        defocusU = [float(r) for r in metadata_star.column_as_list("particles", "_rlnDefocusU")]
        defocusV = [float(r) for r in metadata_star.column_as_list("particles", "_rlnDefocusV")]
        return np.stack([defocusU, defocusV, kV*len(defocusU), Cs*len(defocusU), amp*len(defocusU)], axis=1)

    @classmethod
    def get_positions_star(self, metadata_star):
        x = [float(r) for r in metadata_star.column_as_list("particles", "_rlnCoordinateX")]
        y = [float(r) for r in metadata_star.column_as_list("particles", "_rlnCoordinateY")]
        pos = np.stack([x, y], axis=1)
        return pos

    # loading the config file
    @classmethod
    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

