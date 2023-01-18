
### imports
# general
import os
import yaml
import numpy as np

### i/o functions
class IO(object):
    ## class containing several functions to load metadata from .star (RELION) and .cs (CryoSPARC) files 
    ## and the config file generated during the Parakeet simulation
    
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
        return np.stack([defocusU, defocusV], axis=1)

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

    @classmethod
    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config

