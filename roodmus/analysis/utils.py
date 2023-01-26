
### imports
# general
import os
import yaml
import numpy as np
from typing import Optional
from gemmi import cif

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

    @classmethod
    def load_picked_particles_from_starfile(args, starfile: str, block_name: Optional[str]="particles", loop_name_prefix: Optional[str]="_rln")->dict:
        if os.path.exists(starfile):
            # load the starfile via gemmi 
            gemmi_block = cif.read_file(starfile)[block_name]
            """ 
            Example loop headings
            loop_ 
            _rlnCoordinateX #1 
            _rlnCoordinateY #2 
            _rlnAutopickFigureOfMerit #3 
            _rlnClassNumber #4 
            _rlnAnglePsi #5 
            _rlnImageName #6 
            _rlnMicrographName #7 
            _rlnOpticsGroup #8 
            _rlnCtfMaxResolution #9 
            _rlnCtfFigureOfMerit #10 
            _rlnDefocusU #11 
            _rlnDefocusV #12 
            _rlnDefocusAngle #13 
            _rlnCtfBfactor #14 
            _rlnCtfScalefactor #15 
            _rlnPhaseShift #16 
            """
            # create a dict to read these picked particles into
            # make this indexed by micrograph index
            picked_particles = {}

            CoordinateX = gemmi_block.find_values(loop_name_prefix+"CoordinateX")
            CoordinateY = gemmi_block.find_values(loop_name_prefix+"CoordinateY")
            AutopickFigureOfMerit = gemmi_block.find_values(loop_name_prefix+"AutopickFigureOfMerit")
            ClassNumber = gemmi_block.find_values(loop_name_prefix+"ClassNumber")
            AnglePsi = gemmi_block.find_values(loop_name_prefix+"AnglePsi")
            ImageName = gemmi_block.find_values(loop_name_prefix+"ImageName")
            MicrographName = gemmi_block.find_values(loop_name_prefix+"MicrographName")
            OpticsGroup = gemmi_block.find_values(loop_name_prefix+"OpticsGroup")
            CtfMaxResolution = gemmi_block.find_values(loop_name_prefix+"CtfMaxResolution")
            CtfFigureOfMerit = gemmi_block.find_values(loop_name_prefix+"CtfFigureOfMerit")
            DefocusU = gemmi_block.find_values(loop_name_prefix+"DefocusU")
            DefocusV = gemmi_block.find_values(loop_name_prefix+"DefocusV")
            DefocusAngle = gemmi_block.find_values(loop_name_prefix+"DefocusAngle")
            CtfBfactor = gemmi_block.find_values(loop_name_prefix+"CtfBfactor")
            CtfScalefactor = gemmi_block.find_values(loop_name_prefix+"CtfScalefactor")
            PhaseShift = gemmi_block.find_values(loop_name_prefix+"PhaseShift")
            
            # find the unique micrographs and create a dict with the
            # micrograph indices as keys
            # convert all entries to string instead of raw_string also
            ugraph_names = []
            for i in range(len(MicrographName)):
                ugraph_names.append(MicrographName.str(i))
            ugraph_names = np.array(ugraph_names, dtype=str)
            # get the unique names and sort them
            unique_ugraphs = sorted(np.unique(ugraph_names).tolist())
            # take out only the basename and grab the index only (as string!)
            for i in range(len(unique_ugraphs)):
                unique_ugraphs[i]=os.path.basename(unique_ugraphs[i])[len(args.basename_prefix):-len(args.basename_suffix)]
                # create a dict to hold particle property lists
                picked_particles[unique_ugraphs[i]] = {}
                picked_particles[unique_ugraphs[i]]["CoordinateX"] = []
                picked_particles[unique_ugraphs[i]]["CoordinateY"] = []
                picked_particles[unique_ugraphs[i]]["AutopickFigureOfMerit"] = []
                picked_particles[unique_ugraphs[i]]["ClassNumber"] = []
                picked_particles[unique_ugraphs[i]]["AnglePsi"] = []
                picked_particles[unique_ugraphs[i]]["ImageName"] = []
                picked_particles[unique_ugraphs[i]]["MicrographName"] = []
                picked_particles[unique_ugraphs[i]]["OpticsGroup"] = []
                picked_particles[unique_ugraphs[i]]["CtfMaxResolution"] = []
                picked_particles[unique_ugraphs[i]]["CtfFigureOfMerit"] = []
                picked_particles[unique_ugraphs[i]]["DefocusU"] = []
                picked_particles[unique_ugraphs[i]]["DefocusV"] = []
                picked_particles[unique_ugraphs[i]]["DefocusAngle"] = []
                picked_particles[unique_ugraphs[i]]["CtfBfactor"] = []
                picked_particles[unique_ugraphs[i]]["CtfScalefactor"] = []
                picked_particles[unique_ugraphs[i]]["PhaseShift"] = []

            # fill the dict lists with particle dicts
            for i in range(len(CoordinateX)):
                # grab the micrograph name and decipher which dict to add this particle to
                ugraph_index = os.path.basename(MicrographName.str(i))[len(args.basename_prefix):-len(args.basename_suffix)]

                picked_particles[ugraph_index]["CoordinateX"].append(CoordinateX.str(i))
                picked_particles[ugraph_index]["CoordinateY"].append(CoordinateY.str(i))
                picked_particles[ugraph_index]["AutopickFigureOfMerit"].append(AutopickFigureOfMerit.str(i))
                picked_particles[ugraph_index]["ClassNumber"].append(ClassNumber.str(i))
                picked_particles[ugraph_index]["AnglePsi"].append(AnglePsi.str(i))
                picked_particles[ugraph_index]["ImageName"].append(ImageName.str(i))
                picked_particles[ugraph_index]["MicrographName"].append(MicrographName.str(i))
                picked_particles[ugraph_index]["OpticsGroup"].append(OpticsGroup.str(i))
                picked_particles[ugraph_index]["CtfMaxResolution"].append(CtfMaxResolution.str(i))
                picked_particles[ugraph_index]["CtfFigureOfMerit"].append(CtfFigureOfMerit.str(i))
                picked_particles[ugraph_index]["DefocusU"].append(DefocusU.str(i))
                picked_particles[ugraph_index]["DefocusV"].append(DefocusV.str(i))
                picked_particles[ugraph_index]["DefocusAngle"].append(DefocusAngle.str(i))
                picked_particles[ugraph_index]["CtfBfactor"].append(CtfBfactor.str(i))
                picked_particles[ugraph_index]["CtfScalefactor"].append(CtfScalefactor.str(i))
                picked_particles[ugraph_index]["PhaseShift"].append(PhaseShift.str(i))

            for ugraph_index in picked_particles.keys():
                picked_particles[ugraph_index]["CoordinateX"] = np.array(picked_particles[ugraph_index]["CoordinateX"], dtype=str).astype(float).tolist()
                picked_particles[ugraph_index]["CoordinateY"] =  np.array(picked_particles[ugraph_index]["CoordinateY"], dtype=str).astype(float).tolist()
                picked_particles[ugraph_index]["AutopickFigureOfMerit"] = np.array(picked_particles[ugraph_index]["AutopickFigureOfMerit"], dtype=str).astype(float).tolist()
                picked_particles[ugraph_index]["ClassNumber"] = np.array(picked_particles[ugraph_index]["ClassNumber"], dtype=str).astype(int).tolist()
                picked_particles[ugraph_index]["AnglePsi"] = np.array(picked_particles[ugraph_index]["AnglePsi"], dtype=str).astype(float).tolist()
                picked_particles[ugraph_index]["ImageName"] = np.array(picked_particles[ugraph_index]["ImageName"], dtype=str).astype(str).tolist()
                picked_particles[ugraph_index]["MicrographName"] = np.array(picked_particles[ugraph_index]["MicrographName"], dtype=str).astype(str).tolist()
                picked_particles[ugraph_index]["OpticsGroup"] = np.array(picked_particles[ugraph_index]["OpticsGroup"], dtype=str).astype(int).tolist()
                picked_particles[ugraph_index]["CtfMaxResolution"] = np.array(picked_particles[ugraph_index]["CtfMaxResolution"], dtype=str).astype(float).tolist()
                picked_particles[ugraph_index]["CtfFigureOfMerit"] = np.array(picked_particles[ugraph_index]["CtfFigureOfMerit"], dtype=str).astype(float).tolist()
                picked_particles[ugraph_index]["DefocusU"] = np.array(picked_particles[ugraph_index]["DefocusU"], dtype=str).astype(float).tolist()
                picked_particles[ugraph_index]["DefocusV"] = np.array(picked_particles[ugraph_index]["DefocusV"], dtype=str).astype(float).tolist()
                picked_particles[ugraph_index]["DefocusAngle"] = np.array(picked_particles[ugraph_index]["DefocusAngle"], dtype=str).astype(float).tolist()
                picked_particles[ugraph_index]["CtfBfactor"] = np.array(picked_particles[ugraph_index]["CtfBfactor"], dtype=str).astype(float).tolist()
                picked_particles[ugraph_index]["CtfScalefactor"] = np.array(picked_particles[ugraph_index]["CtfScalefactor"], dtype=str).astype(float).tolist()
                picked_particles[ugraph_index]["PhaseShift"] = np.array(picked_particles[ugraph_index]["PhaseShift"], dtype=str).astype(float).tolist()

        else:
            print('File:\n{}Does not exist! Exiting!'.format(starfile))
            exit(1)

        return picked_particles

