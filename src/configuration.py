
# configutation class to setup the parameters for Parakeet

### imports
# general
import yaml
import numpy as np

# parakeet
from parakeet import config

# roodmus
from orientation_generator import orientation_generator

### configuration class
class configuration(object):
    def __init__(self, config_filename, exposure=100,  voltage=300, box_xy=4000, box_z=200, pixel_size=1, defocus=-5000):
        self.exposure = exposure            # total electron dose in e/A^2
        self.voltage = voltage              # electron beam voltage in kV
        self.box_xy = box_xy                # detector and sample box size in A
        self.box_z = box_z                  # thickness of the ice layer in A
        self.pixel_size = pixel_size        # pixel size in A
        self.defocus = defocus              # average defocus in A
        
        # initialise the config file for Parakeet
        self.config_filename = config_filename
        self.Config = config.new(filename = config_filename, full=True)
        self._set_config()
        
    def _set_config(self):
        self.Config.microscope.beam.electrons_per_angstrom = self.exposure
        self.Config.microscope.beam.energy = self.voltage
        self.Config.microscope.detector.nx = self.box_xy
        self.Config.microscope.detector.ny = self.box_xy
        self.Config.microscope.detector.pixel_size = self.pixel_size
        self.Config.microscope.lens.c_10 = self.defocus
        self.Config.sample.box = [self.box_xy*self.pixel_size, self.box_xy*self.pixel_size, self.box_z]
        self.Config.sample.centre = [self.box_xy*self.pixel_size/2, self.box_xy*self.pixel_size/2, self.box_z/2]
        self.Config.sample.molecules = config.Molecules()
        self.Config.sample.molecules.local = []
        self.Config.sample.shape.cuboid.length_x = self.box_xy*self.pixel_size
        self.Config.sample.shape.cuboid.length_y = self.box_xy*self.pixel_size
        self.Config.sample.shape.cuboid.length_z = self.box_z
        self.Config.sample.shape.type = "cuboid"
        
    def _save_config(self):
        with open(self.config_filename, "w") as f:
            yaml.safe_dump(self.Config.dict(), f)
        
    def _add_molecule(self, pdb_file, n=1, position=[], orientation=[]):
        """
        add a molecule to the configuration file
        
        pdb_file: str
            path to the pdb file
        n: int (default: 1)
            number of instances of the frame
        position: list (default: [])
            positions of the molecule in A
        orientation: list (default: [])
            orientations of the molecule in degrees
        """
        
        ## the molecule can be specified by only the number of instances, in which case parakeet generates the orientation and position
        ## or by the position and orientation, in which case the number of instances is set to 1
        
        # if the number of instances is specified and the position and orientation are not specified
        if not position and not orientation:
            instance = n

        # if the orientation and position are specified            
        elif position and orientation:
            instance = []
            for p, o in zip(position, orientation):
                pose = config.MoleculePose()
                pose.position = p
                pose.orientation = o
                instance.append(pose)
            
        # if the orientation is specified but not the position
        elif orientation and not position:
            instance = []
            for o in orientation:
                pose = config.MoleculePose()
                pose.orientation = o
                instance.append(pose)
            
        # if the position is specified but not the orientation
        elif position and not orientation:
            instance = []
            for p in position:
                pose = config.MoleculePose()
                pose.position = p
                instance.append(pose)
            
        # if none of the above, raise an error
        else:
            raise ValueError("At least one of the following must be specified: n, position, orientation")
        
        self.Config.sample.molecules.local.append(
            config.LocalMolecule(filename=pdb_file, instances=instance)
        )

    def add_molecules(self, frames, instances):
        for frame, instance in zip(frames, instances):
            self._add_molecule(frame, n=instance, orientation=orientation_generator.generate_inplane())
            
        self._save_config()

    def update_config(self, Sample):
        """
        updates the configuration file with the generated positions and orientations
        
        Sample: parakeet.sample.Sample
            sample object from Parakeet
        """

        frame_idx = 0
        for molecule in Sample.molecules:
            _, positions, orientations = Sample.get_molecule(molecule)

            self.Config.sample.molecules.local[frame_idx].instances = []

            for position, orientation in zip(positions, orientations):
                self.Config.sample.molecules.local[frame_idx].instances.append(config.MoleculePose())
                self.Config.sample.molecules.local[frame_idx].instances[-1].position = [float(p) for p in position]
                
                # the orientation can contain negative values, which are not allowed in the configuration file. 
                self.Config.sample.molecules.local[frame_idx].instances[-1].orientation = [float((o+(2*np.pi))%(2*np.pi)) for o in orientation]
                
                
            frame_idx += 1
       
        self._save_config()

