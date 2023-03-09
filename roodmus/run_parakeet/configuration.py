
# configutation class to setup the parameters for Parakeet
import os
import argparse

import yaml
import numpy as np

from parakeet import config
from roodmus.run_parakeet.orientation_generator import orientation_generator

### configuration class
class configuration(object):
    def __init__(self, config_filename, args=None):
        # intermediate file names
        self.sample_filename = "sample.h5"
        self.exit_wave_filename = "exit_wave.h5"
        self.optics_filename = "optics.h5"
        self.image_filename = "image.h5"

        # default value for the leading zeros
        self.leading_zeros = 6

        # initialise the config file for Parakeet
        self.config_filename = config_filename

        self.config = config.new(filename = self.config_filename, full=True)
        if args:
            self.sample_filename = "sample.h5"
            self.exit_wave_filename = os.path.join(args.mrc_dir, "exit_wave.h5")
            self.optics_filename = os.path.join(args.mrc_dir, "optics.h5")
            self.image_filename = os.path.join(args.mrc_dir, "image.h5")
            self.leading_zeros = args.leading_zeros
            self._set_config(args)
        
    def _set_config(self, args: argparse.ArgumentParser):
        """Set the general configuration parameters using user-provided
        inputs and/or defaults 

        Args:
            args (argparse.ArgumentParser): Parsed arguments and defaults
        """
        # cluster
        self.config.cluster.method = args.method
        self.config.cluster.max_workers = args.max_workers

        # device
        self.config.device = args.device

        # microscope
        self.config.microscope.model = args.model
        self.config.microscope.phase_plate = args.phase_plate

        # microscope->beam
        self.config.microscope.beam.energy = args.energy
        self.config.microscope.beam.energy_spread = args.energy_spread
        self.config.microscope.beam.acceleration_voltage_spread = args.acceleration_voltage_spread
        self.config.microscope.beam.electrons_per_angstrom = args.electrons_per_angstrom
        # (newer defn) self.config.microscope.beam.total_electrons_per_angstrom = args.electrons_per_angstrom
        # self.config.microscope.beam.illumination_semiangle = args.illumination_semiangle
        self.config.microscope.beam.phi = args.phi
        self.config.microscope.beam.theta = args.theta

        # microscope->detector
        self.config.microscope.detector.nx = args.nx
        self.config.microscope.detector.ny = args.ny
        self.config.microscope.detector.pixel_size = args.pixel_size
        self.config.microscope.detector.dqe = args.dqe
        self.config.microscope.detector.origin = (args.origin_x, args.origin_y)

        # microscope->lens
        self.config.microscope.lens.c_10 = args.c_10
        self.config.microscope.lens.c_c = args.c_c
        self.config.microscope.lens.c_12 = args.c_12
        self.config.microscope.lens.phi_12 = args.phi_12
        self.config.microscope.lens.c_21 = args.c_21
        self.config.microscope.lens.phi_21 = args.phi_21
        self.config.microscope.lens.c_23 = args.c_23
        self.config.microscope.lens.phi_23 = args.phi_23
        self.config.microscope.lens.c_30 = args.c_30
        self.config.microscope.lens.c_32 = args.c_32
        self.config.microscope.lens.phi_32 = args.phi_32
        self.config.microscope.lens.c_34 = args.c_34
        self.config.microscope.lens.phi_34 = args.phi_34
        self.config.microscope.lens.c_41 = args.c_41
        self.config.microscope.lens.phi_41 = args.phi_41
        self.config.microscope.lens.c_43 = args.c_43
        self.config.microscope.lens.phi_43 = args.phi_43
        self.config.microscope.lens.c_45 = args.c_45
        self.config.microscope.lens.phi_45 = args.phi_45
        self.config.microscope.lens.c_50 = args.c_50
        self.config.microscope.lens.c_52 = args.c_52
        self.config.microscope.lens.phi_52 = args.phi_52
        self.config.microscope.lens.c_54 = args.c_54
        self.config.microscope.lens.phi_54 = args.phi_54
        self.config.microscope.lens.c_56 = args.c_56
        self.config.microscope.lens.phi_56 = args.phi_56
        self.config.microscope.lens.current_spread = args.current_spread

        # sample
        self.config.sample.box = (args.box_x, args.box_y, args.box_z)
        self.config.sample.centre = (args.centre_x, args.centre_y, args.centre_z)
        
        # sample->ice
        self.config.sample.ice = config.Ice()
        self.config.sample.ice.generate = args.slow_ice
        self.config.sample.ice.density = args.slow_ice_density

        # sample->coords
        # self.config.sample.coords = 

        # sample->molecules
        self.config.sample.molecules = config.Molecules()
        self.config.sample.molecules.local = []

        # sample->shape
        self.config.sample.shape.type = args.type
        self.config.sample.shape.cube.length = args.cube_length
        self.config.sample.shape.cuboid.length_x = args.cuboid_length_x
        self.config.sample.shape.cuboid.length_y = args.cuboid_length_y
        self.config.sample.shape.cuboid.length_z = args.cuboid_length_z
        self.config.sample.shape.cylinder.length = args.cylinder_length
        self.config.sample.shape.cylinder.radius = args.cylinder_radius
        self.config.sample.shape.margin = (args.margin_x, args.margin_y, args.margin_z)

        # sample->sputter (not yet supported as user input)
        # self.config.sample.sputter.element = 
        # self.config.sample.sputter.thickness = 

        # scan (not yet supported as user input)
        self.config.scan.mode = "still"
        self.config.scan.axis = (0., 1., 0.)
        self.config.scan.start_angle = 0.
        self.config.scan.step_angle = 0.
        self.config.scan.start_pos = 0.
        self.config.scan.step_pos = "auto"
        self.config.scan.num_images = 1
        # self.config.scan.num_fractions = 1
        # self.config.scan.num_nhelix = 1
        self.config.scan.exposure_time = 1
        self.config.scan.angles = None
        self.config.scan.positions = None
        # self.config.scan.theta = None
        # self.config.scan.phi = None
        # self.config.scan.drift.magnitude = 
        # self.config.scan.drift.kernel_size = 

        # simulation
        self.config.simulation.slice_thickness = args.slice_thickness
        self.config.simulation.margin = args.simulation_margin
        self.config.simulation.padding = args.simulation_padding
        # self.config.simulation.division_thickness = args.
        self.config.simulation.ice = args.fast_ice
        self.config.simulation.radiation_damage_model = args.radiation_damage_model
        self.config.simulation.sensitivity_coefficient = args.sensitivity_coefficient
        self.config.simulation.inelastic_model = args.inelastic_model
        self.config.simulation.mp_loss_width = args.mp_loss_width
        self.config.simulation.mp_loss_position = args.mp_loss_position
        
    def _save_config(self):
        """Save the configuration file to disk
        """
        with open(self.config_filename, "w") as f:
            yaml.safe_dump(self.config.dict(), f)
        
    def _add_molecule(self, pdb_file: str, n: int=1, position: list=[], orientation: list=[]):        
        """
        Add a molecule to the configuration file
        
        pdb_file (str): Molecule filepath to add to configuration file
        n (int): number of instances of the conformation. Defaults to 1.
        position (list): Positions of the molecule in A. Defaults to [].
        orientation (list): Orientations of the molecule in degrees. Defaults to [].
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
        
        self.config.sample.molecules.local.append(
            config.LocalMolecule(filename=pdb_file, instances=instance)
        )

    def add_molecules(self, frames: str, instances: int):
        """Add molecules to the configuration file.

        Args:
            frames (str): Filepath of molecule to add to configuration file
            instances (int): Number of instances of molecule to add to configuration file 
        """
        for frame, instance in zip(frames, instances):
            # self._add_molecule(frame, n=instance, orientation=orientation_generator.generate_inplane())
            self._add_molecule(frame, n=instance)
            
        self._save_config()

    def update_config(self, sample):
        """
        Updates the configuration file with the generated positions and orientations
        
        sample (parakeet.sample.Sample): Sample object from Parakeet
        """
        frame_idx = 0
        for molecule in sample.molecules:
            _, positions, orientations = sample.get_molecule(molecule)

            self.config.sample.molecules.local[frame_idx].instances = []

            for position, orientation in zip(positions, orientations):
<<<<<<< HEAD
                self.config.sample.molecules.local[frame_idx].instances.append(config.MoleculePose())
                self.config.sample.molecules.local[frame_idx].instances[-1].position = [float(p) for p in position]
                self.config.sample.molecules.local[frame_idx].instances[-1].orientation = [float(o) for o in orientation]
=======
                self.Config.sample.molecules.local[frame_idx].instances.append(config.MoleculePose())
                self.Config.sample.molecules.local[frame_idx].instances[-1].position = [float(p) for p in position]
                self.Config.sample.molecules.local[frame_idx].instances[-1].orientation = [float(o) for o in orientation]
                
>>>>>>> development
                
            frame_idx += 1
        self._save_config()
