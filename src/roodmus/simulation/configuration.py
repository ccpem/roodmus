"""Roodmus:
Configutation class to setup the parameters for Parakeet.

Delft University of Technology (TU Delft) hereby disclaims
all copyright interest in the program “Roodmus” written by
the Author(s).
Copyright (C) 2023  Joel Greer(UKRI), Tom Burnley (UKRI),
Maarten Joosten (TU Delft), Arjen Jakobi (TU Delft)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""


import os
from typing import Any

import yaml

from parakeet import config
from roodmus.simulation.orientation_generator import orientation_generator

"""
from run_parakeet.orientation_generator import orientation_generator
"""


class Configuration(object):
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

        self.config = config.new(filename=self.config_filename, full=True)
        if args:
            self.sample_filename = "sample.h5"
            self.exit_wave_filename = os.path.join(
                args.mrc_dir, "exit_wave.h5"
            )
            self.optics_filename = os.path.join(args.mrc_dir, "optics.h5")
            self.image_filename = os.path.join(args.mrc_dir, "image.h5")
            self.leading_zeros = args.leading_zeros
            self.verbose = args.verbose
            self._set_config(args)

    def _set_config(self, args):
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
        self.config.microscope.beam.acceleration_voltage_spread = (
            args.acceleration_voltage_spread
        )
        self.config.microscope.beam.electrons_per_angstrom = (
            args.electrons_per_angstrom
        )
        """
        (newer defn)
        self.config.microscope.beam.total_electrons_per_angstrom = (
            args.electrons_per_angstrom
        )
        self.config.microscope.beam.illumination_semiangle = (
            args.illumination_semiangle
        )
        """
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
        self.config.sample.centre = (
            args.centre_x,
            args.centre_y,
            args.centre_z,
        )

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
        self.config.sample.shape.margin = (
            args.margin_x,
            args.margin_y,
            args.margin_z,
        )

        # sample->sputter (not yet supported as user input)
        # self.config.sample.sputter.element =
        # self.config.sample.sputter.thickness =

        # scan (now supported as user input)
        self.config.scan.mode = args.scan_mode
        assert len(args.scan_axis) == 3, "Provide 3 arguments to --scan_axis!"
        self.config.scan.axis = tuple(args.scan_axis)
        self.config.scan.start_angle = args.scan_start_angle
        self.config.scan.step_angle = args.scan_step_angle
        self.config.scan.start_pos = args.scan_start_pos
        if args.scan_step_pos is None:
            self.config.scan.step_pos = "auto"
        else:
            self.config.scan.step_pos = args.scan_step_pos
        self.config.scan.num_images = args.scan_num_images
        self.config.scan.num_fractions = args.scan_num_fractions
        self.config.scan.num_nhelix = args.scan_num_nhelix
        self.config.scan.exposure_time = args.exposure_time
        self.config.scan.angles = args.scan_angles
        self.config.scan.positions = args.scan_positions
        self.config.scan.theta = args.scan_theta
        self.config.scan.phi = args.scan_phi
        self.config.scan.drift = args.scan_drift
        # self.config.scan.drift.kernel_size =

        # simulation
        self.config.simulation.slice_thickness = args.slice_thickness
        self.config.simulation.margin = args.simulation_margin
        self.config.simulation.padding = args.simulation_padding
        # self.config.simulation.division_thickness = args.
        self.config.simulation.ice = args.fast_ice
        self.config.simulation.radiation_damage_model = (
            args.radiation_damage_model
        )
        self.config.simulation.sensitivity_coefficient = (
            args.sensitivity_coefficient
        )
        self.config.simulation.inelastic_model = args.inelastic_model
        self.config.simulation.mp_loss_width = args.mp_loss_width
        self.config.simulation.mp_loss_position = args.mp_loss_position

    def _save_config(self):
        """Save the configuration file to disk"""
        with open(self.config_filename, "w") as f:
            yaml.safe_dump(self.config.dict(), f)

    def _add_molecule(
        self,
        pdb_file: str,
        n: int | None,
        position: list = [],
        orientation: list = [],
    ):
        """
        Add a molecule to the configuration file

        pdb_file (str): Molecule filepath to add to configuration file
        n (int): number of instances of the conformation. Defaults to 1.
        position (list): Positions of the molecule in A. Defaults to [].
        orientation (list): Orientations of the molecule in degrees.
        Defaults to [].
        """

        # The molecule can be specified by only the number of instances,
        # in which case parakeet generates the orientation and position
        # or by the position and orientation, in which case
        # the number of instances is set to 1
        if not position and not orientation and not n:
            raise ValueError(
                "At least one of the following must be specified:"
                " n, position, orientation"
            )

        # if the number of instances is specified and the
        # position and orientation are not specified
        if not position and not orientation:
            self.config.sample.molecules.local.append(
                config.LocalMolecule(filename=pdb_file, instances=n)
            )
        else:
            instance: list[Any] = []
            # if the orientation and position are specified
            if position and orientation:
                for p, o in zip(position, orientation):
                    pose = config.MoleculePose()
                    pose.position = p
                    pose.orientation = o
                    instance.append(pose)

            # if the orientation is specified but not the position
            elif orientation and not position:
                for o in orientation:
                    pose = config.MoleculePose()
                    pose.orientation = o
                    instance.append(pose)

            # if the position is specified but not the orientation
            elif position and not orientation:
                for p in position:
                    pose = config.MoleculePose()
                    pose.position = p
                    instance.append(pose)

            self.config.sample.molecules.local.append(
                config.LocalMolecule(filename=pdb_file, instances=instance)
            )

    def add_molecules(
        self,
        frames: list[str],
        instances: list[int],
        orientation_method: str = "parakeet",
    ):
        """Add molecules to the configuration file.

        Args:
            frames (str): Filepath of molecule to add to configuration file
            instances (int): Number of instances of molecule to
            add to configuration file
            orientation_generator (str, optional): Orientation generator
        """
        for frame, instance in zip(frames, instances):
            if orientation_method == "parakeet":
                # default behaviour
                # let's parakeet generate the orientations
                self._add_molecule(frame, n=instance)

            elif orientation_method == "inplane":
                # pre-defines orientations
                # only generates in-plane rotations
                # with no tilt
                orientation = orientation_generator.generate_inplane(
                    n=instance
                )
                self._add_molecule(frame, n=None, orientation=orientation)

            elif "discrete_tilt" in orientation_method:
                # pre-defines orientations
                # samples elevation angles from a discrete set
                # samples azimuthal and in-plane rotations from a
                # continuous uniform distribution
                k = int(orientation_method.split("_")[-1])
                orientation = orientation_generator.generate_discrete_tilt(
                    n=instance, k=k, save_to_file=self.verbose
                )
                self._add_molecule(frame, n=None, orientation=orientation)

            else:
                raise ValueError(
                    f"Orientation generator {orientation_generator} \
                        not supported"
                )

        self._save_config()

    def update_config(self, sample):
        """
        Updates the configuration file with the
        generated positions and orientations

        sample (parakeet.sample.Sample): Sample object from Parakeet
        """
        frame_idx = 0
        for molecule in sample.molecules:
            _, positions, orientations = sample.get_molecule(molecule)

            self.config.sample.molecules.local[frame_idx].instances = []

            for position, orientation in zip(positions, orientations):
                self.config.sample.molecules.local[frame_idx].instances.append(
                    config.MoleculePose()
                )
                self.config.sample.molecules.local[frame_idx].instances[
                    -1
                ].position = [float(p) for p in position]
                self.config.sample.molecules.local[frame_idx].instances[
                    -1
                ].orientation = [float(o) for o in orientation]

            frame_idx += 1
        self._save_config()
