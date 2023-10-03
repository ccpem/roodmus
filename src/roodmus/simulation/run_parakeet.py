"""Simulation of micrograph/tomogram dataset using Parakeet software

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
import argparse
from typing import Tuple, List

import numpy as np
from tqdm import tqdm

import parakeet
from .configuration import Configuration


def add_arguments(
    run_parakeet_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Set up arguments for parsing, including both those for whole dataset
     to-be-generated and those for the configuration for each image in dataset

    Args:
        parser (argparse.ArgumentParser):
        argparse.ArgumentParser:  Running parakeet argument parser

    Returns:
        argparse.ArgumentParser:  Running parakeet argument parser
        with arguments
    """

    run_parakeet_parser.add_argument(
        "--pdb_dir",
        help="Path to the directory containing the pdb files",
        type=str,
        required=True,
    )

    run_parakeet_parser.add_argument(
        "--mrc_dir",
        help="Path to the directory in which to save the mrc files",
        type=str,
        required=True,
    )

    run_parakeet_parser.add_argument(
        "-n",
        "--n_images",
        help="Number of images to generate",
        type=int,
        default=1,
        required=False,
    )

    run_parakeet_parser.add_argument(
        "-m",
        "--n_molecules",
        help="Number of molecules to generate in each image",
        type=int,
        default=1,
        required=False,
    )

    run_parakeet_parser.add_argument(
        "--no_replacement",
        help="Disable sampling with replacement",
        default=True,
        action="store_false",
    )

    run_parakeet_parser.add_argument(
        "--tqdm",
        help="Turn on progress bar",
        default=False,
        action="store_true",
    )

    run_parakeet_parser.add_argument(
        "--overwrite_metadata",
        help="Overwrite metdata output",
        default=False,
        action="store_true",
    )
    run_parakeet_parser.add_argument(
        "--orientations",
        help="how to generate the orientations \
            for each particle. Default is random \
            through Parakeet",
        type=lambda s: parse_orientation(s),
        default="parakeet",
    )
    run_parakeet_parser.add_argument(
        "--verbose",
        help="Turn on verbose output",
        default=False,
        action="store_true",
    )

    parser = run_parakeet_parser.add_argument_group("configuration")
    parser.add_argument(
        "--max_workers",
        help=(
            "Maximum number of worker processes to use on a cluster."
            " Must specify cluster method."
        ),
        type=int,
        default=1,
        required=False,
    )

    parser.add_argument(
        "--method",
        help=(
            "Maximum number of worker processes to use on a cluster."
            " Must specify cluster method."
        ),
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--device",
        help=(
            "An enumeration to set whether to run on the GPU or CPU."
            " Options are 'cpu' or 'gpu'. Defaults to gpu"
        ),
        type=str,
        default="gpu",
        required=False,
    )

    parser.add_argument(
        "--acceleration_voltage_spread",
        help=("The acceleration voltage spread (dV/V). Defaults to 8.0e-07"),
        type=float,
        default=8.0e-7,
        required=False,
    )

    parser.add_argument(
        "--electrons_per_angstrom",
        help=(
            "Dose of electrons per square angstrom to use in"
            "Parakeet simulation. Defaults to 45.0"
        ),
        type=float,
        default=45.0,
        required=False,
    )

    parser.add_argument(
        "--energy",
        help=(
            "Electron beam energy (kV) to use in Parakeet simulation"
            " Defaults to 300.0kV"
        ),
        type=float,
        default=300.0,
        required=False,
    )

    parser.add_argument(
        "--energy_spread",
        help=("Electron beam energy spread (dE/E)" " Defaults to 2.66e-6"),
        type=float,
        default=2.66e-6,
        required=False,
    )

    parser.add_argument(
        "--source_spread",
        help=("The source spread (mrad)." " Defaults to 0.1"),
        type=float,
        default=0.1,
        required=False,
    )

    """
    parser.add_argument(
        "--illumination_semiangle",
        help=("The illumination semiangle (mrad)."),
        type=float,
        default=0.02,
        required=False,
    )
    """

    parser.add_argument(
        "--phi",
        help=("The beam tilt phi angle (deg). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--theta",
        help=("The beam tilt theta angle (deg). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--dqe",
        help=("Use the DQE model (True/False). Defaults to False"),
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--nx",
        help=(
            "Number of pixels along the x"
            " axis of the image(s) to be"
            " simulated using Parakeet"
        ),
        type=int,
        default=1000,
        required=False,
    )

    parser.add_argument(
        "--ny",
        help=(
            "Number of pixels along the y"
            " axis of the image(s) to be"
            " simulated using Parakeet"
        ),
        type=int,
        default=1000,
        required=False,
    )

    parser.add_argument(
        "--origin_x",
        help=(
            "Origin of detector along the x"
            " axis in lab fram (A)"
            " simulated using Parakeet"
        ),
        type=int,
        default=0,
        required=False,
    )

    parser.add_argument(
        "--origin_y",
        help=(
            "Origin of detector along the y"
            " axis in lab fram (A)"
            " simulated using Parakeet"
        ),
        type=int,
        default=0,
        required=False,
    )

    parser.add_argument(
        "--pixel_size",
        help=(
            "Pixel size (angstroms) to use"
            " in Parakeet simulation."
            " All pixels are square."
            " Defaults to 1.0"
        ),
        type=float,
        default=1.0,
        required=False,
    )

    parser.add_argument(
        "--phase_plate",
        help=(
            "Whether to use a"
            " phase plate in"
            " Parakeet simulation"
            " Defaults to False"
        ),
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--c_10",
        help=(
            "Average defocus value (A) (input a negative number for"
            " underfocus) to use in Parakeet simulation."
        ),
        type=float,
        default=[-20000],
        required=False,
        nargs="+",
    )

    parser.add_argument(
        "--c_10_stddev",
        help=(
            "Standard deviation of defocus value (A)"
            " to use in Parakeet simulation."
        ),
        type=float,
        default=[5000],
        required=False,
        nargs="+",
    )

    parser.add_argument(
        "--c_12",
        help=("The 2-fold astigmatism (A). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--phi_12",
        help=(
            "The Azimuthal angle of 2-fold astigmatism (rad)." " Defaults to 0"
        ),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--c_21",
        help=("The Axial coma (A). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--phi_21",
        help=("The Azimuthal angle of axial coma (rad). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--c_23",
        help=("The 3-fold astigmatism (A). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--phi_23",
        help=(
            "The Azimuthal angle of 3-fold astigmatism (rad)." " Defaults to 0"
        ),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--c_30",
        help=("Spherical aberration (A). Defaults to 2.7"),
        type=float,
        default=2.7,
        required=False,
    )

    parser.add_argument(
        "--c_32",
        help=("The Axial star aberration (A). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--phi_32",
        help=(
            "The Azimuthal angle of axial star aberration (rad)."
            " Defaults to 0"
        ),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--c_34",
        help=("The 4-fold astigmatism (A). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--phi_34",
        help=(
            "The Azimuthal angle of 4-fold astigmatism (rad)" " Defaults to 0"
        ),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--c_41",
        help=("The 4th order axial coma (A). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--phi_41",
        help=(
            "The Azimuthal angle of 4th order axial coma (rad)."
            " Defaults to 0"
        ),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--c_43",
        help=("The 3-lobe aberration (A). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--phi_43",
        help=("The Azimuthal angle of 3-lobe aberration (rad). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--c_45",
        help=("The 5-fold astigmatism (A).  Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--phi_45",
        help=(
            "The Azimuthal angle of 5-fold astigmatism (rad)." " Defaults to 0"
        ),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--c_50",
        help=("The 5th order spherical aberration (A). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--c_52",
        help=("The 5th order axial star aberration (A). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--phi_52",
        help=(
            "The Azimuthal angle of 5th order axial star aberration (rad)."
            " Defaults to 0"
        ),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--c_54",
        help=("The 5th order rosette aberration (A). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--phi_54",
        help=(
            "The Azimuthal angle of 5th order rosette aberration (rad)."
            " Defaults to 0"
        ),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--c_56",
        help=("The 6-fold astigmatism (A). Defaults to 0"),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--phi_56",
        help=(
            "The Azimuthal angle of 6-fold astigmatism (rad)." " Defaults to 0"
        ),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--c_c",
        help=("Chromatic aberration (A). Defaults to 2.7"),
        type=float,
        default=2.7,
        required=False,
    )

    parser.add_argument(
        "--current_spread",
        help=("The current spread (dI/I). Defaults to 0.33e-6"),
        type=float,
        default=0.33e-6,
        required=False,
    )

    parser.add_argument(
        "--model",
        help=(
            "Use commercial microscope ('talos' or 'krios')"
            " instead of user-provided microscope values."
            " Defaults to None (user-provided values)"
        ),
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--box_x",
        help=("Sample box size along x axis (Angstroms). Defaults to 1000"),
        type=float,
        default=1000.0,
        required=False,
    )

    parser.add_argument(
        "--box_y",
        help=("Sample box size along y axis (Angstroms). Defaults to 1000"),
        type=float,
        default=1000.0,
        required=False,
    )

    parser.add_argument(
        "--box_z",
        help=("Sample box size along z axis (Angstroms). Defaults to 500"),
        type=float,
        default=500.0,
        required=False,
    )

    parser.add_argument(
        "--centre_x",
        help=(
            "Center of tomographic rotation around sample x axis (A)."
            " Defaults to 500"
        ),
        type=float,
        default=500.0,
        required=False,
    )

    parser.add_argument(
        "--centre_y",
        help=(
            "Center of tomographic rotation around sample y axis (A)."
            " Defaults to 500"
        ),
        type=float,
        default=500.0,
        required=False,
    )

    parser.add_argument(
        "--centre_z",
        help=(
            "Center of tomographic rotation around sample z axis (A)."
            " Defaults to 250"
        ),
        type=float,
        default=250.0,
        required=False,
    )

    parser.add_argument(
        "--slow_ice",
        help=(
            "Generate the (very slow) atomic ice model?"
            " Defaults to None (not used)"
        ),
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--slow_ice_density",
        help=("Density of molecular ice (slow ice simulation) in kg/m3"),
        type=float,
        default=940.0,
        required=False,
    )

    parser.add_argument(
        "--type",
        help=(
            "An enumeration of sample shape types."
            " Options are 'cube', 'cuboid' or 'cylinder'."
            " Defaults to cuboid"
        ),
        type=str,
        default="cuboid",
        required=False,
    )

    parser.add_argument(
        "--cube_length",
        help=("The cube side length (A). Defaults to 1000"),
        type=float,
        default=1000.0,
        required=False,
    )

    parser.add_argument(
        "--cuboid_length_x",
        help=("The cuboid X side length (A). Defaults to 1000"),
        type=float,
        default=1000.0,
        required=False,
    )

    parser.add_argument(
        "--cuboid_length_y",
        help=("The cuboid Y side length (A). Defaults to 1000"),
        type=float,
        default=1000.0,
        required=False,
    )

    parser.add_argument(
        "--cuboid_length_z",
        help=("The cuboid Z side length (A). Defaults to 1000"),
        type=float,
        default=500.0,
        required=False,
    )

    parser.add_argument(
        "--cylinder_length",
        help=("The cylinder length (A). Defaults to 1000"),
        type=float,
        default=1000.0,
        required=False,
    )

    parser.add_argument(
        "--cylinder_radius",
        help=("The cylinder radius (A). Defaults to 500"),
        type=float,
        default=500.0,
        required=False,
    )

    parser.add_argument(
        "--margin_x",
        help=(
            "The x axis margin used to define how close to the edges particles"
            " should be placed (A)."
            " Default value is 0"
        ),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--margin_y",
        help=(
            "The y axis margin used to define how close to the edges particles"
            " should be placed (A)."
            " Default value is 0"
        ),
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--margin_z",
        help=(
            "The z axis margin used to define how close to the edges particles"
            " should be placed (A)."
            " Default value is 0"
        ),
        type=float,
        default=0.0,
        required=False,
    )

    # scan args
    parser.add_argument(
        "--scan_mode",
        help="Scan mode, defaults to 'still'.",
        default="still",
        choices=[
            "still",
            "manual",
            "tilt_series",
            "dose_symmetric",
            "single_particle",
            "helical_scan",
            "nhelix",
            "beam_tilt",
        ],
        required=False,
    )

    parser.add_argument(
        "--scan_axis",
        help=("Scan axis vector, provide x y z. Defaults to 0. 1. 0."),
        type=float,
        default=[0.0, 1.0, 0.0],
        nargs=3,
        required=False,
    )

    parser.add_argument(
        "--scan_start_angle",
        help="The start angle for the rotation (deg)",
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--scan_step_angle",
        help="The step angle for the rotation (deg)",
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--scan_start_pos",
        help="The start position for a translational scan (A)",
        type=float,
        default=0.0,
        required=False,
    )

    parser.add_argument(
        "--scan_step_pos",
        help=(
            "The step distance for a translational scan (A)"
            "Defaults to 'auto'"
        ),
        type=float,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--scan_num_images",
        help=(
            "The number of images to simulate. "
            "For a tilt series this is the number of tilt steps. "
            "If num_fractions is also set to something other than 1, "
            "then there will be num_fractions number of 'movie frames' "
            "per 'image'"
        ),
        type=int,
        default=1,
        required=False,
    )

    parser.add_argument(
        "--scan_num_fractions",
        help=(
            "The number of movie frames. This refers to the frames of the "
            "micrograph 'movies'. For a tilt series, all these images will "
            "be at the same step and the dose for a 'single image' will "
            "be fractionated over these image frames"
        ),
        type=int,
        default=1,
        required=False,
    )

    parser.add_argument(
        "--scan_num_nhelix",
        help="The number of scans in an n-helix",
        type=int,
        default=1,
        required=False,
    )

    parser.add_argument(
        "--exposure_time",
        help="The exposure time per image (s)",
        type=float,
        default=1.0,
        required=False,
    )

    parser.add_argument(
        "--scan_angles",
        help=(
            "The list of angles to use (deg). "
            "This field is used when the mode "
            "is set to 'manual' or 'beam tilt'."
        ),
        type=float,
        default=None,
        nargs="+",
        required=False,
    )

    parser.add_argument(
        "--scan_positions",
        help=(
            "The list of positions to use (A). "
            "This field is used when the mode"
            "is set to 'manual' or 'beam tilt'."
        ),
        type=float,
        default=None,
        nargs="+",
        required=False,
    )

    parser.add_argument(
        "--scan_theta",
        help=(
            "The list of theta angles to use (mrad) for the beam tilt."
            "This must either be the same length as phi or a scalar."
        ),
        type=float,
        default=None,
        nargs="+",
        required=False,
    )

    parser.add_argument(
        "--scan_phi",
        help=(
            "The list of phi angles to use (mrad) for the beam tilt."
            "This must either be the same length as theta or a scalar"
        ),
        type=float,
        default=None,
        nargs="+",
        required=False,
    )

    parser.add_argument(
        "--scan_drift",
        help=(
            "The model for the drift along each axis computed via "
            "a + b*theta**4 (A). Provide arguments following: "
            "x_a x_b y_a y_b z_a z_b, where x_a is the a component of "
            "the drift along the x axis."
        ),
        type=float,
        default=None,
        nargs=6,
        required=False,
    )

    parser.add_argument(
        "--fast_ice",
        help=(
            "Use the Gaussian Random Field ice model (True/False)."
            " Defaults to False"
        ),
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--simulation_margin",
        help=("The margin around the image. Defaults to 100"),
        type=int,
        default=100,
        required=False,
    )

    parser.add_argument(
        "--inelastic_model",
        help=("The inelastic model parameters. Defaults to None"),
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--mp_loss_position",
        help=("The MPL energy filter position. Defaults to peak"),
        type=str,
        default="peak",
        required=False,
    )

    parser.add_argument(
        "--mp_loss_width",
        help=("The MPL energy filter width (eV). Defaults to 50"),
        type=float,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--simulation_padding",
        help=("Additional padding. Defaults to 100"),
        type=int,
        default=100,
        required=False,
    )

    parser.add_argument(
        "--radiation_damage_model",
        help=("Use the radiation damage model? Defaults to False"),
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--sensitivity_coefficient",
        help=(
            "The radiation damage model sensitivity coefficient."
            " Defaults to 0.022"
        ),
        type=float,
        default=0.022,
        required=False,
    )

    parser.add_argument(
        "--slice_thickness",
        help=("The multislice thickness (A). Defaults to 3.0"),
        type=float,
        default=3.0,
        required=False,
    )

    parser.add_argument(
        "--leading_zeros",
        help=(
            "Number of decimal integers to use for image filenames."
            " Defaults to 6"
        ),
        type=int,
        default=6,
        required=False,
    )

    return run_parakeet_parser


def get_name():
    return "run_parakeet"


def parse_orientation(s):
    if s.lower() in ["parakeet", "inplane"]:
        return s.lower()
    elif "discrete_tilt" in s:
        return s.lower()
    else:
        raise ValueError(
            f"Orientation must be 'parakeet', 'inplane' \
                or 'discrete_tilt_*'."
            f" Got {s}"
        )


def sample_defocus(c_10: float, c_10_stddev: float) -> float:
    """Generate a defocus value from a Gaussian distribution

    Args:
        c_10 (float): Mean defocus value
        c_10_stddev (float): Std deviation to use

    Returns:
        float: Defocus value
    """
    return np.random.normal(c_10, c_10_stddev)


def get_pdb_files(pdb_dir: str) -> List[str]:
    """Grab a list of molecule/structure definition files (such as PDBs) to add
    to micrographs

    Args:
        pdb_dir (str): Path to directory containing all molecules to use

    Returns:
        list[str]: List of molecules file paths
    """
    pdb_dir = os.path.abspath(pdb_dir)
    pdb_files = []
    for file in os.listdir(pdb_dir):
        if file.endswith(".pdb"):
            pdb_files.append(os.path.join(pdb_dir, file))
        elif file.endswith(".cif"):
            pdb_files.append(os.path.join(pdb_dir, file))
        elif file.endswith(".mmcif"):
            pdb_files.append(os.path.join(pdb_dir, file))
        elif file.endswith(".cif.gz"):
            # untested as to whether parakeet can unpack .cif.gz
            pdb_files.append(os.path.join(pdb_dir, file))
    return pdb_files


def get_instances(
    pdb_files: List[str], n_molecules: int, replace=True
) -> Tuple[List[str], List[int]]:
    """Determine the molecules to simulate in a given image and the number of
    occurrences of each
     in the image

    Args:
        pdb_files list[str]: List of pdb files sample molecules for simulation
        from n_molecules int: Total number of molecules to simulate in each
        image replace (bool, optional): Toggle sampling with replacement on
        or off.
        Defaults to True.

    Returns:
        Tuple[list[str], list[int]]: Same length lists of molecules
         to simulate and # instances of the molecule
    """
    num_structures = len(pdb_files)

    # if the number of structures is <= than the number of molecules,
    # we need to repeat some of the structures
    if num_structures <= n_molecules:
        # start by adding each structure the same number of times to get as
        # close to the number of molecules as possible
        n_instances = [n_molecules // num_structures] * num_structures
        # then randomly add another copy of any pdb until we
        # reach the number of molecules
        for n in range(n_molecules - sum(n_instances)):
            n_instances[np.random.randint(num_structures)] += 1
        new_pdb_files = pdb_files
        new_n_instances = n_instances

        return pdb_files, n_instances

    # if the number of structures is greater than or equal to the number of
    # molecules, randomly sample them
    else:
        # sample the pdb files with replacement by default
        pdb_files_temp, n_instances_temp = np.unique(
            np.random.choice(pdb_files, n_molecules, replace=replace),
            return_counts=True,
        )
        new_pdb_files = pdb_files_temp.tolist()
        new_n_instances = n_instances_temp.tolist()

        return new_pdb_files, new_n_instances


def main(args):
    """
    Loops over the number of images to generate. For each image,
    parakeet is configured and a number of .pdb files is selected.
    The number of instances gets determined based on the number of
    .pdb files. If there are less .pdb files than molecules to generate,
    some of the .pdb files are repeated. If there are more .pdb files than
    molecules to generate, some of the .pdb files are removed.
    The orientations can be sampled, or left to parakeet to generate.
    Then we run parakeet and save the mrc files. From the sample object we
    can pull the orientations and positions of the molecules.
    These, along with the optical parameters, are saved in a .yaml file.

    nomenclature
    frame: .pdb file containing the structure of a molecule from a single
    frame in the MD trajectory
    instance: multiplicity of a frame in the simulated image. If there are
    10 frames and 100 molecules, each frame is an instance 10 times
    molecule: a single particle in the the simulated image
    """

    # create mrc-dir for output images if it doesn't already exist
    if not os.path.exists(args.mrc_dir):
        os.makedirs(args.mrc_dir)

    # check how many images are already in the mrc_dir
    images_in_directory = len(
        [r for r in os.listdir(args.mrc_dir) if r.endswith(".mrc")]
    )

    # get the pdb files
    frames = get_pdb_files(args.pdb_dir)

    # loop over the number of images to generate
    progressbar = tqdm(
        range(args.n_images), desc="Generating images", disable=not args.tqdm
    )
    defocus_idx = 0
    for n_image in range(
        images_in_directory, args.n_images + images_in_directory
    ):
        # full path to where the current image will be saved
        mrc_filename = os.path.join(
            args.mrc_dir, f"{n_image}".zfill(args.leading_zeros) + ".mrc"
        )

        # full path to where the current configuration will be saved
        config_filename = os.path.join(
            args.mrc_dir, f"{n_image}".zfill(args.leading_zeros) + ".yaml"
        )

        # initialise the configuration
        config = Configuration(config_filename, args=args)

        # sample the defocus around the specified value
        config.config.microscope.lens.c_10 = sample_defocus(
            args.c_10[defocus_idx], args.c_10_stddev[defocus_idx]
        )
        defocus_idx = (defocus_idx + 1) % len(
            args.c_10
        )  # in case there are multiple defocus values, we loop over them

        # determine the number of instances we need of each structure
        chosen_frames, n_instances = get_instances(
            frames, args.n_molecules, args.no_replacement
        )
        # add the molecules to the configuration
        config.add_molecules(
            chosen_frames, n_instances, orientation_method=args.orientations
        )

        # run parakeet
        sample = parakeet.sample.new(
            config.config_filename, sample_file=config.sample_filename
        )
        sample = parakeet.sample.add_molecules(
            config.config_filename, sample_file=config.sample_filename
        )

        # Write out the metadata if this is the first image in this
        # run_parakeet session or if overwrite requested
        if n_image == images_in_directory:
            metadata_exporter = parakeet.metadata.RelionMetadataExporter(
                config.config, sample, args.mrc_dir
            )
            if not os.path.exists(
                os.path.join(
                    args.mrc_dir,
                    "relion/mtf_{}kV.star".format(
                        config.config.microscope.beam.energy
                    ),
                )
            ):
                metadata_exporter.write_mtf_file()
            else:
                if args.overwrite_metadata:
                    metadata_exporter.write_mtf_file()
                else:
                    raise FileExistsError(
                        "{} already exists!".format(
                            os.path.join(args.mrc_dir, "relion")
                        )
                    )

        parakeet.simulate.exit_wave(
            config.config_filename,
            config.sample_filename,
            exit_wave_file=config.exit_wave_filename,
        )
        parakeet.simulate.optics(
            config.config_filename,
            exit_wave_file=config.exit_wave_filename,
            optics_file=config.optics_filename,
        )
        parakeet.simulate.image(
            config.config_filename,
            optics_file=config.optics_filename,
            image_file=config.image_filename,
        )

        # save the image
        os.system(
            "parakeet.export {} -o {}".format(
                config.image_filename,
                mrc_filename,
            )
        )

        # remove the intermediate files
        os.system(
            "rm {} {} {} {}".format(
                config.sample_filename,
                config.exit_wave_filename,
                config.optics_filename,
                config.image_filename,
            )
        )

        # update the config from the sample and save/overwrite it
        config.update_config(sample)

        progressbar.update(1)

    progressbar.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_arguments(parser).parse_args())
