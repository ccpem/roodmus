"""Simulation of micrograph/tomogram dataset using Parakeet software

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

import pandas as pd
import numpy as np
from gemmi import cif
from tqdm import tqdm


def add_arguments(
    write_starfile_parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Set up arguments for parsing.

    Args:
        parser (argparse.ArgumentParser):
        argparse.ArgumentParser:  write_starfile argument parser

    Returns:
        argparse.ArgumentParser:  write_starfile argument parser with arguments
    """

    write_starfile_parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Input csv file with particle coordinates",
    )

    write_starfile_parser.add_argument(
        "--type",
        type=str,
        default="coordinate_star",
        help="Type of starfile to write. By default generates a coordinate \
        starfile",
        choices=["coordinate_star", "data_star"],
    )

    write_starfile_parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for starfiles",
    )

    write_starfile_parser.add_argument(
        "--ugraph_dir",
        type=str,
        required=True,
        help="Directory with micrographs. \
            This is where RELION imported them from in case of single-frame \
            micrographs, or the directory in MotionCorr/jobxxx/micrographs/ \
            in case of movie stacks that have been motioncorrected",
    )

    write_starfile_parser.add_argument(
        "--pixel_size",
        type=float,
        required=True,
        help="Pixel size of micrographs (in Angstroms)",
    )

    write_starfile_parser.add_argument(
        "--tqdm",
        action="store_true",
        help="Enable progressbar",
    )
    optics_args = write_starfile_parser.add_argument_group(
        "Optics group arguments"
    )
    optics_args.add_argument(
        "--optics_group_name",
        type=str,
        default="opticsGroup1",
        help="Name of the optics group",
    )
    optics_args.add_argument(
        "--optics_group",
        type=int,
        default=1,
        help="Optics group number",
    )
    optics_args.add_argument(
        "--mtf_filename",
        type=str,
        default="mtf_k3_300kV.star",
        help="Filename of the modulation transfer function",
    )
    optics_args.add_argument(
        "--micrograph_original_pixel_size",
        type=float,
        default=1.0,
        help="Pixel size of the micrograph",
    )
    optics_args.add_argument(
        "--voltage",
        type=float,
        default=300,
        help="Voltage of the microscope",
    )
    optics_args.add_argument(
        "--spherical_aberration",
        type=float,
        default=2.7,
        help="Spherical aberration of the microscope",
    )
    optics_args.add_argument(
        "--amplitude_contrast",
        type=float,
        default=0.1,
        help="Amplitude contrast of the microscope",
    )
    optics_args.add_argument(
        "--image_pixel_size",
        type=float,
        default=1.0,
        help="Pixel size of the extracted particles",
    )
    optics_args.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Size of the extracted particles",
    )
    optics_args.add_argument(
        "--image_dimensionality",
        type=int,
        default=2,
        help="Dimensionality of the extracted particles",
    )
    optics_args.add_argument(
        "--ctf_data_are_ctf_premultiplied",
        type=int,
        default=0,
        help="Whether the CTF data is pre-multiplied",
    )
    return write_starfile_parser


def get_name():
    return "write_starfile"


class particle_coords_star(object):
    """
    The structure of a coords starfile is as follows:
    individual starfile with coordinates of particles in a single micrograph:

    data_

    loop_
    _rlnCoordinateX #1
    _rlnCoordinateY #2
    _rlnAutopickFigureOfMerit #3
    _rlnClassNumber #4
    _rlnAnglePsi #5

    a general starfile pointing to the location of individual starfiles:

    data_coordinate_files

    loop_
    _rlnMicrographName #1
    _rlnMicrographCoordinates #2

    """

    def __init__(
        self,
        ugraph_dir: str,
        output_dir: str,
        pixel_size: float,
        enable_progressbar: bool = False,
    ):
        self.cif_document = cif.Document()
        self.tags = [
            "_rlnMicrographName",
            "_rlnMicrographCoordinates",
        ]
        self.coordinate_files = self.cif_document.add_new_block(
            "coordinate_files"
        )
        self.loop = self.coordinate_files.init_loop(prefix="", tags=self.tags)

        self.ugraph_dir = ugraph_dir
        self.output_dir = output_dir
        self.micrograph_star_output_dir = os.path.join(
            self.output_dir, "micrographs"
        )
        self.pixel_size = pixel_size
        self.enable_progressbar = enable_progressbar

    def parse_df(self, df_particles: pd.DataFrame):
        """
        Parse a dataframe of particles into a starfile.
        The dataframe must contain:
        position_x, position_y, ugraph_filename

        An individual starfile with coordinates for each micrograph
        is created in the output_dir, as well as a collective starfile
        that points to the individual starfiles.
        """

        df_particles_grouped = df_particles.groupby("ugraph_filename")
        progressbar = tqdm(
            total=len(df_particles_grouped),
            desc="Writing starfiles",
            disable=not self.enable_progressbar,
        )
        self.individual_starfiles = {}

        # each micrograph gets its own starfile with coordinates of al
        # particles. These individual starfiles are stored in a dictionary
        # to be saved later. Each micrograph also gets a row in the master
        # starfile that points to the corresponding individual starfiles

        for group in df_particles_grouped.groups:
            ugraph_filename = os.path.join(self.ugraph_dir, str(group))
            df = df_particles_grouped.get_group(group)
            starfile = _individual_micrograph_coords(self.pixel_size)
            starfile.parse_df(df)
            starfile_name = os.path.join(
                self.micrograph_star_output_dir,
                str(group).replace(".mrc", "_groundtruth.star"),
            )
            self.individual_starfiles[starfile_name] = starfile

            self.loop.add_row(
                [
                    ugraph_filename,
                    starfile_name,
                ]
            )

            progressbar.update(1)
        progressbar.close()

    def save_stafile(self):
        """
        Save the starfile to the output directory
        """

        # first make the output directory and a subdirectory to hold the
        # individual starfiles. Then loop over the generated starfiles and
        # save them before saving the master starfile

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        if not os.path.exists(self.micrograph_star_output_dir):
            os.makedirs(self.micrograph_star_output_dir)

        for starfile_name, starfile in self.individual_starfiles.items():
            starfile.cif_document.write_file(starfile_name)

        starfile_name = os.path.join(
            self.output_dir,
            "groundtruth.star",
        )
        self.cif_document.write_file(starfile_name)


class _individual_micrograph_coords(object):
    def __init__(self, pixel_size):
        self.cif_document = cif.Document()
        self.tags = [
            "_rlnCoordinateX",
            "_rlnCoordinateY",
            "_rlnAutopickFigureOfMerit",
            "_rlnClassNumber",
            "_rlnAnglePsi",
        ]
        self.particles = self.cif_document.add_new_block("")
        self.loop = self.particles.init_loop(prefix="", tags=self.tags)

        self.pixel_size = pixel_size

    def parse_df(self, df_particles: pd.DataFrame):
        """
        This function converts the provided dataframe into a starfile.
        The dataframe must contain rows with the following columns:
        position_x, position_y, ugraph_filename

        Args:
            df_particles (pd.DataFrame): dataframe with particle coordinates
            pixel_size (float): pixel size of micrographs (in Angstroms)
        """
        for _, row in df_particles.iterrows():
            self.loop.add_row(
                [
                    str(row["position_x"] * self.pixel_size),
                    str(row["position_y"] * self.pixel_size),
                    "1",  # figure of merit not used
                    "0",  # class number not used
                    "0",  # angle psi not used
                ]
            )


class particle_data_star(object):
    """
    The structure of a (fully filled out) particle data stafile is as follows:
    data_particles

    loop_
    _rlnCoordinateX #1
    _rlnCoordinateY #2
    _rlnAutopickFigureOfMerit #3
    _rlnClassNumber #4 (representing 2D or 3D classes)
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
    _rlnGroupNumber #17
    _rlnAngleRot #18
    _rlnAngleTilt #19
    _rlnOriginXAngst #20
    _rlnOriginYAngst #21
    _rlnNormCorrection #22
    _rlnLogLikeliContribution #23
    _rlnMaxValueProbDistribution #24
    _rlnNrOfSignificantSamples #25
    _rlnRandomSubset #26
    """

    def __init__(
        self, ugraph_dir, output_dir, pixel_size, enable_progressbar=False
    ):
        self.cif_document = cif.Document()
        self.tags = [
            "_rlnCoordinateX",
            "_rlnCoordinateY",
            "_rlnAutopickFigureOfMerit",
            "_rlnClassNumber",
            "_rlnAnglePsi",
            "_rlnImageName",
            "_rlnMicrographName",
            "_rlnOpticsGroup",
            "_rlnCtfMaxResolution",
            "_rlnCtfFigureOfMerit",
            "_rlnDefocusU",
            "_rlnDefocusV",
            "_rlnDefocusAngle",
            "_rlnCtfBfactor",
            "_rlnCtfScalefactor",
            "_rlnPhaseShift",
            "_rlnGroupNumber",
            "_rlnAngleRot",
            "_rlnAngleTilt",
            "_rlnOriginXAngst",
            "_rlnOriginYAngst",
            "_rlnNormCorrection",
            "_rlnLogLikeliContribution",
            "_rlnMaxValueProbDistribution",
            "_rlnNrOfSignificantSamples",
            "_rlnRandomSubset",
        ]
        self.particles = self.cif_document.add_new_block("particles")
        self.loop = self.particles.init_loop(prefix="", tags=self.tags)

        self.ugraph_dir = ugraph_dir
        self.output_dir = output_dir
        self.pixel_size = pixel_size
        self.enable_progressbar = enable_progressbar

    def parse_df(self, df_particles):
        """
        Parse a dataframe of particles into a starfile.
        The dataframe must contain:
        position_x, position_y, ugraph_filename

        The dataframe can also optionally contain:
        euler_phi, euler_psi, euler_theta, defocusU, defocusV, Class2D
        """

        progressbar = tqdm(
            total=len(df_particles),
            desc="Writing starfiles",
            disable=not self.enable_progressbar,
        )
        for i, row in df_particles.iterrows():
            micrograph_filename = os.path.join(
                self.ugraph_dir, row["ugraph_filename"]
            )
            self.loop.add_row(
                [
                    str(row["position_x"] * self.pixel_size),
                    str(row["position_y"] * self.pixel_size),
                    "1",  # figure of merit not used
                    str(row.get("Class2D", "0")),
                    str(np.rad2deg(row.get("euler_psi", "0"))),
                    "image_name",
                    micrograph_filename,
                    "1",  # default optics group is 1
                    "0",  # CTF max resolution not used
                    "0",  # CTF figure of merit not used
                    str(row.get("defocusU", "0")),
                    str(row.get("defocusV", "0")),
                    "0",  # defocus angle not used
                    "0",  # CTF Bfactor not used
                    "1",  # CTF scalefactor defaults to 1
                    "0",  # phase shift not used
                    "1",  # group number not used
                    str(np.rad2deg(row.get("euler_phi", "0"))),
                    str(np.rad2deg(row.get("euler_theta", "0"))),
                    "0",  # origin x not used
                    "0",  # origin y not used
                    "0",  # norm correction not used
                    "0",  # log likelihood contribution not used
                    "0",  # max value prob distribution not used
                    "1",  # nr of significant samples not used
                    "1",  # random subset not used
                ]
            )
            progressbar.update(1)
        progressbar.close()

    def add_optics_group(
        self,
        optics_group_name: str = "opticsGroup1",
        optics_group: int = 1,
        mtf_filename: str = "mtf_k3_300kV.star",
        micrograph_original_pixel_size: float = 1.0,
        voltage: float = 300,
        spherical_aberration: float = 2.7,
        amplitude_contrast: float = 0.1,
        image_pixel_size: float = 1.0,
        image_size: int = 256,
        image_dimensionality: int = 2,
        ctf_data_are_ctf_premultiplied: int = 0,
    ):
        """
        Add an optics group to the starfile.
        """
        self.optics = optics_star()
        self.optics.add_optics_group(
            optics_group_name,
            optics_group,
            mtf_filename,
            micrograph_original_pixel_size,
            voltage,
            spherical_aberration,
            amplitude_contrast,
            image_pixel_size,
            image_size,
            image_dimensionality,
            ctf_data_are_ctf_premultiplied,
        )
        self.cif_document.add_copied_block(
            self.optics.get_optics_block(), pos=0
        )

    def save_stafile(self):
        """
        Save the starfile to the output directory
        """

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        starfile_name = os.path.join(
            self.output_dir,
            "particles.star",
        )
        self.cif_document.write_file(starfile_name)


class optics_star(object):
    """
    The structure of an optics starfile is as follows:
    data_optics

    loop_
    _rlnOpticsGroupName #1
    _rlnOpticsGroup #2
    _rlnMtfFileName #3
    _rlnMicrographOriginalPixelSize #4
    _rlnVoltage #5
    _rlnSphericalAberration #6
    _rlnAmplitudeContrast #7
    _rlnImagePixelSize #8
    _rlnImageSize #9
    _rlnImageDimensionality #10
    _rlnCtfDataAreCtfPremultiplied #11
    """

    def __init__(self):
        self.block = cif.Block("optics")
        self.tags = [
            "_rlnOpticsGroupName",
            "_rlnOpticsGroup",
            "_rlnMtfFileName",
            "_rlnMicrographOriginalPixelSize",
            "_rlnVoltage",
            "_rlnSphericalAberration",
            "_rlnAmplitudeContrast",
            "_rlnImagePixelSize",
            "_rlnImageSize",
            "_rlnImageDimensionality",
            "_rlnCtfDataAreCtfPremultiplied",
        ]
        self.loop = self.block.init_loop(prefix="", tags=self.tags)

    def add_optics_group(
        self,
        optics_group_name: str = "opticsGroup1",
        optics_group: int = 1,
        mtf_filename: str = "mtf_k3_300kV.star",
        micrograph_original_pixel_size: float = 1.0,
        voltage: float = 300,
        spherical_aberration: float = 2.7,
        amplitude_contrast: float = 0.1,
        image_pixel_size: float = 1.0,
        image_size: int = 256,
        image_dimensionality: int = 2,
        ctf_data_are_ctf_premultiplied: int = 0,
    ):
        """
        Add an optics group to the starfile.

        Args:
            optics_group_name (str):
                name of the optics group
            optics_group (int):
                optics group number
            mtf_filename (str):
                filename of the modulation transfer function
            micrograph_original_pixel_size (float):
                pixel size of the micrograph
            voltage (float):
                voltage of the microscope
            spherical_aberration (float):
                spherical aberration of the microscope
            amplitude_contrast (float):
                amplitude contrast of the microscope
            image_pixel_size (float):
                pixel size of the image
            image_size (int):
                size of the image
            image_dimensionality (int):
                dimensionality of the image
            ctf_data_are_ctf_premultiplied (int):
                whether the CTF data is pre-multiplied
        """

        # all values must be converted to strings
        # before adding them to the starfile
        self.loop.add_row(
            [
                optics_group_name,
                f"{optics_group:d}",
                mtf_filename,
                f"{micrograph_original_pixel_size:.2f}",
                f"{voltage:.2f}",
                f"{spherical_aberration:.2f}",
                f"{amplitude_contrast:.2f}",
                f"{image_pixel_size:.3f}",
                f"{image_size:d}",
                f"{image_dimensionality:d}",
                f"{ctf_data_are_ctf_premultiplied:d}",
            ]
        )

    def get_optics_block(self):
        return self.block


def main(args):
    """
    the starfile writer creates a set of starfiles that can be imported into
    relion. For each micrograph, a starfile with particle coordinates
    is created. A master starfile that mimics the result of autopicking
    is also created. This file should be loaded into relion and can then
    be used for particle extraction and further processing.
    """

    df_particles = pd.read_csv(args.input_csv)

    if args.type == "coordinate_star":
        starfile = particle_coords_star(
            args.ugraph_dir,
            args.output_dir,
            args.pixel_size,
            args.tqdm,
        )
        starfile.parse_df(df_particles)
        starfile.save_stafile()

    elif args.type == "data_star":
        starfile = particle_data_star(
            args.ugraph_dir,
            args.output_dir,
            args.pixel_size,
            args.tqdm,
        )
        starfile.parse_df(df_particles)
        starfile.add_optics_group(
            args.optics_group_name,
            args.optics_group,
            args.mtf_filename,
            args.micrograph_original_pixel_size,
            args.voltage,
            args.spherical_aberration,
            args.amplitude_contrast,
            args.image_pixel_size,
            args.image_size,
            args.image_dimensionality,
            args.ctf_data_are_ctf_premultiplied,
        )
        starfile.save_stafile()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_arguments(parser).parse_args())
