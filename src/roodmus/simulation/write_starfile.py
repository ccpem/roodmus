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
            This is where RELION imported them from",
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

    return write_starfile_parser


def get_name():
    return "write_starfile"


def create_relion_star(
    df_particles: pd.DataFrame,
    pixel_size: float,
) -> cif.Document:
    """
    This function creates the starfile as a cif Document object using gemmi.

    Args:
        df_particles (pd.DataFrame): dataframe with particle coordinates
        ugraph_dir (str): directory with micrographs
        master_starfile_name (str, optional): name of the master starfile.
        Defaults to "groundtruth.star".
        enable_progressbar (bool, optional): enable progressbar.
        Defaults to True.
        verbose (bool, optional): verbose output. Defaults to False.

    Returns:
        cif.Document: cif Document object with starfile
    """

    tags = [
        "_rlnCoordinateX",
        "_rlnCoordinateY",
        "_rlnAutopickFigureOfMerit",
        "_rlnClassNumber",
        "_rlnAnglePsi",
    ]
    out_star = cif.Document()
    particles = out_star.add_new_block("")
    loop = particles.init_loop(prefix="", tags=tags)
    for i, row in df_particles.iterrows():
        loop.add_row(
            [
                str(row["position_x"] * pixel_size),
                str(row["position_y"] * pixel_size),
                "1",  # figure of merit not used
                "0",  # class number not used
                "0",  # angle psi not used
            ]
        )
    return out_star


def create_autopicker_star(
    df_particles: pd.DataFrame,
    ugraph_dir: str,
    output_dir: str,
) -> cif.Document:
    """
    Creates a master starfile that mimics the result of autopicking.
    Points to the created starfiles to be used for particle extraction.

    Args:
        df_particles (pd.DataFrame): dataframe with particle coordinates
        ugraph_dir (str): directory with micrographs
        output_dir (str): directory where starfiles are written

    Returns:
        cif.Document: cif Document object with starfile
    """

    df_particles_grouped = df_particles.groupby("ugraph_filename")
    # write master starfile
    out_star = cif.Document()
    coordinate_files = out_star.add_new_block("coordinate_files")
    tags = [
        "_rlnMicrographName",
        "_rlnMicrographCoordinates",
    ]
    loop = coordinate_files.init_loop(prefix="", tags=tags)
    for group in df_particles_grouped["ugraph_filename"].unique():
        ugraph_filename = os.path.join(ugraph_dir, group)
        star_filename = os.path.join(
            output_dir, group.replace(".mrc", "_groundtruth.star")
        )
        loop.add_row(
            [
                ugraph_filename,
                star_filename,
            ]
        )
    return out_star


def main(args):
    """
    the starfile writer creates a set of starfiles that can be imported into
    relion. For each micrograph, a starfile with particle coordinates
    is created. A master starfile that mimics the result of autopicking
    is also created. This file should be loaded into relion and can then
    be used for particle extraction and further processing.
    """

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    micrograph_star_output_dir = os.path.join(output_dir, "micrographs")
    if not os.path.exists(micrograph_star_output_dir):
        os.makedirs(micrograph_star_output_dir)

    df_particles = pd.read_csv(args.input_csv)
    # first create a starfile document for each micrograph
    df_particles_grouped = df_particles.groupby("ugraph_filename")
    progressbar = tqdm(
        total=len(df_particles_grouped),
        desc="Writing starfiles",
        disable=not args.tqdm,
    )
    for group in df_particles_grouped.groups:
        df = df_particles_grouped.get_group(group)
        starfile = create_relion_star(df, args.pixel_size)
        starfile_name = os.path.join(
            micrograph_star_output_dir,
            group.replace(".mrc", "_groundtruth.star"),
        )
        starfile.write_file(starfile_name)
        progressbar.update(1)
    progressbar.close()

    # now create a master starfile that mimics the result of autopicking
    master_starfile = create_autopicker_star(
        df_particles, args.ugraph_dir, micrograph_star_output_dir
    )
    master_starfile_name = os.path.join(output_dir, "groundtruth.star")
    master_starfile.write_file(master_starfile_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    main(add_arguments(parser).parse_args())
