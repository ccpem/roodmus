"""Extract a stack of particles from a set of simulated micrographs.

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

import argparse
import os
import yaml
from typing import List

from tqdm import tqdm
import numpy as np
import mrcfile
import tifffile


def add_arguments(parser):
    parser.add_argument(
        "--config_dir",
        help="Directory with .mrc files and .yaml config files",
        type=str,
    )
    parser.add_argument(
        "--mrc_dir",
        help="Directory with .mrc files. The same as 'config_dir' by default",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-N",
        "--num_ugraphs",
        help="Number of micrographs to consider in analyses. Default 'all'",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--box_size",
        help="Box size in pixels. Default 128",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--particle_dir",
        help="Directory to save extracted particles. Default 'particles'",
        type=str,
        default="particles",
    )
    parser.add_argument(
        "--tqdm",
        help="Use tqdm progress bar. Default False",
        action="store_true",
    )
    parser.add_argument(
        "--verbose",
        help="Print details to stdout",
        action="store_true",
    )
    return parser


def get_name():
    return "extract_particles"


def get_num_molecules(metadata: dict) -> int:
    num_molecules = 0
    for molecule in metadata["sample"]["molecules"]["local"]:
        if type(molecule["instances"]) == int:
            num_molecules += int(molecule["instances"])
        elif type(molecule["instances"]) == list:
            num_molecules += len(molecule["instances"])
    return num_molecules


def get_particle(
    micrograph: np.ndarray,
    position: List[int],
    box_size: int,
):
    particle = micrograph[
        0,
        position[0] - box_size // 2 : position[0] + box_size // 2,
        position[1] - box_size // 2 : position[1] + box_size // 2,
    ]
    return particle


def main(args):
    if args.mrc_dir is None:
        args.mrc_dir = args.config_dir

    if not os.path.exists(args.particle_dir):
        os.makedirs(args.particle_dir)

    # loop over the micrographs in the directory and load their configs
    # then extract the particles in a box of size 'box_size'
    # if the particles are too close to the edge of the micrograph, skip it

    for filename in os.listdir(args.mrc_dir)[: args.num_ugraphs]:
        if filename.endswith(".mrc"):
            micrograph = mrcfile.open(
                os.path.join(args.mrc_dir, filename), mode="r"
            ).data

            if args.verbose:
                print("Adding {} to particle stack".format(filename))

            config_filename = filename.replace(".mrc", ".yaml")
            config_path = os.path.join(args.config_dir, config_filename)
            config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

            particle_list: List = []
            progressbar = tqdm(
                total=get_num_molecules(config), disable=not args.tqdm
            )
            for molecule in config["sample"]["molecules"]["local"]:
                # filename = molecule["filename"]
                for instance in molecule["instances"]:
                    position = instance["position"]
                    position = [int(float(r)) for r in position]

                    particle = get_particle(
                        micrograph, position, args.box_size
                    )

                    # if the particle is not the right size, skip it
                    if particle.shape != (args.box_size, args.box_size):
                        progressbar.update(1)
                        continue

                    particle_list.append(particle)
                    progressbar.update(1)

            progressbar.close()
            particle_stack = np.array(particle_list)
            if args.verbose:
                print(f"particle stack shape: {particle_stack.shape}")

            # save the particle stack
            particle_file = os.path.join(
                args.particle_dir,
                "particle_stack.mrc",
            )

            with mrcfile.new(particle_file, overwrite=True) as mrc:
                mrc.set_data(np.float32(particle_stack))
                mrc.set_image_stack()
                mrc.update_header_from_data()
            tifffile.imwrite(
                particle_file.replace(".mrc", ".tif"), particle_stack
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
