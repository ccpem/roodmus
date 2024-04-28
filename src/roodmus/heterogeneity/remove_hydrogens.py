"""Read a directory of pdb files, remove the hydrogens to speed read/write
input/output for computation in future and then save to a new dir

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

import argparse
import os
import shutil

from roodmus.heterogeneity.het_metrics import (
    get_pdb_list,
    select_confs,
)


def add_arguments(parser: argparse.ArgumentParser):
    """Parse arguments for performing one or more clustering workflows
    (as configured by provided argument permutations)

    Args:
        parser (argparse.ArgumentParser): _description_

    Returns:
        _type_: _description_
    """

    parser.add_argument(
        "--conformations_dir",
        "-c",
        help="Directory with .pdb files",
        type=str,
    )

    parser.add_argument(
        "--n_confs",
        help="Limit to <n_confs> conformations",
        type=int,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--contiguous_confs",
        help="Set the sampled conformations to be contiguous instead of"
        " uniformly sampled in time",
        action="store_true",
    )

    parser.add_argument(
        "--first_conf",
        help="Set an index (used after alphanumeric sorting) to set first"
        " conformation to be sampled from",
        type=int,
        default=0,
        required=False,
    )

    parser.add_argument(
        "--verbose", help="increase output verbosity", action="store_true"
    )

    parser.add_argument(
        "--output_dir",
        help="Directory to save results and intermediate results",
        type=str,
        default="het_metrics",
        required=False,
    )

    parser.add_argument(
        "--file_ext",
        help="File extension of the conformation files. Default is .pdb",
        type=str,
        default=".pdb",
    )


def get_name():
    return "remove_hydrogens"


def read_pdbs(args) -> list[str]:
    conf_files = get_pdb_list(
        args.conformations_dir,
        args.file_ext,
    )

    conf_files = sorted(conf_files)

    if args.n_confs:
        conf_files = conf_files[: args.n_confs]

    if args.verbose:
        print(conf_files)

    return conf_files


def main(args):
    # check if gemmi executable is in path
    assert shutil.which("gemmi"), "Install gemmi cmd line program via"
    " pip install gemmi-program==0.6.5"

    # read in the pdb files
    conf_files = get_pdb_list(
        args.conformations_dir,
        args.file_ext,
    )

    # grab the specified ones
    conf_files = select_confs(
        conf_files,
        args.n_confs,
        args.first_conf,
        args.contiguous_confs,
    )

    # make new dir for outputs
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # remove the hydrogens and save to output dir
    # one-by-one
    for conf_file in conf_files:
        # get fpath for output file
        output_file = os.path.join(
            args.output_dir,
            os.path.basename(conf_file),
        )
        # remove the hydrogens via gemmi cmd line program
        command = "gemmi h --remove {} {}".format(conf_file, output_file)
        os.system(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
