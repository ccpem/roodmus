"""Roodmus:
A software framework to allow simulation of cryo-EM micrographs and
tomograms by sampling conformations from MD simulations using Parakeet
software (https://doi.org/10.1098/rsob.210160). Utilities allow analysis
and evaluation of the performance of structure determination pipelines
and heterogeneous reconstruction algorithms.

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
import pkg_resources
from pathlib import Path

import roodmus.simulation.run_parakeet
import roodmus.trajectory.conformations_sampling
import roodmus.analysis.plot_ctf
import roodmus.analysis.plot_picking
import roodmus.analysis.plot_frames
import roodmus.analysis.plot_classes
import roodmus.analysis.extract_particles
import roodmus.analysis.plot_alignment

# import analysis.analyse_alignment


def get_roodmus_parent() -> Path:
    return Path(__file__).parent.parent.parent


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Roodmus version {}, license is found in {}".format(
            pkg_resources.get_distribution("roodmus").version,
            os.path.join(get_roodmus_parent(), "LICENSE"),
        ),
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        description="valid subcommands",
        help="",
    )
    subparsers.required = True

    modules = [
        roodmus.trajectory.conformations_sampling,
        roodmus.simulation.run_parakeet,
        roodmus.analysis.plot_ctf,
        roodmus.analysis.plot_picking,
        roodmus.analysis.plot_frames,
        roodmus.analysis.plot_classes,
        roodmus.analysis.plot_alignment,
        roodmus.analysis.extract_particles,
    ]

    module_helptext = [
        "Sampling a molecular dynamics trajectory and saving the"
        + " conformations to PDB files.",
        "Simulation of micrograph/tomogram dataset using Parakeet software.",
        "Plot a comparison between the estimated CTF parameters and the"
        + " true values used in data generation.",
        "Plot statistics from picking analyses and overlays of"
        + " picked and truth particles on micrographs.",
        "Plot the distribution of frames in a job.",
        "Visualise 2D classification results.",
        "Compare estimated 3D alignments from RELION or CryoSPARC to the"
        + " ground-truth orientation values used in Parakeet data generation.",
        "Extract a stack of particles from a set of simulated micrographs"
        + " using the ground-truth positions.",
    ]

    for helptext, module in zip(module_helptext, modules):
        this_parser = subparsers.add_parser(
            module.get_name(),
            help=helptext,
        )
        module.add_arguments(this_parser)
        this_parser.set_defaults(func=module.main)

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()
