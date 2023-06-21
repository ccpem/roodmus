"""
    Script to compare estimated 3D alignments from RELION or CryoSPARC to the
    ground-truth orientation values used in Parakeet data generation.

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
from typing import List
import os

import pandas as pd
import seaborn as sns
import numpy as np

from roodmus.analysis.utils import load_data


def add_arguments(parser):
    parser.add_argument(
        "--config_dir",
        help="Directory with .mrc files and .yaml config files",
        type=str,
    )
    parser.add_argument(
        "--mrc_dir",
        help="directory with .mrc files and .yaml config files",
        type=str,
    )
    parser.add_argument(
        "--meta_file",
        help=(
            "Particle metadata file. Can be .star"
            " (RELION) or .cs (CryoSPARC)"
        ),
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--job_types",
        help=(
            "Labels for each metadata file. Must be the same length as"
            " 'meta_file'"
        ),
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--plot_dir",
        help="Output file name",
        type=str,
        default="alignment.png",
    )
    parser.add_argument(
        "--verbose", help="Increase output verbosity", action="store_true"
    )
    parser.add_argument(
        "--tqdm", help="show tqdm progress bar", action="store_true"
    )
    return parser


def get_name():
    return "plot_alignment"


def plot_picked_pose_distribution(
    df_picked: pd.DataFrame,
    metadata_filename: str | List[str],
    vmin: float | None = None,
    vmax: float | None = None,
):
    # group the picked particles by metadata file
    if isinstance(metadata_filename, list):
        metadata_filename = metadata_filename[0]
    df_picked_grouped = df_picked.groupby("metadata_filename").get_group(
        metadata_filename
    )

    # change data type of column euler_phi to float
    df_picked_grouped["euler_phi"] = df_picked_grouped["euler_phi"].astype(
        float
    )
    df_picked_grouped["euler_theta"] = -(
        df_picked_grouped["euler_theta"].astype(float) - np.pi / 2
    )
    df_picked_grouped["euler_psi"] = df_picked_grouped["euler_psi"].astype(
        float
    )

    # plot the alignment
    grid = sns.jointplot(
        x="euler_phi",
        y="euler_theta",
        data=df_picked_grouped,
        kind="hex",
        color="k",
        gridsize=55,
        bins="log",
        cmap="RdYlBu_r",
        marginal_kws=dict(bins=100, fill=False),
    )
    grid.fig.set_size_inches(14, 7)
    # adjust the x and y ticks to show multiples of pi
    grid.ax_joint.set_xticks(
        [
            -np.pi,
            -3 / 4 * np.pi,
            -np.pi / 2,
            -np.pi / 4,
            0,
            np.pi / 4,
            np.pi / 2,
            3 / 4 * np.pi,
            np.pi,
        ]
    )
    grid.ax_joint.set_xticklabels(
        [
            "\u03C0",
            "-3/4\u03C0",
            "-\u03C0/2",
            "-\u03C0/4",
            "0",
            "\u03C0/4",
            "\u03C0/2",
            "3/4\u03C0",
            "\u03C0",
        ]
    )
    grid.ax_joint.set_yticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
    grid.ax_joint.set_yticklabels(
        ["-\u03C0/2", "-\u03C0/4", "0", "\u03C0/4", "\u03C0/2"]
    )
    grid.ax_joint.set_xlabel("Azimuth")
    grid.ax_joint.set_ylabel("Elevation")
    # add new sublot to the right of the jointplot
    cbar_ax = grid.fig.add_axes([1, 0.15, 0.02, 0.7])
    # add colorbar to the new subplot
    grid.fig.colorbar(grid.ax_joint.collections[0], cax=cbar_ax, label="count")
    if vmin and vmax:
        # set limits of the colorbar to the same as for the picked particles
        grid.ax_joint.collections[0].set_clim(vmin, vmax)
    else:
        # get the limits of the colorbar
        vmin, vmax = grid.ax_joint.collections[0].get_clim()
    # add title to the top of the jointplot
    grid.fig.suptitle("picked particle pose distribution", fontsize=20, y=1.05)

    return grid, vmin, vmax


def plot_true_pose_distribution(
    df_truth: pd.DataFrame,
    vmin: float | None = None,
    vmax: float | None = None,
):
    df_truth["euler_phi"] = df_truth["euler_phi"].astype(float)
    df_truth["euler_theta"] = -(
        df_truth["euler_theta"].astype(float) - np.pi / 2
    )
    df_truth["euler_psi"] = df_truth["euler_psi"].astype(float)

    grid = sns.jointplot(
        x="euler_phi",
        y="euler_theta",
        data=df_truth,
        kind="hex",
        color="k",
        gridsize=55,
        bins="log",
        cmap="RdYlBu_r",
        marginal_kws=dict(bins=100, fill=False),
    )
    grid.fig.set_size_inches(14, 7)
    # adjust the x and y ticks to show multiples of pi
    grid.ax_joint.set_xticks(
        [
            -np.pi,
            -3 / 4 * np.pi,
            -np.pi / 2,
            -np.pi / 4,
            0,
            np.pi / 4,
            np.pi / 2,
            3 / 4 * np.pi,
            np.pi,
        ]
    )
    grid.ax_joint.set_xticklabels(
        [
            "\u03C0",
            "-3/4\u03C0",
            "-\u03C0/2",
            "-\u03C0/4",
            "0",
            "\u03C0/4",
            "\u03C0/2",
            "3/4\u03C0",
            "\u03C0",
        ]
    )
    grid.ax_joint.set_yticks([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2])
    grid.ax_joint.set_yticklabels(
        ["\u03C0/2", "\u03C0/4", "0", "\u03C0/4", "\u03C0/2"]
    )
    grid.ax_joint.set_xlabel("Azimuth")
    grid.ax_joint.set_ylabel("Elevation")
    # add new sublot to the right of the jointplot
    cbar_ax = grid.fig.add_axes([1, 0.15, 0.02, 0.7])
    # add colorbar to the new subplot
    grid.fig.colorbar(grid.ax_joint.collections[0], cax=cbar_ax, label="count")
    if vmin and vmax:
        # set limits of the colorbar to the same as for the picked particles
        grid.ax_joint.collections[0].set_clim(vmin, vmax)
    else:
        # get the limits of the colorbar
        vmin, vmax = grid.ax_joint.collections[0].get_clim()
    # add title to the top of the jointplot
    grid.fig.suptitle("true particle pose distribution", fontsize=20, y=1.05)

    return grid, vmin, vmax


def main(args):
    """This script analsyses the alignment of the picked
    particles with the true particles.
    """

    if not os.path.isdir(args.plot_dir):
        os.makedirs(args.plot_dir)

    if args.mrc_dir is None:
        args.mrc_dir = args.config_dir

    # parse the metadata files and job types
    meta_files, job_types, order = load_data.parse_jobtypes(
        args.meta_file, args.job_types
    )
    if args.verbose:
        print("Job types: {}".format(job_types))
        print("Metadata files: {}".format(meta_files))

    for i, meta_file in enumerate(meta_files):
        if i == 0:
            analysis = load_data(
                meta_file,
                args.config_dir,
                0,
                verbose=args.verbose,
                enable_tqdm=args.tqdm,
            )
        else:
            analysis.add_data(
                meta_file,
                args.config_dir,
                verbose=args.verbose,
                enable_tqdm=args.tqdm,
            )
    df_picked = pd.DataFrame(
        analysis.results_picking
    )  # data frame containing the picked particles
    df_truth = pd.DataFrame(
        analysis.results_truth
    )  # data frame containing the ground-truth particles

    print(f"meta_files in df: {df_picked['metadata_filename'].unique()}")

    # plot the picked particle pose distribution
    for i, meta_file in enumerate(df_picked["metadata_filename"].unique()):
        grid, vmin, vmax = plot_picked_pose_distribution(df_picked, meta_file)
        if i == 0:
            vmin_total = vmin
            vmax_total = vmax
        else:
            vmin_total = min(vmin_total, vmin)
            vmax_total = max(vmax_total, vmax)

        # save the plot
        outfilename = os.path.join(
            args.plot_dir,
            os.path.splitext(os.path.basename(meta_file))[0]
            + "_picked_pose_distribution.png",
        )
        grid.savefig(outfilename, dpi=300, bbox_inches="tight")
        grid.savefig(outfilename.replace(".png", ".pdf"), bbox_inches="tight")

    # plot the true particle pose distribution
    grid, vmin, vmax = plot_true_pose_distribution(
        df_truth, vmin_total, vmax_total
    )

    # save the plot
    outfilename = os.path.join(args.plot_dir, "true_pose_distribution.png")
    grid.savefig(outfilename, dpi=300, bbox_inches="tight")
    grid.savefig(outfilename.replace(".png", ".pdf"), bbox_inches="tight")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
