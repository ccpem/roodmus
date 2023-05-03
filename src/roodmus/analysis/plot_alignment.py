"""
    Script to compare estimated 3D alignments from RELION or CryoSPARC to the
    ground-truth orientation values used in Parakeet data generation
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
        "--mrc-dir",
        help="directory with .mrc files and .yaml config files",
        type=str,
    )
    parser.add_argument(
        "--meta-file",
        help=(
            "Particle metadata file. Can be .star"
            " (RELION) or .cs (CryoSPARC)"
        ),
        type=str,
    )
    parser.add_argument(
        "--plot-dir",
        help="Output file name",
        type=str,
        default="alignment.png",
    )
    parser.add_argument(
        "--verbose", help="Increase output verbosity", action="store_true"
    )
    return parser


def get_name():
    return "analyse_alignment"


def plot_picked_pose_distribution(
    df_picked: pd.DataFrame,
    metadata_filename: str | List[str],
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
    # get the limits of the colorbar
    vmin, vmax = grid.ax_joint.collections[0].get_clim()
    # add title to the top of the jointplot
    grid.fig.suptitle("picked particle pose distribution", fontsize=20, y=1.05)

    return grid, vmin, vmax


def plot_true_pose_distribution(
    df_truth: pd.DataFrame,
    vmin: float,
    vmax: float,
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
    # set the limits of the colorbar to the same as for the picked particles
    grid.ax_joint.collections[0].set_clim(vmin, vmax)
    # add title to the top of the jointplot
    grid.fig.suptitle("true particle pose distribution", fontsize=20, y=1.05)

    return grid


def main(args):
    """This script analsyses the alignment of the picked
    particles with the true particles.
    """
    if args.mrc_dir is None:
        args.mrc_dir = args.config_dir

    for i, meta_file in enumerate(args.meta_file):
        if i == 0:
            analysis = load_data(
                meta_file,
                args.config_dir,
                args.particle_diameter,
                verbose=args.verbose,
            )
        else:
            analysis.add_data(meta_file, args.config_dir, verbose=args.verbose)
    df_picked = pd.DataFrame(
        analysis.results_picking
    )  # data frame containing the picked particles
    df_truth = pd.DataFrame(
        analysis.results_truth
    )  # data frame containing the ground-truth particles

    # plot the picked particle pose distribution
    vmin_total = 0
    vmax_total = 0
    for meta_file in args.meta_file:
        grid, vmin, vmax = plot_picked_pose_distribution(df_picked, meta_file)
        vmin_total = min(vmin_total, vmin)
        vmax_total = max(vmax_total, vmax)

        # save the plot
        outfilename = os.path.join(
            args.plot_dir,
            os.path.splitext(os.path.basename(meta_file))[0]
            + "_picked_pose_distribution.png",
        )
        grid.savefig(outfilename, dpi=300, bbox_inches="tight")

    # plot the true particle pose distribution
    grid = plot_true_pose_distribution(df_truth, vmin_total, vmax_total)

    # save the plot
    outfilename = os.path.join(args.plot_dir, "true_pose_distribution.png")
    grid.savefig(outfilename, dpi=300, bbox_inches="tight")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
