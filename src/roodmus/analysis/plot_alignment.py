"""Compare estimated 3D alignments from RELION or CryoSPARC to the
ground-truth orientation values used in Parakeet data generation.

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
from typing import List
import os

import pandas as pd
import seaborn as sns
import numpy as np

from roodmus.analysis.utils import load_data, plotDataFrame


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
    parser.add_argument(
        "--dpi",
        help="choose dots per inch in png plots, default to 100",
        type=int,
        default=100,
        required=False,
    )
    parser.add_argument("--pdf", help="save plot as pdf", action="store_true")
    return parser


def get_name():
    return "plot_alignment"


class plotTruePoseDistribution(plotDataFrame):
    def __init__(
        self,
        plot_data: dict[str, dict[str, pd.DataFrame]] | None = None,
        plot_dir: str = "",
        dpi: int = 300,
        pdf: bool = False,
    ) -> None:
        super().__init__(plot_data)

        # set up the dataframe
        if plot_data:
            self.plot_data = plot_data

        self.plot_dir = plot_dir
        self.dpi = dpi
        self.pdf = pdf

    def setup_plot_data(self, df_truth: pd.DataFrame):
        # set up the dict[str: dict[str: pd.DataFrame | None]] object
        # this can be
        self.plot_data = {
            "plot_truth_pose_distribution": {"df_truth": df_truth}
        }

    def setup_plot_data_empty(self):
        # set up the dict[str: dict[str: pd.DataFrame | None]] object
        # this can be
        self.plot_data = {"plot_truth_pose_distribution": {"df_truth": None}}

    def make_and_save_plots(
        self,
        vmin: float | None = None,
        vmax: float | None = None,
        overwrite_data: bool = False,
    ):
        # save/overwrite data file
        self.save_dataframes(self.plot_dir, overwrite_data)

        if isinstance(
            self.plot_data["plot_truth_pose_distribution"]["df_truth"],
            pd.DataFrame,
        ):
            self.grid, self.vmin, self.vmax = true_pose_distribution_plot(
                self.plot_data["plot_truth_pose_distribution"]["df_truth"],
                vmin,
                vmax,
            )
        else:
            raise TypeError(
                'self.plot_data["plot_truth_pose_distribution'
                '"]["df_truth"] is not a pd.DataFrame'
            )

        self._save_plot()

    def _save_plot(self):
        # save the plot
        outfilename = os.path.join(self.plot_dir, "true_pose_distribution.png")
        self.grid.savefig(outfilename, dpi=self.dpi, bbox_inches="tight")
        if self.pdf:
            self.grid.savefig(
                outfilename.replace(".png", ".pdf"), bbox_inches="tight"
            )


def true_pose_distribution_plot(
    df_truth: pd.DataFrame,
    vmin: float | None = None,
    vmax: float | None = None,
):
    df_truth["euler_phi"] = df_truth["euler_phi"].astype(float)
    df_truth["euler_theta"] = df_truth["euler_theta"].astype(float)
    # df_truth["euler_theta"] = -(
    #     df_truth["euler_theta"].astype(float) - np.pi / 2
    # )
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
            "-\u03C0",
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
    grid.ax_joint.set_yticks([0, np.pi / 4, np.pi / 2, 3 / 4 * np.pi, np.pi])
    grid.ax_joint.set_yticklabels(
        ["0", "\u03C0/4", "\u03C0/2", "3/4\u03C0", "\u03C0"]
    )
    grid.ax_joint.set_xlabel("Azimuth")
    grid.ax_joint.set_ylabel("Tilt")
    # add new sublot to the right of the jointplot
    cbar_ax = grid.fig.add_axes([1, 0.15, 0.02, 0.7])
    # add colorbar to the new subplot
    grid.fig.colorbar(grid.ax_joint.collections[0], cax=cbar_ax, label="count")
    if vmin and vmax:
        # set limits of the colorbar to the same as for
        # the picked particles
        grid.ax_joint.collections[0].set_clim(vmin, vmax)
    else:
        # get the limits of the colorbar
        vmin, vmax = grid.ax_joint.collections[0].get_clim()
    # add title to the top of the jointplot
    grid.fig.suptitle("true particle pose distribution", fontsize=20, y=1.05)

    return grid, vmin, vmax


class plotPickedPoseDistribution(plotDataFrame):
    def __init__(
        self,
        plot_data: dict[str, dict[str, pd.DataFrame]] | None = None,
        plot_dir: str = "",
        dpi: int = 300,
        pdf: bool = False,
    ) -> None:
        super().__init__(plot_data)

        # set up the dataframe
        if plot_data:
            self.plot_data = plot_data

        self.plot_dir = plot_dir
        self.dpi = dpi
        self.pdf = pdf

    def setup_plot_data(self, df_picked: pd.DataFrame):
        # set up the dict[str: dict[str: pd.DataFrame | None]] object
        # this can be
        self.plot_data = {
            "plot_picked_pose_distribution": {"df_picked": df_picked}
        }

    def setup_plot_data_empty(self):
        # set up the dict[str: dict[str: pd.DataFrame | None]] object
        # this can be
        self.plot_data = {"plot_picked_pose_distribution": {"df_picked": None}}

    def make_and_save_plots(
        self,
        overwrite_data: bool = False,
    ):
        # save/overwrite data file
        self.save_dataframes(self.plot_dir, overwrite_data)

        # now use the data to make and save the plots
        if isinstance(
            self.plot_data["plot_picked_pose_distribution"]["df_picked"],
            pd.DataFrame,
        ):
            for i, meta_file in enumerate(
                self.plot_data["plot_picked_pose_distribution"]["df_picked"][
                    "metadata_filename"
                ].unique()
            ):
                (
                    self.grid,
                    self.vmin,
                    self.vmax,
                ) = picked_pose_distribution_plot(
                    self.plot_data["plot_picked_pose_distribution"][
                        "df_picked"
                    ],
                    meta_file,
                )

                if i == 0:
                    self.vmin_total = self.vmin
                    self.vmax_total = self.vmax
                else:
                    self.vmin_total = min(self.vmin_total, self.vmin)
                    self.vmax_total = max(self.vmax_total, self.vmax)

                self._save_plot(meta_file)
        else:
            raise TypeError(
                'self.plot_data["plot_picked_pose_distribution'
                '"]["df_pickedh"] is not a pd.DataFrame'
            )

    def _save_plot(self, meta_file: str):
        # save the plot
        outfilename = os.path.join(
            self.plot_dir,
            os.path.splitext(os.path.basename(meta_file))[0]
            + "_picked_pose_distribution.png",
        )
        self.grid.savefig(outfilename, dpi=self.dpi, bbox_inches="tight")
        if self.pdf:
            self.grid.savefig(
                outfilename.replace(".png", ".pdf"), bbox_inches="tight"
            )


def picked_pose_distribution_plot(
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
    # df_picked_grouped["euler_phi"] = df_picked_grouped["euler_phi"].astype(
    #     float
    # )
    # df_picked_grouped["euler_theta"] = -(
    #     df_picked_grouped["euler_theta"].astype(float) - np.pi / 2
    # )
    # df_picked_grouped["euler_psi"] = df_picked_grouped["euler_psi"].astype(
    #     float
    # )

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
            "-\u03C0",
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
    grid.ax_joint.set_yticks([0, np.pi / 4, np.pi / 2, 3 / 4 * np.pi, np.pi])
    grid.ax_joint.set_yticklabels(
        ["0", "\u03C0/4", "\u03C0/2", "3/4\u03C0", "\u03C0"]
    )
    grid.ax_joint.set_xlabel("Azimuth")
    grid.ax_joint.set_ylabel("Tilt")
    # add new sublot to the right of the jointplot
    cbar_ax = grid.fig.add_axes([1, 0.15, 0.02, 0.7])
    # add colorbar to the new subplot
    grid.fig.colorbar(grid.ax_joint.collections[0], cax=cbar_ax, label="count")
    if vmin and vmax:
        # set limits of the colorbar to the
        # same as for the picked particles
        grid.ax_joint.collections[0].set_clim(vmin, vmax)
    else:
        # get the limits of the colorbar
        vmin, vmax = grid.ax_joint.collections[0].get_clim()
    # add title to the top of the jointplot
    grid.fig.suptitle("picked particle pose distribution", fontsize=20, y=1.05)

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
    plot_picked_pose_distribution = plotPickedPoseDistribution(
        plot_dir=args.plot_dir,
        dpi=args.dpi,
        pdf=args.pdf,
    )
    plot_picked_pose_distribution.setup_plot_data(df_picked)
    plot_picked_pose_distribution.make_and_save_plots(overwrite_data=True)

    # plot the true particle pose distribution
    plot_true_pose_distribution = plotTruePoseDistribution(
        plot_dir=args.plot_dir,
        dpi=args.dpi,
        pdf=args.pdf,
    )
    plot_true_pose_distribution.setup_plot_data(df_truth)
    plot_true_pose_distribution.make_and_save_plots(
        vmin=plot_picked_pose_distribution.vmin_total,
        vmax=plot_picked_pose_distribution.vmax_total,
        overwrite_data=True,
    )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
