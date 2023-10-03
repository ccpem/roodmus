"""Plot the distribution of frames in a job.

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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from roodmus.analysis.utils import load_data, plotDataFrame


def add_arguments(parser):
    parser.add_argument(
        "--config_dir",
        help="Directory with .mrc files and .yaml config files",
        type=str,
    )
    parser.add_argument(
        "--mrc_dir",
        help="Directory with .mrc files. The same as 'config-dir' by default",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--meta_file",
        help=(
            "Particle metadata file. Can be .star (RELION) or .cs (CryoSPARC)"
        ),
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--job_types",
        help=(
            "Labels for each metadata file. Must be the same length as"
            " 'meta-file'"
        ),
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--particle_diameter",
        help=(
            "Expected maximum particle diameter. Used to limit search radius"
            " for matching picked particles to truth particles"
        ),
        type=float,
        default=250.0,
        required=False,
    )
    parser.add_argument("--plot_dir", help="output file name", type=str)
    parser.add_argument(
        "--verbose", help="increase output verbosity", action="store_true"
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
    return "plot_frames"


class plotFrameDistribution(plotDataFrame):
    def __init__(
        self,
        job_types: dict[str, str],
        plot_data: dict[str, dict[str, pd.DataFrame]] | None = None,
        plot_dir: str = "",
        particle_diameter: float = 200,
        dpi: int = 300,
        pdf: bool = False,
    ) -> None:
        super().__init__(plot_data)

        if plot_data:
            self.plot_data = plot_data

        self.job_types = job_types
        self.plot_dir = plot_dir
        self.particle_diameter = particle_diameter
        self.dpi = dpi
        self.pdf = pdf

    def setup_plot_data(
        self,
        df_truth: pd.DataFrame,
        df_picked: pd.DataFrame,
    ):
        self.plot_data = {"frame_distribution": {}}
        self.plot_data["frame_distribution"]["df_truth"] = df_truth
        self.plot_data["frame_distribution"]["df_picked"] = df_picked

    def setup_plot_data_empty(
        self,
    ):
        self.plot_data = {"frame_distribution": {}}
        self.plot_data["frame_distribution"]["df_truth"] = None
        self.plot_data["frame_distribution"]["df_picked"] = None

    def make_and_save_plots(
        self,
        overwrite_data: bool = False,
    ):
        self.save_dataframes(self.plot_dir, overwrite_data)

        if isinstance(
            self.plot_data["frame_distribution"]["df_truth"],
            pd.DataFrame,
        ) and isinstance(
            self.plot_data["frame_distribution"]["df_picked"],
            pd.DataFrame,
        ):
            # plot the distribution of frames
            print("plotting the frame distribution...")
            for i, metadata_filename in enumerate(
                self.plot_data["frame_distribution"]["df_picked"][
                    "metadata_filename"
                ].unique()
            ):
                print(f"plotting frames in {metadata_filename}...")
                fig, ax = plot_frame_distribution(
                    self.plot_data["frame_distribution"]["df_picked"],
                    metadata_filename,
                    self.plot_data["frame_distribution"]["df_truth"],
                    self.particle_diameter,
                    self.job_types,
                )
                meta_basename = os.path.basename(metadata_filename)
                outfilename = os.path.join(
                    self.plot_dir,
                    f"{meta_basename.split('.')[0]}_frame_distribution.png",
                )

                self._save_plot(fig, ax, outfilename)

        else:
            raise TypeError("df_truth or df_picked is not a pd.DataFrame!")

    def _save_plot(self, fig, ax, outfilename: str):
        # save the plot
        fig.savefig(outfilename, dpi=self.dpi, bbox_inches="tight")
        if self.pdf:
            fig.savefig(
                outfilename.replace(".png", ".pdf"),
                bbox_inches="tight",
            )
        fig.clf()


def plot_frame_distribution(
    df_picked: pd.DataFrame,
    metadata_filename: str,
    df_truth: pd.DataFrame,
    particle_diameter: float,
    jobtypes: dict[str, str],
):
    # check if the precision has been computed for the picked particles
    if "closest_pdb_index" not in df_picked.columns:
        raise ValueError(
            f"Precision has not been computed for {metadata_filename}."
            " Run 'compute_precision' first"
        )

    df_picked["closest_pdb_index"] = df_picked["closest_pdb"].apply(
        lambda x: int(x.split("_")[-1].split(".")[0])
    )
    # set the closest_pdb_index to np.nan if the particle
    # is not closer to a truth particle than the particle diameter
    df_picked.loc[
        df_picked["closest_dist"] > particle_diameter, "closest_pdb_index"
    ] = np.nan
    df_truth["pdb_index"] = df_truth["pdb_filename"].apply(
        lambda x: int(x.split("_")[-1].split(".")[0])
    )

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    sns.histplot(
        df_picked.groupby("metadata_filename").get_group(metadata_filename)[
            "closest_pdb_index"
        ],
        ax=ax,
        bins=25,
        kde=True,
    )
    sns.histplot(
        df_truth["pdb_index"],
        ax=ax,
        bins=25,
        kde=True,
        color="red",
        alpha=0.2,
    )
    ax.set_xlabel("frame index", fontsize=12)
    ax.set_ylabel("count", fontsize=12)
    ax.set_title(jobtypes[metadata_filename], fontsize=14)
    fig.tight_layout()
    fig.legend(
        ["picked", "truth"],
        loc="lower center",
        ncol=2,
        bbox_to_anchor=(0.5, -0.1),
        fontsize=12,
    )
    return fig, ax


def main(args):
    """plot the distribution of frames from the source MD trajectory
    in the particle set for the given job"""

    if not os.path.isdir(args.plot_dir):
        os.makedirs(args.plot_dir)

    if args.mrc_dir is None:
        args.mrc_dir = args.config_dir

    # parse the metadata files and job types
    meta_files, job_types, _ = load_data.parse_jobtypes(
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
                args.particle_diameter,
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

    # compute the precision for the picked particles
    _, df_picked = analysis.compute_precision(
        df_picked, df_truth, verbose=args.verbose
    )

    frame_distribution = plotFrameDistribution(
        job_types,
        plot_dir=args.plot_dir,
        particle_diameter=args.particle_diameter,
        dpi=args.dpi,
        pdf=args.pdf,
    )
    frame_distribution.setup_plot_data(df_truth, df_picked)
    frame_distribution.make_and_save_plots(overwrite_data=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
