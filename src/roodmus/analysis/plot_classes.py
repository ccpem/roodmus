"""Visualise 2D classification results.

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

# from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import zoom

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
            " 'meta_file'"
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
    parser.add_argument(
        "--bin_factor",
        help=("Binning for the frame_distribution plot. Defaults to 100"),
        type=int,
        default=100,
        required=False,
    )
    parser.add_argument("--plot_dir", help="Output directory", type=str)
    parser.add_argument(
        "--plot_types",
        help="Types of analysis results to plot",
        type=str,
        nargs="+",
        choices=[
            "precision",
            "frame_distribution",
        ],
    )
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
    return "plot_classes"


class plot2DClasses(plotDataFrame):
    def __init__(
        self,
        job_types: dict[str, str],
        plot_types: list[str],
        plot_data: dict[str, dict[str, pd.DataFrame]] | None = None,
        plot_dir: str = "",
        bin_factor: int = 100,
        dpi: int = 300,
        pdf: bool = False,
    ) -> None:
        super().__init__(plot_data)

        if plot_data:
            self.plot_data = plot_data

        self.job_types = job_types
        self.plot_types = plot_types
        self.plot_dir = plot_dir
        self.bin_factor = bin_factor
        self.dpi = dpi
        self.pdf = pdf

    def setup_plot_data(self, df_picked: pd.DataFrame):
        self.plot_data = {"plot_classes": {"df_picked": df_picked}}

    def setup_plot_data_empty(self):
        self.plot_data = {"plot_classes": {"df_picked": None}}

    def make_and_save_plots(
        self,
        overwrite_data: bool = False,
    ):
        # save/overwrite data file
        self.save_dataframes(self.plot_dir, overwrite_data)

        # now use the data to make and save the plots
        if isinstance(
            self.plot_data["plot_classes"]["df_picked"],
            pd.DataFrame,
        ):
            for metadata_filename in self.plot_data["plot_classes"][
                "df_picked"
            ]["metadata_filename"].unique():
                # check there are 2d classes
                if check_for_2d_class_labels(
                    self.plot_data["plot_classes"]["df_picked"],
                    metadata_filename,
                ):
                    if "precision" in self.plot_types:
                        print(
                            f"plotting 2D class precision for \
                            {metadata_filename}..."
                        )
                        (
                            self.precision_fig,
                            self.precision_ax,
                        ) = plot_2Dclass_precision(
                            self.plot_data["plot_classes"]["df_picked"],
                            metadata_filename,
                            self.job_types,
                        )
                        self._save_precision_plot(metadata_filename)

                    if "frame_distribution" in self.plot_types:
                        print(
                            f"plotting 2D class frame distribution \
                            for {metadata_filename}..."
                        )
                        (
                            self.frame_dist_fig,
                            self.frame_dist_ax,
                        ) = plot_2Dclasses_frames(
                            self.plot_data["plot_classes"]["df_picked"],
                            metadata_filename,
                            self.bin_factor,
                        )
                        self._save_frame_distribution_plot(metadata_filename)
        else:
            raise TypeError(
                'self.plot_data["plot_classes"]["df_picked"]'
                "is not a pd.DataFrame"
            )

    def _save_precision_plot(self, meta_file: str):
        outfilename = os.path.join(
            self.plot_dir,
            "{}_2Dclass_precision.png".format(self.job_types[meta_file]),
        )
        self.precision_fig.savefig(
            outfilename, dpi=self.dpi, bbox_inches="tight"
        )
        if self.pdf:
            self.precision_fig.savefig(
                outfilename.replace(".png", ".pdf"),
                bbox_inches="tight",
            )

    def _save_frame_distribution_plot(self, meta_file: str):
        outfilename = os.path.join(
            self.plot_dir,
            "{}_2Dclass_frame_distribution.png".format(
                self.job_types[meta_file]
            ),
        )
        self.frame_dist_fig.savefig(
            outfilename, dpi=self.dpi, bbox_inches="tight"
        )
        if self.pdf:
            self.frame_dist_fig.savefig(
                outfilename.replace(".png", ".pdf"),
                bbox_inches="tight",
            )


def plot_2Dclass_precision(
    df_picked: pd.DataFrame,
    metadata_filename: str,
    job_types: dict,
    palette: str = "YlGnBu",
):
    df_grouped = df_picked.groupby("metadata_filename").get_group(
        metadata_filename
    )
    results: dict = {
        "class2D": [],
        "precision": [],
        "average defocus": [],
    }
    for groupname in df_grouped.groupby("class2D").groups.keys():
        precision = df_grouped.groupby("class2D").get_group(groupname)[
            "TP"
        ].sum() / (
            df_grouped.groupby("class2D").get_group(groupname)["TP"].size
        )
        results["class2D"].append(groupname)
        results["precision"].append(precision)
        results["average defocus"].append(
            df_grouped.groupby("class2D")
            .get_group(groupname)["defocusU"]
            .mean()
        )
    df = pd.DataFrame(results)
    df["class2D"] = df["class2D"].astype(int)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    sns.barplot(x="class2D", y="precision", data=df, ax=ax, palette=palette)
    ax.set_xlabel("class2D", fontsize=12)
    ax.set_ylabel("precision", fontsize=12)
    ax.set_title(job_types[metadata_filename], fontsize=14)
    xticklabels = ax.get_xticklabels()
    ax.set_xticklabels(xticklabels, rotation=45, fontsize=6)
    fig.tight_layout()
    return fig, ax


def plot_2Dclasses_frames(
    df_picked: pd.DataFrame,
    metadata_filename: str,
    bin_factor: int = 100,
    palette="YlGnBu",
):
    df_filtered = df_picked.groupby("metadata_filename").get_group(
        metadata_filename
    )
    # df_grouped = df_filtered.groupby(["class2D", "closest_pdb_index"])
    heatmap = np.zeros(
        (
            int(np.max(df_filtered["class2D"])) + 1,
            int(np.max(df_filtered["closest_pdb_index"])) + 1,
        )
    )
    # class_id: float
    # pdb_id: float
    # df_grouped.groups: Dict[Tuple[float, float], pd.Index[Any]]
    # for class_id, pdb_id in df_grouped.groups.keys():
    #     if not np.isnan(class_id) and not np.isnan(pdb_id):

    for class_id in df_filtered["class2D"].unique():
        for pdb_id in df_filtered["closest_pdb_index"].unique():
            if not np.isnan(class_id) and not np.isnan(pdb_id):
                # num = df_grouped.get_group((class_id, pdb_id)).size
                num = df_filtered[
                    (df_filtered["class2D"] == class_id)
                    & (df_filtered["closest_pdb_index"] == pdb_id)
                ].size
                if num != np.nan:
                    heatmap[int(class_id), int(pdb_id)] += num

    # apply binning to the heatmap
    heatmap = zoom(heatmap, [1, 1 / bin_factor], order=0)
    heatmap[heatmap == 0] = np.nan

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(heatmap, ax=ax, cmap=palette)
    return fig, ax


def check_for_2d_class_labels(
    df_picked: pd.DataFrame, metadata_filename: str
) -> bool:
    if np.sum(
        pd.isnull(
            np.unique(
                df_picked.groupby("metadata_filename").get_group(
                    metadata_filename
                )["class2D"]
            )
        )
    ) == len(
        np.unique(
            df_picked.groupby("metadata_filename").get_group(
                metadata_filename
            )["class2D"]
        )
    ):
        print(
            "{} metadata contains no 2D class labels, skipping...".format(
                metadata_filename
            )
        )
        return False
    else:
        return True


def main(args):
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

    print(f"meta_files in df: {df_picked['metadata_filename'].unique()}")

    # compute the precision for the picked particles
    _, df_picked = analysis.compute_precision(
        df_picked, df_truth, verbose=args.verbose
    )

    plot_2d_classes = plot2DClasses(
        job_types,
        args.plot_types,
        plot_dir=args.plot_dir,
        bin_factor=args.bin_factor,
        dpi=args.dpi,
        pdf=args.pdf,
    )
    plot_2d_classes.setup_plot_data(df_picked)
    plot_2d_classes.make_and_save_plots(overwrite_data=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
