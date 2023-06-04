"""script to visualise 2D classification results
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import zoom

from roodmus.analysis.utils import load_data


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
    return parser


def get_name():
    return "plot_classes"


def plot_2Dclass_precision(
    df_picked: pd.DataFrame,
    metadata_filename: str,
    job_types: dict,
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
        results["class2D"].append(int(groupname))
        results["precision"].append(precision)
        results["average defocus"].append(
            df_grouped.groupby("class2D")
            .get_group(groupname)["defocusU"]
            .mean()
        )
    df = pd.DataFrame(results)

    plt.rcParams["font.size"] = 20
    fig, ax = plt.subplots(figsize=(25, 10))
    sns.barplot(x="class2D", y="precision", data=df, ax=ax, palette="YlGnBu")
    ax.set_xlabel("class2D")
    ax.set_ylabel("precision")
    ax.set_title(job_types[metadata_filename])
    # remove every second xtick label
    fig.tight_layout()
    return fig, ax


def plot_2Dclasses_frames(
    df_picked: pd.DataFrame, metadata_filename: str, bin_factor: int = 100
):
    df_filtered = df_picked.groupby("metadata_filename").get_group(
        metadata_filename
    )
    df_grouped = df_filtered.groupby(["class2D", "closest_pdb_index"])
    heatmap = np.zeros(
        (
            int(np.max(df_filtered["class2D"])) + 1,
            int(np.max(df_filtered["closest_pdb_index"])) + 1,
        )
    )
    for class_id, pdb_id in df_grouped.groups.keys():
        if int(class_id) > 0 and int(pdb_id) > 0:
            num = df_grouped.get_group((class_id, pdb_id)).size
            if num != np.nan:
                heatmap[int(class_id), int(pdb_id)] += num

    # apply binning to the heatmap
    heatmap = zoom(heatmap, [1, 1 / bin_factor], order=0)
    heatmap[heatmap == 0] = np.nan

    fig, ax = plt.subplots(figsize=(15, 5))
    sns.heatmap(heatmap, ax=ax, cmap="RdYlBu")
    return fig, ax


def main(args):
    if not os.path.isdir(args.plot_dir):
        os.makedirs(args.plot_dir)

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

    # compute the precision for the picked particles
    _, df_picked = analysis.compute_precision(
        df_picked, df_truth, verbose=args.verbose
    )

    if args.job_types:
        job_types = {
            meta_file: jobtype
            for meta_file, jobtype in zip(args.meta_file, args.job_types)
        }

    # plot the precision per class
    for plot_type in args.plot_types:
        if plot_type == "precision":
            for metadata_filename in args.meta_file:
                # gives TypeError: ufunc 'isnan' not supported for the input
                #  types, and the inputs could not be safely coerced to any
                # supported types according to the casting rule ''safe''
                """
                if np.sum(
                    np.isnan(
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
                    print("metadata contains no 2D class labels, skipping...")
                else:
                """
                print(
                    f"plotting 2D class precision for \
                    {metadata_filename}..."
                )
                fig, ax = plot_2Dclass_precision(
                    df_picked,
                    metadata_filename,
                    job_types,
                )
                outfilename = os.path.join(
                    args.plot_dir,
                    f"{job_types[metadata_filename]}_2Dclass_precision.png",
                )
                fig.savefig(outfilename, dpi=300)
                fig.clf()

        if plot_type == "frame_distribution":
            for metadata_filename in args.meta_file:
                # gives TypeError: ufunc 'isnan' not supported for the input
                # types, and the inputs could not be safely coerced to any
                # supported types according to the casting rule ''safe''
                """
                if np.sum(
                    np.isnan(
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
                    print("metadata contains no 2D class labels, skipping...")
                else:
                """
                print(
                    f"plotting 2D class frame distribution \
                    for {metadata_filename}..."
                )
                fig, ax = plot_2Dclasses_frames(
                    df_picked, metadata_filename, args.bin_factor
                )
                outfilename = os.path.join(
                    args.plot_dir,
                    "{}_2Dclass_frame_distribution.png".format(
                        job_types[metadata_filename]
                    ),
                )
                fig.savefig(outfilename, dpi=300)
                fig.clf()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
