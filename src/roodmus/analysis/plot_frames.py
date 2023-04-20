"""script to plot the distribution of frames in a job
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from roodmus.analysis.utils import load_data


def add_arguments(parser):
    parser.add_argument(
        "--config-dir",
        help="Directory with .mrc files and .yaml config files",
        type=str,
    )
    parser.add_argument(
        "--mrc-dir",
        help="Directory with .mrc files. The same as 'config-dir' by default",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--meta-file",
        help=(
            "Particle metadata file. Can be .star (RELION) or .cs (CryoSPARC)"
        ),
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--jobtypes",
        help=(
            "Labels for each metadata file. Must be the same length as"
            " 'meta-file'"
        ),
        type=str,
        nargs="+",
    )
    parser.add_argument(
        "--particle-diameter",
        help=(
            "Expected maximum particle diameter. Used to limit search radius"
            " for matching picked particles to truth particles"
        ),
        type=float,
        default=250.0,
        required=False,
    )
    parser.add_argument("--plot-dir", help="output file name", type=str)
    parser.add_argument(
        "--verbose", help="increase output verbosity", action="store_true"
    )
    return parser


def get_name():
    return "plot_frames"


def plot_frame_distribution(
    df_picked: pd.DataFrame,
    metadata_filename: str,
    df_truth: pd.DataFrame,
    particle_diameter: float,
    jobtypes: dict,
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

    plt.rcParams["font.size"] = 20
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.histplot(
        df_picked.groupby("metadata_filename").get_group(metadata_filename)[
            "closest_pdb_index"
        ],
        ax=ax,
        bins=100,
        kde=True,
    )
    sns.histplot(
        df_truth["pdb_index"],
        ax=ax,
        bins=100,
        kde=True,
        color="red",
        alpha=0.2,
    )
    ax.set_xlabel("frame index")
    ax.set_ylabel("count")
    ax.set_title(jobtypes[metadata_filename])
    fig.tight_layout()
    fig.legend(
        ["picked", "truth"],
        loc="lower center",
        ncol=1,
        bbox_to_anchor=(1.1, 0.85),
    )
    return fig, ax


def main(args):
    """plot the distribution of frames from the source MD trajectory
    in the particle set for the given job"""

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

    # plot the distribution of frames
    print("plotting the frame distribution...")
    if args.jobtypes:
        jobtypes = {
            meta_file: jobtype
            for meta_file, jobtype in zip(args.meta_file, args.jobtypes)
        }
    for i, metadata_filename in enumerate(args.meta_file):
        print(f"plotting frames in {metadata_filename}...")
        fig, ax = plot_frame_distribution(
            df_picked,
            metadata_filename,
            df_truth,
            args.particle_diameter,
            jobtypes,
        )
        outfilename = os.path.join(
            args.plot_dir,
            f"{jobtypes[metadata_filename]}_frame_distribution.png",
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
