"""Plotting functions for analysis of heterogeneous reconstructions

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
import pandas as pd
import seaborn as sns

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
    parser.add_argument("--pdf", help="save plot as pdf", action="store_true")
    return parser


def get_name():
    return "plot_heterogeneous_reconstruction"


def plot_latent_space_scatter(
    df_picked,
    dim_1: int = 0,
    dim_2: int = 1,
    pca: bool = False,
    tsne: bool = False,
    color_by: str | None = None,
    palette: str = "warmcool",
):
    """
    plotting the latent space of the picked particles
    if dim_1 and dim_2 are specified,
        plot the latent space in those dimensions
    if dim_1 or dim_2 is specified,
        plot a histogram of the latent space in that dimension
    if pca is True,
        use PCA to reduce the dimensionality of the latent space
    if umap is True,
        use UMAP to reduce the dimensionality of the latent space
    if color_by is provided,
        color the points by the relevant variable
    """

    if pca:
        varname = "PCA"
    elif tsne:
        varname = "tSNE"
    else:
        varname = "latent"

    if color_by is None:
        grid = sns.jointplot(
            x=f"{varname}_{dim_1}",
            y=f"{varname}_{dim_2}",
            data=df_picked,
            kind="scatter",
            color=(107 / 255, 174 / 255, 214 / 255),
            marginal_kws=dict(bins=100, fill=False),
            s=5,
            height=3.5,
        )
        grid.set_axis_labels(f"{varname}_{dim_1}", f"{varname}_{dim_2}")
        return grid
    else:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))
        sns.scatterplot(
            x=f"{varname}_{dim_1}",
            y=f"{varname}_{dim_2}",
            data=df_picked,
            hue=color_by,
            palette=palette,
            ax=ax,
            s=5,
        )
        ax.set_xlabel(f"{varname}_{dim_1}")
        ax.set_ylabel(f"{varname}_{dim_2}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        return fig, ax


def plot_latent_space_hexbin(
    df_picked,
    dim_1: int = 0,
    dim_2: int = 1,
    pca: bool = False,
    tsne: bool = False,
    palette: str = "warmcool",
):
    """
    plotting the latent space of the picked particles
    if dim_1 and dim_2 are specified,
        plot the latent space in those dimensions
    if dim_1 or dim_2 is specified,
        plot a histogram of the latent space in that dimension
    if pca is True,
        use PCA to reduce the dimensionality of the latent space
    if umap is True,
        use UMAP to reduce the dimensionality of the latent space
    """

    if pca:
        varname = "PCA"
    elif tsne:
        varname = "tSNE"
    else:
        varname = "latent"

    grid = sns.jointplot(
        x=f"{varname}_{dim_1}",
        y=f"{varname}_{dim_2}",
        data=df_picked,
        kind="hex",
        color=(107 / 255, 174 / 255, 214 / 255),
        palette=palette,
        marginal_kws=dict(bins=100, fill=False),
        height=3.5,
    )
    grid.set_axis_labels(f"{varname}_{dim_1}", f"{varname}_{dim_2}")
    return grid


def main(args):
    """plot the distribution of frames from the source MD trajectory
    in the particle set for the given job"""

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

    # compute the precision for the picked particles
    _, df_picked = analysis.compute_precision(
        df_picked, df_truth, verbose=args.verbose
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
