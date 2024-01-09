"""Plotting functions for analysis of heterogeneous reconstructions.
Maybe be deprecated soon, with plotting functions moved elsewhere.

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


import matplotlib.pyplot as plt
import seaborn as sns


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
            x=df_picked["{}_{}".format(varname, dim_1)],
            y=df_picked["{}_{}".format(varname, dim_2)],
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
            x=df_picked["{}_{}".format(varname, dim_1)],
            y=df_picked["{}_{}".format(varname, dim_2)],
            data=df_picked,
            hue=color_by,
            palette=palette,
            ax=ax,
            s=5,
        )
        ax.set_xlabel(f"{varname}_{dim_1}")
        ax.set_ylabel(f"{varname}_{dim_2}")
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
        # fig.colorbar()
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
        x=df_picked["{}_{}".format(varname, dim_1)],
        y=df_picked["{}_{}".format(varname, dim_2)],
        data=df_picked,
        kind="hex",
        color=(107 / 255, 174 / 255, 214 / 255),
        palette=palette,
        marginal_kws=dict(bins=100, fill=False),
        height=3.5,
    )
    grid.set_axis_labels(f"{varname}_{dim_1}", f"{varname}_{dim_2}")
    return grid
