"""Script to plot statistics from picking analyses and example overlays of
picked and truth particles on micrographs
"""

import os
from typing import Tuple, Dict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import mrcfile

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
        "-N",
        "--num-ugraphs",
        help="Number of micrographs to consider in analyses. Default 'all'",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--box-width",
        help="Full width of overlay boxes on images",
        type=float,
        default=50.0,
        required=False,
    )
    parser.add_argument(
        "--box-height",
        help="Full height of overlay boxes on images",
        type=float,
        default=50.0,
        required=False,
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
        "--plot-types",
        help="Types of analysis results to plot",
        type=str,
        nargs="+",
        choices=[
            "label_truth",
            "label_picked",
            "label_truth_and_picked",
            "precision",
            "boundary",
            "overlap",
        ],
    )
    parser.add_argument(
        "--verbose", help="increase output verbosity", action="store_true"
    )
    parser.add_argument(
        "--tqdm", help="show tqdm progress bar", action="store_true"
    )
    return parser


def get_name():
    return "plot_picking"


def _twoD_image_bboxs(
    particles_x: np.array,
    particles_y: np.array,
    box_width: float,
    box_height: float,
    verbose: bool = False,
) -> list[list[float]]:
    box_half_width = box_width / 2.0
    box_half_height = box_height / 2.0

    if verbose:
        print(
            "Using box half width: {} and half height: {}".format(
                box_half_width, box_half_height
            )
        )

    # now fill a list with x,y point positions of the particles
    twod_pos = []
    for x, y in zip(particles_x, particles_y):
        twod_pos.append([float(x), float(y)])

    # use this list to fill a list of boxes, each corresponding to a particle
    boxes = []
    for i in range(0, len(twod_pos)):
        temp_box = [
            twod_pos[i][0] - box_half_width,
            twod_pos[i][1] - box_half_height,
            twod_pos[i][0] + box_half_width,
            twod_pos[i][1] + box_half_height,
        ]
        boxes.append(temp_box)

    return boxes


def label_micrograph_truth(
    particles: pd.DataFrame,
    ugraph_index: int = 0,
    mrc_dir: str = "",
    box_width: int = 50,
    box_height: int = 50,
    verbose: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    # get the micrograph name
    ugraph_filename = np.unique(particles["ugraph_filename"])[ugraph_index]
    print(f"plotted index {ugraph_index}; micrograph: {ugraph_filename}")
    ugraph_path = os.path.join(mrc_dir, ugraph_filename)
    particles_ugraph = particles.groupby("ugraph_filename").get_group(
        ugraph_filename
    )

    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(ugraph_path) as mrc:
        data = mrc.data

        fig, ax = plt.subplots(figsize=[16, 16])
        ax.imshow(data[0], cmap="gray")
        fig.tight_layout()

        # Now that you've plotted the true central points of each particle,
        # also plot the boxes
        boxes = _twoD_image_bboxs(
            particles_ugraph["position_x"],
            particles_ugraph["position_y"],
            box_width,
            box_height,
            verbose,
        )
        if verbose:
            print(f"number of boxes: {len(boxes)}")

        for bbox in boxes:
            corner = [bbox[0], bbox[1]]
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner,
                width,
                height,
                linewidth=1,
                edgecolor=[0, 1, 0],
                facecolor="none",
            )
            ax.add_patch(rect)
        # green_patch = patches.Patch(color="green", label="Truth particles")
        # ax.legend(handles=[green_patch])
    return fig, ax


def label_micrograph_picked(
    particles: pd.DataFrame,
    metadata_filename: str | list[str],
    ugraph_index: int = 0,
    mrc_dir: str = "",
    box_width: int = 50,
    box_height: int = 50,
    verbose: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    # group the picked particles by metadata file
    if isinstance(metadata_filename, list):
        metadata_filename = metadata_filename[0]
    particles = particles.groupby("metadata_filename").get_group(
        metadata_filename
    )

    # get the micrograph name
    ugraph_filename = np.unique(particles["ugraph_filename"])[ugraph_index]
    print(f"plotted index {ugraph_index}; micrograph: {ugraph_filename}")
    ugraph_path = os.path.join(mrc_dir, ugraph_filename)
    particles_ugraph = particles.groupby("ugraph_filename").get_group(
        ugraph_filename
    )
    particles_ugraph.reset_index(inplace=True)

    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(ugraph_path) as mrc:
        data = mrc.data

        fig, ax = plt.subplots(figsize=[16, 16])
        ax.imshow(data[0], cmap="gray")
        fig.tight_layout()

        # Now that you've plotted the true central points of each particle,
        # also plot the boxes
        boxes = _twoD_image_bboxs(
            particles_ugraph["position_x"],
            particles_ugraph["position_y"],
            box_width,
            box_height,
            verbose,
        )
        if verbose:
            print(f"number of boxes: {len(boxes)}")

        for i, bbox in enumerate(boxes):
            corner = [bbox[0], bbox[1]]
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            if "TP" in particles_ugraph.columns:
                facecolor = "lime" if particles_ugraph["TP"][i] else "red"
                alpha = 0.6
            else:
                facecolor = "none"
                alpha = 1
            rect = patches.Rectangle(
                corner,
                width,
                height,
                linewidth=1,
                edgecolor=[1, 0, 0],
                facecolor=facecolor,
                alpha=alpha,
            )
            ax.add_patch(rect)
        # red_patch = patches.Patch(color="red", label="Picked particles")
        # ax.legend(handles=[red_patch])
    return fig, ax


def label_micrograph_truth_and_picked(
    picked_particles: pd.DataFrame,
    metadata_filename: str | list[str],
    truth_particles: pd.DataFrame,
    ugraph_index: int = 0,
    mrc_dir: str = "",
    box_width: int = 50,
    box_height: int = 50,
    verbose: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    # group the picked particles by metadata file
    if isinstance(metadata_filename, list):
        metadata_filename = metadata_filename[0]
    picked_particles = picked_particles.groupby("metadata_filename").get_group(
        metadata_filename
    )
    # get the micrograph name
    ugraph_filename = np.unique(truth_particles["ugraph_filename"])[
        ugraph_index
    ]
    print(f"plotted index {ugraph_index}; micrograph: {ugraph_filename}")
    ugraph_path = os.path.join(mrc_dir, ugraph_filename)
    truth_particles_ugraph = truth_particles.groupby(
        "ugraph_filename"
    ).get_group(ugraph_filename)
    picked_particles_ugraph = picked_particles.groupby(
        "ugraph_filename"
    ).get_group(ugraph_filename)

    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(ugraph_path) as mrc:
        data = mrc.data

        fig, ax = plt.subplots(figsize=[16, 16])
        ax.imshow(data[0], cmap="gray")
        fig.tight_layout()

        # Now that you've plotted the true central points of each particle,
        # also plot the boxes
        boxes = _twoD_image_bboxs(
            picked_particles_ugraph["position_x"],
            picked_particles_ugraph["position_y"],
            box_width,
            box_height,
            verbose,
        )
        if verbose:
            print(f"number of boxes: {len(boxes)}")

        for bbox in boxes:
            corner = [bbox[0], bbox[1]]
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner,
                width,
                height,
                linewidth=1,
                edgecolor=[1, 0, 0],
                facecolor="none",
            )
            ax.add_patch(rect)
        # red_patch = patches.Patch(color="red", label="Picked particles")

        boxes = _twoD_image_bboxs(
            truth_particles_ugraph["position_x"],
            truth_particles_ugraph["position_y"],
            box_width,
            box_height,
            verbose,
        )
        if verbose:
            print(f"number of boxes: {len(boxes)}")

        for bbox in boxes:
            corner = [bbox[0], bbox[1]]
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner,
                width,
                height,
                linewidth=1,
                edgecolor=[0, 1, 0],
                facecolor="none",
            )
            ax.add_patch(rect)
        # green_patch = patches.Patch(color="green", label="Truth particles")
        # ax.legend(handles=[red_patch, green_patch])
    return fig, ax


def plot_precision(
    df_precision: pd.DataFrame,
    jobtypes: Dict[str, str],
    order: list[str] | None = None,
):
    """Precision is calculated as follows:
    precision = TP / (TP + FP)
    where TP is the number of true positives, which is stored in the picked
    particles dataframe
    and FP is the number of false positives, which can be extracted from
    the truth_particles dataframe by looking at the number of truth
    particles that have 0 multiplicity
    """

    # plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        x="metadata_filename",
        y="precision",
        data=df_precision,
        ax=ax,
        fliersize=0,
        palette="Blues",
        order=order,
    )
    sns.stripplot(
        x="metadata_filename",
        y="precision",
        data=df_precision,
        ax=ax,
        hue="defocus",
        alpha=0.7,
        palette="RdYlBu",
        order=order,
    )
    # change the xticklabels to the jobtype
    if order is None:
        ax.set_xticklabels(
            [
                jobtypes[metadata_filename]
                for metadata_filename in df_precision[
                    "metadata_filename"
                ].unique()
            ]
        )
    else:
        ax.set_xticklabels(
            [jobtypes[metadata_filename] for metadata_filename in order]
        )
    # remove legend
    ax.legend().remove()
    # add colorbar
    sm = plt.cm.ScalarMappable(
        cmap="RdYlBu",
        norm=plt.Normalize(
            vmin=df_precision["defocus"].min(),
            vmax=df_precision["defocus"].max(),
        ),
    )
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("defocus (Å)", rotation=270, labelpad=20)
    # add labels
    ax.set_xlabel("job type")
    ax.set_ylabel("precision")
    ax.set_title("Precision for different job types")
    # rotate xtiklabels 45 degrees
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )
    fig.tight_layout()
    return fig, ax


def plot_recall(
    df_precision: pd.DataFrame,
    jobtypes: Dict[str, str],
    order: list[str] | None = None,
):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        x="metadata_filename",
        y="recall",
        data=df_precision,
        ax=ax,
        fliersize=0,
        palette="Blues",
        order=order,
    )
    sns.stripplot(
        x="metadata_filename",
        y="recall",
        data=df_precision,
        ax=ax,
        hue="defocus",
        alpha=0.7,
        palette="RdYlBu",
        order=order,
    )
    # change the xticklabels to the jobtype
    if order is None:
        ax.set_xticklabels(
            [
                jobtypes[metadata_filename]
                for metadata_filename in df_precision[
                    "metadata_filename"
                ].unique()
            ]
        )
    else:
        ax.set_xticklabels(
            [jobtypes[metadata_filename] for metadata_filename in order]
        )
    # remove legend
    ax.legend().remove()
    # add colorbar
    sm = plt.cm.ScalarMappable(
        cmap="RdYlBu",
        norm=plt.Normalize(
            vmin=df_precision["defocus"].min(),
            vmax=df_precision["defocus"].max(),
        ),
    )
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("defocus (Å)", rotation=270, labelpad=20)
    # add labels
    ax.set_xlabel("job type")
    ax.set_ylabel("recall")
    ax.set_title("Recall for different job types")
    # rotate xtiklabels 45 degrees
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )
    fig.tight_layout()
    return fig, ax


def plot_precision_and_recall(
    df_precision: pd.DataFrame,
    jobtypes: Dict[str, str],
    order: list[str] | None = None,
):
    # get all column names
    col_names = df_precision.columns.tolist()
    # remove the precsion and recall columns
    col_names.remove("precision")
    col_names.remove("recall")
    df = df_precision.melt(
        id_vars=col_names, var_name="variable", value_name="value"
    )

    plt.rcParams["font.size"] = 20
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.boxplot(
        x="metadata_filename",
        y="value",
        data=df,
        ax=ax,
        fliersize=0,
        palette="RdYlBu",
        hue="variable",
        order=order,
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    # change the xtix labels to the jobtypes
    ax.set_xticklabels(
        [
            jobtypes[meta_file]
            for meta_file in np.unique(df["metadata_filename"])
            if meta_file in jobtypes.keys()
        ]
    )
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )
    # add legend below axis
    ax.legend().set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=1, bbox_to_anchor=(1.1, 0.85)
    )
    fig.tight_layout()
    return fig, ax


def plot_f1_score(
    df_precision: pd.DataFrame,
    jobtypes: Dict[str, str],
    order: list[str] | None = None,
):
    # compute f1 score from the precision and recall
    df_precision["f1_score"] = (
        2
        * (df_precision["precision"] * df_precision["recall"])
        / (df_precision["precision"] + df_precision["recall"])
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(
        x="metadata_filename",
        y="f1_score",
        data=df_precision,
        ax=ax,
        fliersize=0,
        palette="Blues",
        order=order,
    )

    sns.stripplot(
        x="metadata_filename",
        y="f1_score",
        data=df_precision,
        ax=ax,
        hue="defocus",
        alpha=0.7,
        palette="RdYlBu",
        order=order,
    )

    # change the xticklabels to the jobtype
    if order is None:
        ax.set_xticklabels(
            [
                jobtypes[metadata_filename]
                for metadata_filename in df_precision[
                    "metadata_filename"
                ].unique()
            ]
        )
    else:
        ax.set_xticklabels(
            [jobtypes[metadata_filename] for metadata_filename in order]
        )
    # remove legend
    ax.legend().remove()
    # add colorbar
    sm = plt.cm.ScalarMappable(
        cmap="RdYlBu",
        norm=plt.Normalize(
            vmin=df_precision["defocus"].min(),
            vmax=df_precision["defocus"].max(),
        ),
    )
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("defocus (Å)", rotation=270, labelpad=20)
    # add labels
    ax.set_xlabel("job type")
    ax.set_ylabel("f1 score")
    ax.set_title("F1 score for different job types")
    # rotate xtiklabels 45 degrees
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor"
    )
    fig.tight_layout()
    return fig, ax


def plot_boundary_investigation(
    df_truth: pd.DataFrame,
    df_picked: pd.DataFrame,
    metadata_filename: str,
    bin_width: int = 100,
    axis: str = "x",
):
    if isinstance(metadata_filename, list):
        metadata_filename = metadata_filename[0]

    particles_per_ugraph = (
        df_truth.groupby("ugraph_filename")
        .size()
        .reset_index(name="particles_per_ugraph")
    )
    avg_particles_per_ugraph = particles_per_ugraph[
        "particles_per_ugraph"
    ].mean()
    num_ugraphs = len(particles_per_ugraph)
    if axis != "z":
        particles_per_bin = (
            avg_particles_per_ugraph
            * (bin_width / df_picked["ugraph_shape"][0][0])
            * num_ugraphs
        )
        bins = np.arange(0, df_picked["ugraph_shape"][0][0], bin_width)
    else:
        particles_per_bin = (
            avg_particles_per_ugraph
            * (bin_width / df_truth["ice_thickness"][0])
            * num_ugraphs
        )
        bins = np.arange(0, df_truth["ice_thickness"][0], bin_width)

    fig, ax = plt.subplots()
    if axis != "z":
        sns.histplot(
            x=f"position_{axis}",
            data=df_picked.groupby("metadata_filename").get_group(
                metadata_filename
            ),
            stat="count",
            bins=bins,
            color="red",
            label="picked",
            fill=False,
            ax=ax,
        )
    sns.histplot(
        x=f"position_{axis}",
        data=df_truth,
        stat="count",
        bins=bins,
        color="blue",
        label="truth",
        fill=False,
        ax=ax,
    )
    # plot a line at the expected number of particles per bin
    if axis != "z":
        ax.hlines(
            [particles_per_bin],
            0.0,
            df_picked["ugraph_shape"][0][0],
            colors=["black"],
            linestyles=["dashed"],
            label="expected",
        )
    else:
        ax.hlines(
            [particles_per_bin],
            0.0,
            df_truth["ice_thickness"][0],
            colors=["black"],
            linestyles=["dashed"],
            label="expected",
        )
    ax.set_xlabel(f"{axis} position (Angstroms)")
    ax.set_ylabel("Count")
    ax.set_title(os.path.basename(metadata_filename))
    return fig, ax


def plot_overlap_investigation(
    df_overlap: pd.DataFrame,
    metadata_filename: str | list[str] | None = None,
    jobtypes: Dict[str, str] | None = None,
):
    if metadata_filename is None:
        # plot all metadata files in one plot
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.lineplot(
            x="radius",
            y="neighbours_truth",
            data=df_overlap,
            ax=ax,
            marker="o",
            errorbar="sd",
            hue="metadata_filename",
            markeredgecolor="black",
            palette="Dark2",
        )
        sns.lineplot(
            x="radius",
            y="neighbours_picked",
            data=df_overlap,
            ax=ax,
            marker="x",
            errorbar="sd",
            hue="metadata_filename",
            markeredgecolor="black",
            palette="Dark2",
        )
        ax.grid(which="both")
        # only show legend for the first half of the lines
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles[: len(handles) // 2],
            labels=labels[: len(labels) // 2],
        )
        ax.set_ylabel("# Overlaps with a truth particle")
        fig.tight_layout()
        return fig, ax

    else:
        if isinstance(metadata_filename, list):
            metadata_filename = metadata_filename[0]
        print(f"plotting overlap for {metadata_filename}")
        # make a plot for each metadata file
        fig, ax = plt.subplots(figsize=(18, 8))
        sns.lineplot(
            x="radius",
            y="neighbours_truth",
            data=df_overlap.groupby("metadata_filename").get_group(
                metadata_filename
            ),
            ax=ax,
            marker="o",
            errorbar="sd",
            markeredgecolor="black",
            color="blue",
            label="truth",
        )
        sns.lineplot(
            x="radius",
            y="neighbours_picked",
            data=df_overlap.groupby("metadata_filename").get_group(
                metadata_filename
            ),
            ax=ax,
            marker="x",
            errorbar="sd",
            markeredgecolor="black",
            color="red",
            label="picked",
        )
        ax.grid(which="both")
        ax.set_ylabel("# which overlap with a truth particle")
        fig.tight_layout()
        return fig, ax


def main(args):
    """This analysis tool makes plots of the picked and ground-truth particles
    in a number of micrographs. It then makes quantitative comparisons between
    the two.
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

    # get a dictionary of the jobtypes
    if args.jobtypes is not None:
        jobtypes = {
            meta_file: jobtype
            for meta_file, jobtype in zip(args.meta_file, args.jobtypes)
        }
    else:
        jobtypes = {
            meta_file: os.path.basename(meta_file).split(".")[0]
            for meta_file in args.meta_file
        }

    for plot_type in args.plot_types:
        if plot_type == "label_truth":  # plot the ground-truth particles
            for ugraph_index, ugraph_filename in enumerate(
                np.unique(df_truth["ugraph_filename"])[: args.num_ugraphs]
            ):
                print(
                    "Plotting truth particles in micrograph {}".format(
                        ugraph_filename,
                    )
                )
                fig, ax = label_micrograph_truth(
                    df_truth,
                    ugraph_index,
                    args.mrc_dir,
                    box_width=args.box_width,
                    box_height=args.box_height,
                    verbose=args.verbose,
                )
                # remove axis ticks
                ax.set_xticks([])
                ax.set_yticks([])
                outfilename = os.path.join(
                    args.plot_dir,
                    "{}_truth.png".format(
                        ugraph_filename.strip(".mrc"),
                    ),
                )
                fig.savefig(outfilename, dpi=600, bbox_inches="tight")
                fig.savefig(
                    outfilename.replace(".png", ".pdf"), bbox_inches="tight"
                )
                fig.clf()

        if plot_type == "label_picked":  # plot the picked particles
            for meta_file in args.meta_file:
                meta_basename = os.path.basename(meta_file)
                for ugraph_index, ugraph_filename in enumerate(
                    np.unique(df_picked["ugraph_filename"])[: args.num_ugraphs]
                ):
                    print(
                        "Plotting picked particles in micrograph {}, \
                        from metadata file {}".format(
                            ugraph_filename, meta_basename
                        )
                    )
                    fig, ax = label_micrograph_picked(
                        df_picked,
                        meta_file,
                        ugraph_index,
                        args.mrc_dir,
                        box_width=args.box_width,
                        box_height=args.box_height,
                        verbose=args.verbose,
                    )
                    # remove axis ticks
                    ax.set_xticks([])
                    ax.set_yticks([])
                    outfilename = os.path.join(
                        args.plot_dir,
                        "{}_{}_picked.png".format(
                            ugraph_filename.strip(".mrc"),
                            meta_basename.split(".")[0],
                        ),
                    )
                    # remove axis ticks
                    ax.set_xticks([])
                    ax.set_yticks([])
                    fig.savefig(outfilename, dpi=600, bbox_inches="tight")
                    fig.savefig(
                        outfilename.replace(".png", ".pdf"),
                        bbox_inches="tight",
                    )
                    fig.clf()

        if plot_type == "label_truth_and_picked":
            for meta_file in args.meta_file:
                meta_basename = os.path.basename(meta_file)
                for ugraph_index, ugraph_filename in enumerate(
                    np.unique(df_picked["ugraph_filename"])[: args.num_ugraphs]
                ):
                    print(
                        "Plotting picked and truth particles in micrograph"
                        " {}, from metadata file {}".format(
                            ugraph_filename,
                            meta_basename,
                        )
                    )
                    fig, ax = label_micrograph_truth_and_picked(
                        df_picked,
                        meta_file,
                        df_truth,
                        ugraph_index,
                        args.mrc_dir,
                        box_width=args.box_width,
                        box_height=args.box_height,
                        verbose=args.verbose,
                    )
                    # remove axis ticks
                    ax.set_xticks([])
                    ax.set_yticks([])
                    outfilename = os.path.join(
                        args.plot_dir,
                        "{}_{}_truth_and_picked.png".format(
                            ugraph_filename.strip(".mrc"),
                            meta_basename.split(".")[0],
                        ),
                    )
                    fig.savefig(outfilename, dpi=600, bbox_inches="tight")
                    fig.savefig(
                        outfilename.replace(".png", ".pdf"),
                        bbox_inches="tight",
                    )
                    fig.clf()

        if plot_type == "precision":
            # first need to compute the precision statistics
            df_precision, _ = analysis.compute_precision(
                df_picked, df_truth, verbose=args.verbose
            )

            print("plotting precision...")
            fig, ax = plot_precision(df_precision, jobtypes, args.meta_file)
            outfilename = os.path.join(args.plot_dir, "precision.png")
            fig.savefig(outfilename, dpi=600, bbox_inches="tight")
            fig.savefig(
                outfilename.replace(".png", ".pdf"), bbox_inches="tight"
            )
            fig.clf()

            print("plotting recall...")
            fig, ax = plot_recall(df_precision, jobtypes, args.meta_file)
            outfilename = os.path.join(args.plot_dir, "recall.png")
            fig.savefig(outfilename, dpi=600, bbox_inches="tight")
            fig.savefig(
                outfilename.replace(".png", ".pdf"), bbox_inches="tight"
            )
            fig.clf()

            print("plotting precsion and recall in one plot...")
            fig, ax = plot_precision_and_recall(
                df_precision, jobtypes, args.meta_file
            )
            outfilename = os.path.join(
                args.plot_dir, "precision_and_recall.png"
            )
            fig.savefig(outfilename, dpi=600, bbox_inches="tight")
            fig.savefig(
                outfilename.replace(".png", ".pdf"), bbox_inches="tight"
            )
            fig.clf()

            print("plotting F1 score...")
            fig, ax = plot_f1_score(df_precision, jobtypes, args.meta_file)
            outfilename = os.path.join(args.plot_dir, "f1_score.png")
            fig.savefig(outfilename, dpi=600, bbox_inches="tight")
            fig.savefig(
                outfilename.replace(".png", ".pdf"), bbox_inches="tight"
            )
            fig.clf()

        if plot_type == "boundary":
            bin_width = [100, 100, 10]  # bin width for x, y, z
            axis = ["x", "y", "z"]

            for meta_file in args.meta_file:
                meta_basename = os.path.basename(meta_file)
                print(f"plotting boundary for metadata file {meta_file}")
                for a, bnwdth in zip(axis, bin_width):
                    fig, ax = plot_boundary_investigation(
                        df_truth, df_picked, meta_file, bnwdth, a
                    )
                    outfilename = os.path.join(
                        args.plot_dir,
                        f"{meta_basename.split('.')[0]}_boundary_{a}.png",
                    )
                    fig.savefig(outfilename, dpi=600, bbox_inches="tight")
                    fig.savefig(
                        outfilename.replace(".png", ".pdf"),
                        bbox_inches="tight",
                    )
                    fig.clf()

        if plot_type == "overlap":
            df_overlap = analysis.compute_overlap(
                df_picked, df_truth, verbose=args.verbose
            )
            print("plotting overlap...")
            for meta_file in args.meta_file:
                meta_basename = os.path.basename(meta_file)
                fig, ax = plot_overlap_investigation(df_overlap, meta_file)
                outfilename = os.path.join(
                    args.plot_dir, f"{meta_basename.split('.')[0]}_overlap.png"
                )
                fig.savefig(outfilename)
                fig.clf()

            fig, ax = plot_overlap_investigation(
                df_overlap, None, jobtypes
            )  # plot all
            outfilename = os.path.join(args.plot_dir, "overlap.png")
            fig.savefig(outfilename, dpi=600, bbox_inches="tight")
            fig.savefig(
                outfilename.replace(".png", ".pdf"), bbox_inches="tight"
            )
            fig.clf()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
