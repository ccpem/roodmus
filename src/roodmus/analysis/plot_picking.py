"""Plot statistics from picking analyses and overlays of
picked and truth particles on micrographs.

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

import os
from typing import Tuple, Dict, List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import mrcfile

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
        "-N",
        "--num_ugraphs",
        help="Number of micrographs to consider in analyses. Default 'all'",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--box_width",
        help="Full width of overlay boxes on images",
        type=float,
        default=150.0,
        required=False,
    )
    parser.add_argument(
        "--box_height",
        help="Full height of overlay boxes on images",
        type=float,
        default=150.0,
        required=False,
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
        "--plot_types",
        help="Types of analysis results to plot",
        type=str,
        nargs="+",
        choices=[
            "label_truth",
            "label_picked",
            "label_truth_and_picked",
            "label_matched_and_unmatched",
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
    return "plot_picking"


class plotLabelTruth(plotDataFrame):
    def __init__(
        self,
        mrc_dir: str,
        plot_data: Dict[str, Dict[str, pd.DataFrame]] | None = None,
        plot_dir: str = "",
        num_ugraphs: int | None = None,
        box_width: float = 200,
        box_height: float = 200,
        dpi: int = 300,
        pdf: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(plot_data)

        if plot_data:
            self.plot_data = plot_data

        self.mrc_dir = mrc_dir
        self.plot_dir = plot_dir
        self.num_ugraphs = num_ugraphs
        self.box_width = box_width
        self.box_height = box_height
        self.dpi = dpi
        self.pdf = pdf
        self.verbose = verbose

    def setup_plot_data(
        self,
        df_truth: pd.DataFrame,
    ):
        self.plot_data = {"label_truth": {"df_truth": df_truth}}

    def setup_plot_data_empty(
        self,
    ):
        self.plot_data = {"label_truth": {"df_truth": None}}

    def make_and_save_plots(
        self,
        overwrite_data: bool = False,
    ):
        self.save_dataframes(self.plot_dir, overwrite_data)

        if isinstance(
            self.plot_data["label_truth"]["df_truth"],
            pd.DataFrame,
        ):
            for ugraph_index, ugraph_filename in enumerate(
                np.unique(
                    self.plot_data["label_truth"]["df_truth"][
                        "ugraph_filename"
                    ]
                )[: self.num_ugraphs]
            ):
                fig, ax, outfilename = plot_label_truth(
                    self.plot_data["label_truth"]["df_truth"],
                    ugraph_index,
                    ugraph_filename,
                    self.box_width,
                    self.box_height,
                    self.mrc_dir,
                    self.verbose,
                    self.plot_dir,
                )
                self._save_plot(fig, ax, outfilename)

        else:
            raise TypeError(
                'self.plot_data["label_truth"]["df_truth"] is not'
                " a pd.DataFrame"
            )

    def _save_plot(self, fig, ax, outfilename: str):
        # save the plot
        fig.savefig(outfilename, dpi=self.dpi, bbox_inches="tight")
        if self.pdf:
            fig.savefig(
                outfilename.replace(".png", ".pdf"),
                bbox_inches="tight",
            )
        fig.clf()


def plot_label_truth(
    df_truth: pd.DataFrame,
    ugraph_index: int,
    ugraph_filename: str,
    box_width: float,
    box_height: float,
    mrc_dir: str,
    verbose: bool,
    plot_dir: str = "",
):
    print(
        "Plotting truth particles in micrograph {}".format(
            ugraph_filename,
        )
    )
    fig, ax = labelMicrograph.label_micrograph_truth(
        df_truth,
        ugraph_index,
        mrc_dir,
        box_width=box_width,
        box_height=box_height,
        verbose=verbose,
    )
    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    outfilename = os.path.join(
        plot_dir,
        "{}_truth.png".format(
            ugraph_filename.strip(".mrc"),
        ),
    )
    return fig, ax, outfilename


class plotLabelPicked(plotDataFrame):
    def __init__(
        self,
        mrc_dir: str,
        plot_data: dict[str, dict[str, pd.DataFrame]] | None = None,
        plot_dir: str = "",
        num_ugraphs: int | None = None,
        box_width: float = 200,
        box_height: float = 200,
        dpi: int = 300,
        pdf: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(plot_data)

        if plot_data:
            self.plot_data = plot_data

        self.mrc_dir = mrc_dir
        self.plot_dir = plot_dir
        self.num_ugraphs = num_ugraphs
        self.box_width = box_width
        self.box_height = box_height
        self.dpi = dpi
        self.pdf = pdf
        self.verbose = verbose

    def setup_plot_data(
        self,
        df_picked: pd.DataFrame,
    ):
        self.plot_data = {"label_picked": {"df_picked": df_picked}}

    def setup_plot_data_empty(
        self,
    ):
        self.plot_data = {"label_picked": {"df_picked": None}}

    def make_and_save_plots(
        self,
        overwrite_data: bool = False,
    ):
        self.save_dataframes(self.plot_dir, overwrite_data)

        if isinstance(
            self.plot_data["label_picked"]["df_picked"],
            pd.DataFrame,
        ):
            for meta_file in self.plot_data["label_picked"]["df_picked"][
                "metadata_filename"
            ].unique():
                for ugraph_index, ugraph_filename in enumerate(
                    np.unique(
                        self.plot_data["label_picked"]["df_picked"][
                            "ugraph_filename"
                        ]
                    )[: self.num_ugraphs]
                ):
                    fig, ax, outfilename = plot_label_picked(
                        self.plot_data["label_picked"]["df_picked"],
                        meta_file,
                        ugraph_filename,
                        ugraph_index,
                        self.box_width,
                        self.box_height,
                        self.mrc_dir,
                        self.verbose,
                        self.plot_dir,
                    )

                    self._save_plot(fig, ax, outfilename)
        else:
            raise TypeError(
                'self.plot_data["label_picked"]["df_picked"] is not'
                " a pd.DataFrame"
            )

    def _save_plot(self, fig, ax, outfilename: str):
        # save the plot
        fig.savefig(outfilename, dpi=self.dpi, bbox_inches="tight")
        if self.pdf:
            fig.savefig(
                outfilename.replace(".png", ".pdf"),
                bbox_inches="tight",
            )
        fig.clf()


def plot_label_picked(
    df_picked: pd.DataFrame,
    meta_file: str,
    ugraph_filename: str,
    ugraph_index: int,
    box_width: float,
    box_height: float,
    mrc_dir: str,
    verbose: bool,
    plot_dir: str = "",
):
    meta_basename = os.path.basename(meta_file)
    print(
        "Plotting picked particles in micrograph {}, \
        from metadata file {}".format(
            ugraph_filename, meta_basename
        )
    )
    fig, ax = labelMicrograph.label_micrograph_picked(
        df_picked,
        meta_file,
        ugraph_index,
        mrc_dir,
        box_width=box_width,
        box_height=box_height,
        verbose=verbose,
    )

    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    outfilename = os.path.join(
        plot_dir,
        "{}_{}_picked.png".format(
            ugraph_filename.strip(".mrc"),
            meta_basename.split(".")[0],
        ),
    )
    return fig, ax, outfilename


class plotLabelTruthAndPicked(plotDataFrame):
    def __init__(
        self,
        mrc_dir: str,
        plot_data: dict[str, dict[str, pd.DataFrame]] | None = None,
        plot_dir: str = "",
        num_ugraphs: int | None = None,
        box_width: float = 200,
        box_height: float = 200,
        dpi: int = 300,
        pdf: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(plot_data)

        if plot_data:
            self.plot_data = plot_data

        self.mrc_dir = mrc_dir
        self.plot_dir = plot_dir
        self.num_ugraphs = num_ugraphs
        self.box_width = box_width
        self.box_height = box_height
        self.dpi = dpi
        self.pdf = pdf
        self.verbose = verbose

    def setup_plot_data(
        self,
        df_truth: pd.DataFrame,
        df_picked: pd.DataFrame,
    ):
        self.plot_data = {"label_truth_and_picked": {}}
        self.plot_data["label_truth_and_picked"]["df_truth"] = df_truth
        self.plot_data["label_truth_and_picked"]["df_picked"] = df_picked

    def setup_plot_data_empty(
        self,
    ):
        self.plot_data = {"label_truth_and_picked": {}}
        self.plot_data["label_truth_and_picked"]["df_truth"] = None
        self.plot_data["label_truth_and_picked"]["df_picked"] = None

    def make_and_save_plots(
        self,
        overwrite_data: bool = False,
    ):
        self.save_dataframes(self.plot_dir, overwrite_data)

        if isinstance(
            self.plot_data["label_truth_and_picked"]["df_truth"],
            pd.DataFrame,
        ) and isinstance(
            self.plot_data["label_truth_and_picked"]["df_picked"],
            pd.DataFrame,
        ):
            for meta_file in self.plot_data["label_truth_and_picked"][
                "df_picked"
            ]["metadata_filename"].unique():
                for ugraph_index, ugraph_filename in enumerate(
                    np.unique(
                        self.plot_data["label_truth_and_picked"]["df_picked"][
                            "ugraph_filename"
                        ]
                    )[: self.num_ugraphs]
                ):
                    fig, ax, outfilename = plot_label_picked_and_truth(
                        self.plot_data["label_truth_and_picked"]["df_truth"],
                        self.plot_data["label_truth_and_picked"]["df_picked"],
                        meta_file,
                        ugraph_filename,
                        ugraph_index,
                        self.box_width,
                        self.box_height,
                        self.mrc_dir,
                        self.verbose,
                        self.plot_dir,
                    )

                    self._save_plot(fig, ax, outfilename)

        else:
            raise TypeError(
                'One or more of self.plot_data["label_truth_and_picked"]'
                '["df_truth"]'
                ' and self.plot_data["label_truth_and_picked"]'
                '["df_picked"] is not'
                " a pd.DataFrame"
            )

    def _save_plot(self, fig, ax, outfilename: str):
        # save the plot
        fig.savefig(outfilename, dpi=self.dpi, bbox_inches="tight")
        if self.pdf:
            fig.savefig(
                outfilename.replace(".png", ".pdf"),
                bbox_inches="tight",
            )
        fig.clf()


def plot_label_picked_and_truth(
    df_truth: pd.DataFrame,
    df_picked: pd.DataFrame,
    meta_file: str,
    ugraph_filename: str,
    ugraph_index: int,
    box_width: float,
    box_height: float,
    mrc_dir: str,
    verbose: bool,
    plot_dir: str = "",
):
    meta_basename = os.path.basename(meta_file)
    print(
        "Plotting picked and truth particles in micrograph"
        " {}, from metadata file {}".format(
            ugraph_filename,
            meta_basename,
        )
    )
    fig, ax = labelMicrograph.label_micrograph_truth_and_picked(
        df_picked,
        meta_file,
        df_truth,
        ugraph_index,
        mrc_dir,
        box_width=box_width,
        box_height=box_height,
        verbose=verbose,
    )
    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    outfilename = os.path.join(
        plot_dir,
        "{}_{}_truth_and_picked.png".format(
            ugraph_filename.strip(".mrc"),
            meta_basename.split(".")[0],
        ),
    )
    return fig, ax, outfilename


class plotMatchedAndUnmatched(plotDataFrame):
    """
    for matching picked particles to truth particles
    and plotting:
    matched_picked_particles [0,0,1] which is the box color
    matched_truth_particles [0,0,0.5]
    umatched_picked_particles [1,1,0]
    unmatched_truth_particles [1,1,0.5]

    matched_picked_particles [0,0,1] and
    unmatched_picked_particles [1,1,0]

    matched_truth_particles [0,0,0.5] and
    unmatched_truth_particles [1,1,0.5]

    note - the below compare to original truth particles or
    picked particles:
    unmatched_picked particles [1,1,0] and truth_particles [0,1,0]
    picked_particles [1,0,0] and unmatched_truth_particles [1,1,0.5]
    """

    def __init__(
        self,
        mrc_dir: str,
        plot_data: dict[str, dict[str, pd.DataFrame]] | None = None,
        plot_dir: str = "",
        num_ugraphs: int | None = None,
        box_width: float = 200,
        box_height: float = 200,
        dpi: int = 300,
        pdf: bool = False,
        verbose: bool = False,
        analysis: load_data | None = None,
    ) -> None:
        super().__init__(plot_data)

        self.setup_plot_data(analysis)
        if plot_data:
            self.plot_data = plot_data

        self.mrc_dir = mrc_dir
        self.plot_dir = plot_dir
        self.num_ugraphs = num_ugraphs
        self.box_width = box_width
        self.box_height = box_height
        self.dpi = dpi
        self.pdf = pdf
        self.verbose = verbose

    def setup_plot_data(
        self,
        analysis: load_data | None,
    ):
        """
        Provide data loaded into the load_data class from metadata
        file.
        """
        # provide load_data class (analysis) to compute matching
        if analysis is not None:
            """
            First, do the matching for all the micrographs
            in the metadata file if they are not provided
            matched_picked_df = mp_df
            matched_truth_df = mt_df
            unmatched_picked_df = up_df
            unmatched_truth_df = ut_df
            """
            self.plot_data = {"label_matched_and_unmatched": {}}
            self.plot_data["label_matched_and_unmatched"][
                "df_truth"
            ] = pd.DataFrame(analysis.results_truth)
            self.plot_data["label_matched_and_unmatched"][
                "df_picked"
            ] = pd.DataFrame(analysis.results_picking)

            if isinstance(
                self.plot_data["label_matched_and_unmatched"]["df_truth"],
                pd.DataFrame,
            ) and isinstance(
                self.plot_data["label_matched_and_unmatched"]["df_picked"],
                pd.DataFrame,
            ):
                for meta_file in self.plot_data["label_matched_and_unmatched"][
                    "df_picked"
                ]["metadata_filename"].unique():
                    meta_basename = os.path.basename(meta_file)
                    self.plot_data[
                        "label_matched_and_unmatched_{}".format(
                            meta_basename.split(".")[0]
                        )
                    ] = {}
                    (
                        self.plot_data[
                            "label_matched_and_unmatched_{}".format(
                                meta_basename.split(".")[0]
                            )
                        ]["df_mp"],
                        self.plot_data[
                            "label_matched_and_unmatched_{}".format(
                                meta_basename.split(".")[0]
                            )
                        ]["df_mt"],
                        self.plot_data[
                            "label_matched_and_unmatched_{}".format(
                                meta_basename.split(".")[0]
                            )
                        ]["df_up"],
                        self.plot_data[
                            "label_matched_and_unmatched_{}".format(
                                meta_basename.split(".")[0]
                            )
                        ]["df_ut"],
                    ) = analysis._match_particles(
                        meta_file,
                        self.plot_data["label_matched_and_unmatched"][
                            "df_picked"
                        ],
                        self.plot_data["label_matched_and_unmatched"][
                            "df_truth"
                        ],
                        verbose=self.verbose,
                    )
            else:
                raise TypeError(
                    'Either self.plot_data["label_matched_and_unmatched"]'
                    '["df_picked"]'
                    ' or self.plot_data["label_matched_and_unmatched"]'
                    '["df_truth"] is not a pd.DataFrame when extracting'
                    " from analysis (load_data() class) object"
                )

    """
    def setup_plot_data_empty(
        self,
        metadata_file_basenames: list[str] | None = None,
    ):
        # Provide data loaded into the load_data class from metadata file

        self.plot_data = {"label_matched_and_unmatched": {}}
        self.plot_data["label_matched_and_unmatched"]["df_truth"] = None
        self.plot_data["label_matched_and_unmatched"]["df_picked"] = None

        if metadata_file_basenames:
            for metadata_file in metadata_file_basenames:
                self.plot_data[
                    "label_matched_and_unmatched_{}".format(metadata_file)
                ]["df_mp"] = None
                self.plot_data[
                    "label_matched_and_unmatched_{}".format(metadata_file)
                ]["df_mt"] = None
                self.plot_data[
                    "label_matched_and_unmatched_{}".format(metadata_file)
                ]["df_up"] = None
                self.plot_data[
                    "label_matched_and_unmatched_{}".format(metadata_file)
                ]["df_ut"] = None
    """

    def make_and_save_plots(
        self,
        analysis: load_data | None = None,
        overwrite_data: bool = False,
    ):
        # compute the matching if the matched dfs are not provided
        # but the load_data() class is provided
        self.setup_plot_data(analysis=analysis)

        # df_truth and df_picked contain data for
        # all metadata files
        # df_mp df_mt df_ut and df_up contain data
        # for a single metadata file only
        if isinstance(
            self.plot_data["label_matched_and_unmatched"]["df_picked"],
            pd.DataFrame,
        ):
            for meta_file in self.plot_data["label_matched_and_unmatched"][
                "df_picked"
            ]["metadata_filename"].unique():
                meta_basename = os.path.basename(meta_file)

                # now for each micrograph, make the plots
                for ugraph_index, ugraph_filename in enumerate(
                    np.unique(
                        self.plot_data["label_matched_and_unmatched"][
                            "df_picked"
                        ]["ugraph_filename"]
                    )[: self.num_ugraphs]
                ):
                    if (
                        isinstance(
                            self.plot_data["label_matched_and_unmatched"][
                                "df_truth"
                            ],
                            pd.DataFrame,
                        )
                        and isinstance(
                            self.plot_data["label_matched_and_unmatched"][
                                "df_picked"
                            ],
                            pd.DataFrame,
                        )
                        and isinstance(
                            self.plot_data[
                                "label_matched_and_unmatched_{}".format(
                                    meta_basename.split(".")[0]
                                )
                            ]["df_mp"],
                            pd.DataFrame,
                        )
                        and isinstance(
                            self.plot_data[
                                "label_matched_and_unmatched_{}".format(
                                    meta_basename.split(".")[0]
                                )
                            ]["df_up"],
                            pd.DataFrame,
                        )
                        and isinstance(
                            self.plot_data[
                                "label_matched_and_unmatched_{}".format(
                                    meta_basename.split(".")[0]
                                )
                            ]["df_mt"],
                            pd.DataFrame,
                        )
                        and isinstance(
                            self.plot_data[
                                "label_matched_and_unmatched_{}".format(
                                    meta_basename.split(".")[0]
                                )
                            ]["df_ut"],
                            pd.DataFrame,
                        )
                    ):
                        # matched_picked_particles
                        if (
                            len(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_mp"]
                            )
                            > 0
                        ):
                            print(
                                "Plotting matched picked particles in"
                                " micrograph {}, \
                                from metadata file {}".format(
                                    ugraph_filename, meta_basename
                                )
                            )
                            (
                                fig,
                                ax,
                                outfilename,
                            ) = make_matched_picked_plot(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_mp"],
                                meta_file,
                                ugraph_filename,
                                ugraph_index,
                                self.box_width,
                                self.box_height,
                                self.mrc_dir,
                                self.verbose,
                                self.plot_dir,
                            )

                            self._save_plot(fig, ax, outfilename)
                        else:
                            print("There are no matched picked particles!")

                        # matched_truth_particles
                        if (
                            len(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_mt"]
                            )
                            > 0
                        ):
                            print(
                                "Plotting matched truth particles in"
                                " micrograph {}, \
                                from metadata file {}".format(
                                    ugraph_filename, meta_basename
                                )
                            )
                            (
                                fig,
                                ax,
                                outfilename,
                            ) = make_matched_truth_plot(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_mt"],
                                meta_file,
                                ugraph_filename,
                                ugraph_index,
                                self.box_width,
                                self.box_height,
                                self.mrc_dir,
                                self.verbose,
                                self.plot_dir,
                            )
                            self._save_plot(fig, ax, outfilename)
                        else:
                            print("There are no matched truth particles!")

                        # umatched_picked_particles
                        if (
                            len(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_up"]
                            )
                            > 0
                        ):
                            print(
                                "Plotting unmatched picked particles in"
                                " micrograph {}, \
                                from metadata file {}".format(
                                    ugraph_filename, meta_basename
                                )
                            )
                            (
                                fig,
                                ax,
                                outfilename,
                            ) = make_unmatched_picked_plot(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_up"],
                                meta_file,
                                ugraph_filename,
                                ugraph_index,
                                self.box_width,
                                self.box_height,
                                self.mrc_dir,
                                self.verbose,
                                self.plot_dir,
                            )
                            self._save_plot(fig, ax, outfilename)
                        else:
                            print("There are no matched truth particles!")

                        # unmatched_truth_particles
                        if (
                            len(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_ut"]
                            )
                            > 0
                        ):
                            print(
                                "Plotting unmatched truth particles in"
                                " micrograph {}, \
                                from metadata file {}".format(
                                    ugraph_filename, meta_basename
                                )
                            )
                            (
                                fig,
                                ax,
                                outfilename,
                            ) = make_unmatched_truth_plot(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_ut"],
                                meta_file,
                                ugraph_filename,
                                ugraph_index,
                                self.box_width,
                                self.box_height,
                                self.mrc_dir,
                                self.verbose,
                                self.plot_dir,
                            )
                            self._save_plot(fig, ax, outfilename)
                        else:
                            print("There are no unmatched truth particles!")

                        # matched_picked_particles and
                        # unmatched_picked_particles
                        if (
                            len(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_mp"]
                            )
                            > 0
                            and len(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_up"]
                            )
                            > 0
                        ):
                            print(
                                "Plotting matched picked particles and"
                                " unmatched picked particles"
                                " in micrograph"
                                " {}, from metadata file {}".format(
                                    ugraph_filename,
                                    meta_basename,
                                )
                            )
                            (
                                fig,
                                ax,
                                outfilename,
                            ) = make_matched_unmatched_picked_plot(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_mp"],
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_up"],
                                meta_file,
                                self.plot_data["label_matched_and_unmatched"][
                                    "df_truth"
                                ],
                                ugraph_filename,
                                ugraph_index,
                                self.box_width,
                                self.box_height,
                                self.mrc_dir,
                                self.verbose,
                                self.plot_dir,
                            )
                            self._save_plot(fig, ax, outfilename)
                        else:
                            if (
                                len(
                                    self.plot_data[
                                        "label_matched_and_"
                                        "unmatched_{}".format(
                                            meta_basename.split(".")[0]
                                        )
                                    ]["df_mp"]
                                )
                                == 0
                            ):
                                print(
                                    "There are no matched picked particles!"
                                    " Not plotting comparison to unmatched"
                                    " picked particles!"
                                )
                            if (
                                len(
                                    self.plot_data[
                                        "label_matched_and_"
                                        "unmatched_{}".format(
                                            meta_basename.split(".")[0]
                                        )
                                    ]["df_up"]
                                )
                                == 0
                            ):
                                print(
                                    "There are no unmatched picked particles!"
                                    " Not plotting comparison to matched"
                                    " picked particles!"
                                )

                        # matched_truth_particles and
                        # unmatched_truth_particles
                        if (
                            len(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_mt"]
                            )
                            > 0
                            and len(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_ut"]
                            )
                            > 0
                        ):
                            print(
                                "Plotting matched truth particles and"
                                " unmatched truth particles"
                                " in micrograph"
                                " {}, from metadata file {}".format(
                                    ugraph_filename,
                                    meta_basename,
                                )
                            )
                            (
                                fig,
                                ax,
                                outfilename,
                            ) = make_matched_unmatched_truth_plot(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_mt"],
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_ut"],
                                meta_file,
                                ugraph_filename,
                                ugraph_index,
                                self.box_width,
                                self.box_height,
                                self.mrc_dir,
                                self.verbose,
                                self.plot_dir,
                            )
                            self._save_plot(fig, ax, outfilename)
                        else:
                            if (
                                len(
                                    self.plot_data[
                                        "label_matched_and_"
                                        "unmatched_{}".format(
                                            meta_basename.split(".")[0]
                                        )
                                    ]["mt_up"]
                                )
                                == 0
                            ):
                                print(
                                    "There are no matched truth particles!"
                                    " Not plotting comparison to unmatched"
                                    " truth particles!"
                                )
                            if (
                                len(
                                    self.plot_data[
                                        "label_matched_and"
                                        "_unmatched_{}".format(
                                            meta_basename.split(".")[0]
                                        )
                                    ]["df_ut"]
                                )
                                == 0
                            ):
                                print(
                                    "There are no unmatched truth particles!"
                                    " Not plotting comparison to matched"
                                    " truth particles!"
                                )

                        # unmatched_picked particles and truth_particles
                        if (
                            len(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_up"]
                            )
                            > 0
                            and len(
                                self.plot_data["label_matched_and_unmatched"][
                                    "df_truth"
                                ]
                            )
                            > 0
                        ):
                            print(
                                "Plotting unmatched particles and"
                                " truth particles in micrograph"
                                " {}, from metadata file {}".format(
                                    ugraph_filename,
                                    meta_basename,
                                )
                            )
                            (
                                fig,
                                ax,
                                outfilename,
                            ) = make_unmatched_picked_and_truth_plot(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_up"],
                                meta_file,
                                self.plot_data["label_matched_and_unmatched"][
                                    "df_truth"
                                ],
                                ugraph_filename,
                                ugraph_index,
                                self.box_width,
                                self.box_height,
                                self.mrc_dir,
                                self.verbose,
                                self.plot_dir,
                            )
                            self._save_plot(fig, ax, outfilename)

                        else:
                            if (
                                len(
                                    self.plot_data[
                                        "label_matched_and"
                                        "_unmatched_{}".format(
                                            meta_basename.split(".")[0]
                                        )
                                    ]["df_up"]
                                )
                                == 0
                            ):
                                print(
                                    "There are no unmatched picked particles!"
                                    " Not plotting comparison to truth"
                                    " particles!"
                                )
                            if (
                                len(
                                    self.plot_data[
                                        "label_matched_and_unmatched"
                                    ]["df_truth"]
                                )
                                == 0
                            ):
                                print(
                                    "There are no truth particles!"
                                    " Not plotting comparison to unmatched"
                                    " picked particles!"
                                )

                        # picked_particles and unmatched_truth_particles
                        if (
                            len(
                                self.plot_data["label_matched_and_unmatched"][
                                    "df_picked"
                                ]
                            )
                            > 0
                            and len(
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_ut"]
                            )
                            > 0
                        ):
                            print(
                                "Plotting picked particles and"
                                " unmatched truth particles"
                                " in micrograph"
                                " {}, from metadata file {}".format(
                                    ugraph_filename,
                                    meta_basename,
                                )
                            )
                            (
                                fig,
                                ax,
                                outfilename,
                            ) = make_unmatched_truth_and_picked_plot(
                                self.plot_data["label_matched_and_unmatched"][
                                    "df_picked"
                                ],
                                meta_file,
                                self.plot_data[
                                    "label_matched_and_unmatched_{}".format(
                                        meta_basename.split(".")[0]
                                    )
                                ]["df_ut"],
                                ugraph_filename,
                                ugraph_index,
                                self.box_width,
                                self.box_height,
                                self.mrc_dir,
                                self.verbose,
                                self.plot_dir,
                            )
                            self._save_plot(
                                fig,
                                ax,
                                outfilename,
                            )
                        else:
                            if (
                                len(
                                    self.plot_data[
                                        "label_matched_and_unmatched"
                                    ]["df_picked"]
                                )
                                == 0
                            ):
                                print(
                                    "There are no picked particles!"
                                    " Not plotting comparison to unmatched"
                                    " truth particles!"
                                )
                            if (
                                len(
                                    self.plot_data[
                                        "label_matched_and_"
                                        "unmatched_{}".format(
                                            meta_basename.split(".")[0]
                                        )
                                    ]["df_ut"]
                                )
                                == 0
                            ):
                                print(
                                    "There are no unmatched truth particles!"
                                    " Not plotting comparison to"
                                    " picked particles!"
                                )
                        self.save_dataframes(self.plot_dir, overwrite_data)

                    else:
                        raise TypeError(
                            "One of the following is not a a pd.DataFrame:"
                            'self.plot_data["label_matched_and_unmatched'
                            '["df_truth"]'
                            'self.plot_data["label_matched_and_unmatched'
                            '["df_picked"]'
                            'self.plot_data["label_matched_and_unmatched'
                            f"_{meta_basename}"
                            '"]["df_mp"]'
                            'self.plot_data["label_matched_and_unmatched'
                            f"_{meta_basename}"
                            '"]["df_up"]'
                            'self.plot_data["label_matched_and_unmatched'
                            f"_{meta_basename}"
                            '"]["df_mt"]'
                            'self.plot_data["label_matched_and_unmatched'
                            f"_{meta_basename}"
                            '"]["df_ut"]'
                        )
        else:
            raise TypeError(
                'self.plot_data["label_matched_and_unmatched"]['
                '"df_picked"] is not a pd.DataFrame!'
            )

    def _save_plot(self, fig, ax, outfilename: str):
        # save the plot
        fig.savefig(outfilename, dpi=self.dpi, bbox_inches="tight")
        if self.pdf:
            fig.savefig(
                outfilename.replace(".png", ".pdf"),
                bbox_inches="tight",
            )
        fig.clf()


def make_matched_picked_plot(
    df_mp: pd.DataFrame,
    meta_file: str,
    ugraph_filename: str,
    ugraph_index: int,
    box_width: float,
    box_height: float,
    mrc_dir: str,
    verbose: bool,
    plot_dir: str = "",
):
    meta_basename = os.path.basename(meta_file)
    fig, ax = labelMicrograph.label_micrograph_picked(
        df_mp,
        meta_file,
        ugraph_index,
        mrc_dir,
        box_width=box_width,
        box_height=box_height,
        edgecolor=[0, 0, 1],
        verbose=verbose,
    )
    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    outfilename = os.path.join(
        plot_dir,
        "{}_{}_matched_picked.png".format(
            ugraph_filename.strip(".mrc"),
            meta_basename.split(".")[0],
        ),
    )
    return fig, ax, outfilename


def make_matched_truth_plot(
    df_mt: pd.DataFrame,
    meta_file: str,
    ugraph_filename: str,
    ugraph_index: int,
    box_width: float,
    box_height: float,
    mrc_dir: str,
    verbose: bool,
    plot_dir: str = "",
):
    meta_basename = os.path.basename(meta_file)
    fig, ax = labelMicrograph.label_micrograph_truth(
        df_mt,
        ugraph_index,
        mrc_dir,
        box_width=box_width,
        box_height=box_height,
        edgecolor=[0, 0, 0.5],
        verbose=verbose,
    )
    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    outfilename = os.path.join(
        plot_dir,
        "{}_{}_matched_truth.png".format(
            ugraph_filename.strip(".mrc"),
            meta_basename.split(".")[0],
        ),
    )
    return fig, ax, outfilename


def make_unmatched_picked_plot(
    df_up: pd.DataFrame,
    meta_file: str,
    ugraph_filename: str,
    ugraph_index: int,
    box_width: float,
    box_height: float,
    mrc_dir: str,
    verbose: bool,
    plot_dir: str = "",
):
    meta_basename = os.path.basename(meta_file)
    fig, ax = labelMicrograph.label_micrograph_picked(
        df_up,
        meta_file,
        ugraph_index,
        mrc_dir,
        box_width=box_width,
        box_height=box_height,
        edgecolor=[1, 1, 0],
        verbose=verbose,
    )
    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    outfilename = os.path.join(
        plot_dir,
        "{}_{}_unmatched_picked.png".format(
            ugraph_filename.strip(".mrc"),
            meta_basename.split(".")[0],
        ),
    )
    return fig, ax, outfilename


def make_unmatched_truth_plot(
    df_ut: pd.DataFrame,
    meta_file: str,
    ugraph_filename: str,
    ugraph_index: int,
    box_width: float,
    box_height: float,
    mrc_dir: str,
    verbose: bool,
    plot_dir: str = "",
):
    meta_basename = os.path.basename(meta_file)
    fig, ax = labelMicrograph.label_micrograph_truth(
        df_ut,
        ugraph_index,
        mrc_dir,
        box_width=box_width,
        box_height=box_height,
        edgecolor=[1, 1, 0.5],
        verbose=verbose,
    )
    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    outfilename = os.path.join(
        plot_dir,
        "{}_{}_unmatched_truth.png".format(
            ugraph_filename.strip(".mrc"),
            meta_basename.split(".")[0],
        ),
    )
    return fig, ax, outfilename


def make_matched_unmatched_picked_plot(
    df_mp: pd.DataFrame,
    df_up: pd.DataFrame,
    meta_file: str,
    df_truth: pd.DataFrame,
    ugraph_filename: str,
    ugraph_index: int,
    box_width: float,
    box_height: float,
    mrc_dir: str,
    verbose: bool,
    plot_dir: str = "",
):
    meta_basename = os.path.basename(meta_file)
    fig, ax = labelMicrograph.label_micrograph_picked_and_picked(
        df_mp,
        df_up,
        meta_file,
        df_truth,
        ugraph_index,
        mrc_dir,
        box_width=box_width,
        box_height=box_height,
        picked1_color=[0, 0, 1],
        picked2_color=[1, 1, 0],
        verbose=verbose,
    )
    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    outfilename = os.path.join(
        plot_dir,
        "{}_{}_matched_picked_and_"
        "unmatched_picked.png".format(
            ugraph_filename.strip(".mrc"),
            meta_basename.split(".")[0],
        ),
    )
    return fig, ax, outfilename


def make_matched_unmatched_truth_plot(
    df_mt: pd.DataFrame,
    df_ut: pd.DataFrame,
    meta_file: str,
    ugraph_filename: str,
    ugraph_index: int,
    box_width: float,
    box_height: float,
    mrc_dir: str,
    verbose: bool,
    plot_dir: str = "",
):
    meta_basename = os.path.basename(meta_file)
    fig, ax = labelMicrograph.label_micrograph_truth_and_truth(
        df_mt,
        df_ut,
        ugraph_index,
        mrc_dir,
        box_width=box_width,
        box_height=box_height,
        truth_color1=[0, 0, 0.5],
        truth_color2=[1, 1, 0.5],
        verbose=verbose,
    )
    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    outfilename = os.path.join(
        plot_dir,
        "{}_{}_matched_truth_and"
        "_unmatched_truth.png".format(
            ugraph_filename.strip(".mrc"),
            meta_basename.split(".")[0],
        ),
    )
    return fig, ax, outfilename


def make_unmatched_picked_and_truth_plot(
    df_up: pd.DataFrame,
    meta_file: str,
    df_truth: pd.DataFrame,
    ugraph_filename: str,
    ugraph_index: int,
    box_width: float,
    box_height: float,
    mrc_dir: str,
    verbose: bool,
    plot_dir: str = "",
):
    meta_basename = os.path.basename(meta_file)
    fig, ax = labelMicrograph.label_micrograph_truth_and_picked(
        df_up,
        meta_file,
        df_truth,
        ugraph_index,
        mrc_dir,
        box_width=box_width,
        box_height=box_height,
        picked_color=[1, 1, 0],
        truth_color=[0, 1, 0],
        verbose=verbose,
    )
    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    outfilename = os.path.join(
        plot_dir,
        "{}_{}_unmatched_picked_and_truth.png".format(
            ugraph_filename.strip(".mrc"),
            meta_basename.split(".")[0],
        ),
    )
    return fig, ax, outfilename


def make_unmatched_truth_and_picked_plot(
    df_picked: pd.DataFrame,
    meta_file: str,
    df_ut: pd.DataFrame,
    ugraph_filename: str,
    ugraph_index: int,
    box_width: float,
    box_height: float,
    mrc_dir: str,
    verbose: bool,
    plot_dir: str = "",
):
    meta_basename = os.path.basename(meta_file)
    fig, ax = labelMicrograph.label_micrograph_truth_and_picked(
        df_picked,
        meta_file,
        df_ut,
        ugraph_index,
        mrc_dir,
        box_width=box_width,
        box_height=box_height,
        picked_color=[1, 0, 0],
        truth_color=[1, 1, 0.5],
        verbose=verbose,
    )
    # remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    outfilename = os.path.join(
        plot_dir,
        "{}_{}_picked_and_unmatched_truth.png".format(
            ugraph_filename.strip(".mrc"),
            meta_basename.split(".")[0],
        ),
    )
    return fig, ax, outfilename


class labelMicrograph(object):
    """Containing static methods to label ugraphs
    with bounding boxes around truth and/or predicted
    particles

    Args:
        object (_type_): _description_
    """

    @staticmethod
    def _twoD_image_bboxs(
        particles_x: np.ndarray,
        particles_y: np.ndarray,
        box_width: float,
        box_height: float,
        verbose: bool = False,
    ) -> List[List[float]]:
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

        # use this list to fill a list of boxes,
        # each corresponding to a particle
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

    @staticmethod
    def label_micrograph_truth(
        particles: pd.DataFrame,
        ugraph_index: int = 0,
        mrc_dir: str = "",
        box_width: float = 50,
        box_height: float = 50,
        edgecolor: list = [0, 1, 0],
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

            fig, ax = plt.subplots(figsize=[14, 14])
            ax.imshow(data[0], cmap="gray")
            fig.tight_layout()

            # Now that you've plotted the true central points of each particle,
            # also plot the boxes
            boxes = labelMicrograph._twoD_image_bboxs(
                np.array(particles_ugraph["position_x"]),
                np.array(particles_ugraph["position_y"]),
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
                    edgecolor=edgecolor,
                    facecolor="none",
                )
                ax.add_patch(rect)
            # green_patch = patches.Patch(color="green",
            # label="Truth particles")
            # ax.legend(handles=[green_patch])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        return fig, ax

    @staticmethod
    def label_micrograph_picked(
        particles: pd.DataFrame,
        metadata_filename: str | list[str],
        ugraph_index: int = 0,
        mrc_dir: str = "",
        box_width: float = 50,
        box_height: float = 50,
        edgecolor: list = [1, 0, 0],
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

            fig, ax = plt.subplots(figsize=[14, 14])
            ax.imshow(data[0], cmap="gray")
            fig.tight_layout()

            # Now that you've plotted the true central points of each particle,
            # also plot the boxes
            boxes = labelMicrograph._twoD_image_bboxs(
                np.array(particles_ugraph["position_x"]),
                np.array(particles_ugraph["position_y"]),
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
                    edgecolor=edgecolor,
                    facecolor=facecolor,
                    alpha=alpha,
                )
                ax.add_patch(rect)
            # red_patch = patches.Patch(color="red", label="Picked particles")
            # ax.legend(handles=[red_patch])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        return fig, ax

    @staticmethod
    def label_micrograph_truth_and_picked(
        picked_particles: pd.DataFrame,
        metadata_filename: str | List[str],
        truth_particles: pd.DataFrame,
        ugraph_index: int = 0,
        mrc_dir: str = "",
        box_width: float = 50,
        box_height: float = 50,
        picked_color: list = [1, 0, 0],
        truth_color: list = [0, 1, 0],
        verbose: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        # group the picked particles by metadata file
        if isinstance(metadata_filename, list):
            metadata_filename = metadata_filename[0]
        picked_particles = picked_particles.groupby(
            "metadata_filename"
        ).get_group(metadata_filename)
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

            fig, ax = plt.subplots(figsize=[14, 14])
            ax.imshow(data[0], cmap="gray")
            fig.tight_layout()

            # Now that you've plotted the true central points of each particle,
            # also plot the boxes
            boxes = labelMicrograph._twoD_image_bboxs(
                np.array(picked_particles_ugraph["position_x"]),
                np.array(picked_particles_ugraph["position_y"]),
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
                    edgecolor=picked_color,
                    facecolor="none",
                )
                ax.add_patch(rect)

            boxes = labelMicrograph._twoD_image_bboxs(
                np.array(truth_particles_ugraph["position_x"]),
                np.array(truth_particles_ugraph["position_y"]),
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
                    edgecolor=truth_color,
                    facecolor="none",
                )
                ax.add_patch(rect)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        return fig, ax

    @staticmethod
    def label_micrograph_truth_and_truth(
        truth_particles1: pd.DataFrame,
        truth_particles2: pd.DataFrame,
        ugraph_index: int = 0,
        mrc_dir: str = "",
        box_width: float = 50,
        box_height: float = 50,
        truth_color1: list = [1, 0, 0],
        truth_color2: list = [0, 1, 0],
        verbose: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        # get the micrograph name
        ugraph_filename = np.unique(
            np.append(
                truth_particles1["ugraph_filename"].to_numpy(dtype=str),
                truth_particles2["ugraph_filename"].to_numpy(dtype=str),
                axis=0,
            )
        )[ugraph_index]
        print(f"plotted index {ugraph_index}; micrograph: {ugraph_filename}")
        ugraph_path = os.path.join(mrc_dir, ugraph_filename)
        truth_particles1_ugraph = truth_particles1.groupby(
            "ugraph_filename"
        ).get_group(ugraph_filename)
        truth_particles2_ugraph = truth_particles2.groupby(
            "ugraph_filename"
        ).get_group(ugraph_filename)

        # Open up a mrc file to overlay the boxes with
        with mrcfile.open(ugraph_path) as mrc:
            data = mrc.data

            fig, ax = plt.subplots(figsize=[14, 14])
            ax.imshow(data[0], cmap="gray")
            fig.tight_layout()

            # Now that you've plotted the true central points of each particle,
            # also plot the boxes
            boxes = labelMicrograph._twoD_image_bboxs(
                np.array(truth_particles1_ugraph["position_x"]),
                np.array(truth_particles1_ugraph["position_y"]),
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
                    edgecolor=truth_color1,
                    facecolor="none",
                )
                ax.add_patch(rect)

            boxes = labelMicrograph._twoD_image_bboxs(
                np.array(truth_particles2_ugraph["position_x"]),
                np.array(truth_particles2_ugraph["position_y"]),
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
                    edgecolor=truth_color2,
                    facecolor="none",
                )
                ax.add_patch(rect)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        return fig, ax

    @staticmethod
    def label_micrograph_picked_and_picked(
        picked_particles1: pd.DataFrame,
        picked_particles2: pd.DataFrame,
        metadata_filename: str | List[str],
        truth_particles: pd.DataFrame,
        ugraph_index: int = 0,
        mrc_dir: str = "",
        box_width: float = 50,
        box_height: float = 50,
        picked1_color: list = [1, 0, 0],
        picked2_color: list = [0, 1, 0],
        verbose: bool = False,
    ) -> Tuple[plt.Figure, plt.Axes]:
        # group the picked particles by metadata file
        if isinstance(metadata_filename, list):
            metadata_filename = metadata_filename[0]
        picked_particles1 = picked_particles1.groupby(
            "metadata_filename"
        ).get_group(metadata_filename)
        picked_particles2 = picked_particles2.groupby(
            "metadata_filename"
        ).get_group(metadata_filename)
        # get the micrograph name
        ugraph_filename = np.unique(truth_particles["ugraph_filename"])[
            ugraph_index
        ]
        print(f"plotted index {ugraph_index}; micrograph: {ugraph_filename}")
        ugraph_path = os.path.join(mrc_dir, ugraph_filename)
        picked_particles1_ugraph = picked_particles1.groupby(
            "ugraph_filename"
        ).get_group(ugraph_filename)
        picked_particles2_ugraph = picked_particles2.groupby(
            "ugraph_filename"
        ).get_group(ugraph_filename)

        # Open up a mrc file to overlay the boxes with
        with mrcfile.open(ugraph_path) as mrc:
            data = mrc.data

            fig, ax = plt.subplots(figsize=[14, 14])
            ax.imshow(data[0], cmap="gray")
            fig.tight_layout()

            # Now that you've plotted the true central points of each particle,
            # also plot the boxes
            boxes = labelMicrograph._twoD_image_bboxs(
                np.array(picked_particles1_ugraph["position_x"]),
                np.array(picked_particles1_ugraph["position_y"]),
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
                    edgecolor=picked1_color,
                    facecolor="none",
                )
                ax.add_patch(rect)

            boxes = labelMicrograph._twoD_image_bboxs(
                np.array(picked_particles2_ugraph["position_x"]),
                np.array(picked_particles2_ugraph["position_y"]),
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
                    edgecolor=picked2_color,
                    facecolor="none",
                )
                ax.add_patch(rect)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        return fig, ax


class plotPrecision(plotDataFrame):
    def __init__(
        self,
        job_types: dict[str, str],
        order: list[str],
        plot_data: dict[str, dict[str, pd.DataFrame]] | None = None,
        plot_dir: str = "",
        dpi: int = 300,
        pdf: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(plot_data)

        # set up the dataframe
        if plot_data:
            self.plot_data = plot_data

        self.job_types = job_types
        self.order = order
        self.plot_dir = plot_dir
        self.dpi = dpi
        self.pdf = pdf
        self.verbose = verbose

    def setup_plot_data(self, df_precision: pd.DataFrame):
        self.plot_data = {"plot_precision": {"df_precision": df_precision}}

    def setup_plot_data_empty(self):
        self.plot_data = {"plot_precision": {"df_precision": None}}

    def make_and_save_plots(
        self,
        overwrite_data: bool = False,
    ):
        # save/overwrite data file
        self.save_dataframes(self.plot_dir, overwrite_data)

        if self.verbose:
            print(
                "meta_files in df: {}".format(
                    self.plot_data["plot_precision"]["df_precision"][
                        "metadata_filename"
                    ].unique()
                )
            )
            print("plotting precision...")
        # precision
        fig, ax = plot_precision(
            self.plot_data["plot_precision"]["df_precision"],
            self.job_types,
            self.order,
        )
        outfilename = os.path.join(self.plot_dir, "precision.png")
        self._save_plot(fig, ax, outfilename)

        if self.verbose:
            print("plotting recall...")
        # recall
        fig, ax = plot_recall(
            self.plot_data["plot_precision"]["df_precision"],
            self.job_types,
            self.order,
        )
        outfilename = os.path.join(self.plot_dir, "recall.png")
        self._save_plot(fig, ax, outfilename)

        if self.verbose:
            print("plotting precision and recall in one plot...")
        # precision and recall
        fig, ax = plot_precision_and_recall(
            self.plot_data["plot_precision"]["df_precision"],
            self.job_types,
            self.order,
        )
        outfilename = os.path.join(self.plot_dir, "precision_and_recall.png")
        self._save_plot(fig, ax, outfilename)

        if self.verbose:
            print("plotting F1 score...")
        # f1 score
        fig, ax = plot_f1_score(
            self.plot_data["plot_precision"]["df_precision"],
            self.job_types,
            self.order,
        )
        outfilename = os.path.join(self.plot_dir, "f1_score.png")
        self._save_plot(fig, ax, outfilename)

    def _save_plot(self, fig, ax, outfilename: str):
        # save the plot
        fig.savefig(outfilename, dpi=self.dpi, bbox_inches="tight")
        if self.pdf:
            fig.savefig(
                outfilename.replace(".png", ".pdf"), bbox_inches="tight"
            )
        fig.clf()


def plot_precision(
    df_precision: pd.DataFrame,
    job_types: Dict[str, str],
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

    fig, ax = plt.subplots(figsize=(7, 3.5))
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
                job_types[metadata_filename]
                for metadata_filename in df_precision[
                    "metadata_filename"
                ].unique()
            ],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=12,
        )
    else:
        ax.set_xticklabels(
            [job_types[metadata_filename] for metadata_filename in order],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=12,
        )
    # remove legend
    ax.legend().remove()
    # add colorbar
    sm = plt.cm.ScalarMappable(
        cmap="RdYlBu",
        norm=plt.Normalize(
            vmin=df_precision["defocus"].min() / 10000,
            vmax=df_precision["defocus"].max() / 10000,
        ),
    )
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("defocus (\u03bcm)", rotation=270, labelpad=20, fontsize=12)
    # add labels
    ax.set_xlabel("")
    ax.set_ylabel("precision", fontsize=14)
    ax.set_title("Precision for different job types", fontsize=16)
    fig.tight_layout()
    return fig, ax


def plot_recall(
    df_precision: pd.DataFrame,
    job_types: Dict[str, str],
    order: list[str] | None = None,
):
    fig, ax = plt.subplots(figsize=(7, 3.5))
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
                job_types[metadata_filename]
                for metadata_filename in df_precision[
                    "metadata_filename"
                ].unique()
            ],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=12,
        )
    else:
        ax.set_xticklabels(
            [job_types[metadata_filename] for metadata_filename in order],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=12,
        )
    # remove legend
    ax.legend().remove()
    # add colorbar
    sm = plt.cm.ScalarMappable(
        cmap="RdYlBu",
        norm=plt.Normalize(
            vmin=df_precision["defocus"].min() / 10000,
            vmax=df_precision["defocus"].max() / 10000,
        ),
    )
    sm._A = []
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label("defocus (\u03bcm)", rotation=270, labelpad=20, fontsize=12)
    # add labels
    ax.set_xlabel("")
    ax.set_ylabel("recall", fontsize=14)
    ax.set_title("Recall for different job types", fontsize=16)
    fig.tight_layout()
    return fig, ax


def plot_precision_and_recall(
    df_precision: pd.DataFrame,
    job_types: Dict[str, str],
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

    fig, ax = plt.subplots(figsize=(7, 3.5))
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
    # change the xtix labels to the job_types
    if order is None:
        ax.set_xticklabels(
            [
                job_types[meta_file]
                for meta_file in np.unique(df["metadata_filename"])
                if meta_file in job_types.keys()
            ],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=12,
        )
    else:
        ax.set_xticklabels(
            [job_types[meta_file] for meta_file in order],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=12,
        )
    # add legend below axis
    ax.legend().set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 1.1),
        fontsize=10,
    )
    fig.tight_layout()
    return fig, ax


def plot_f1_score(
    df_precision: pd.DataFrame,
    job_types: Dict[str, str],
    order: list[str] | None = None,
):
    # compute f1 score from the precision and recall
    df_precision["f1_score"] = (
        2
        * (df_precision["precision"] * df_precision["recall"])
        / (df_precision["precision"] + df_precision["recall"])
    )
    fig, ax = plt.subplots(figsize=(7, 3.5))
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
                job_types[metadata_filename]
                for metadata_filename in df_precision[
                    "metadata_filename"
                ].unique()
            ],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=12,
        )
    else:
        ax.set_xticklabels(
            [job_types[metadata_filename] for metadata_filename in order],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=12,
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
    cbar.set_label("defocus (Å)", rotation=270, labelpad=20, fontsize=12)
    # add labels
    ax.set_xlabel("")
    ax.set_ylabel("f1 score", fontsize=14)
    ax.set_title("F1 score for different job types", fontsize=16)
    fig.tight_layout()
    return fig, ax


class plotBoundaryInvestigation(plotDataFrame):
    def __init__(
        self,
        job_types: dict[str, str],
        bin_width: list[int],
        axis: list[str],
        plot_data: dict[str, dict[str, pd.DataFrame]] | None = None,
        plot_dir: str = "",
        dpi: int = 300,
        pdf: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(plot_data)

        if plot_data:
            self.plot_data = plot_data

        self.job_types = job_types
        self.bin_width = bin_width
        self.axis = axis
        self.plot_dir = plot_dir
        self.dpi = dpi
        self.pdf = pdf
        self.verbose = verbose

    def setup_plot_data(
        self,
        df_truth,
        df_picked,
    ):
        self.plot_data = {"plot_boundary_investigation": {}}
        self.plot_data["plot_boundary_investigation"]["df_truth"] = df_truth
        self.plot_data["plot_boundary_investigation"]["df_picked"] = df_picked

    def setup_plot_data_empty(self):
        self.plot_data = {"plot_boundary_investigation": {}}
        self.plot_data["plot_boundary_investigation"]["df_truth"] = None
        self.plot_data["plot_boundary_investigation"]["df_picked"] = None

    def make_and_save_plots(
        self,
        overwrite_data=False,
    ):
        # save/overwrite data file
        self.save_dataframes(self.plot_dir, overwrite_data)

        if isinstance(
            self.plot_data["plot_boundary_investigation"]["df_truth"],
            pd.DataFrame,
        ) and isinstance(
            self.plot_data["plot_boundary_investigation"]["df_picked"],
            pd.DataFrame,
        ):
            for meta_file in self.plot_data["plot_boundary_investigation"][
                "df_picked"
            ]["metadata_filename"].unique():
                meta_basename = os.path.basename(meta_file)
                if self.verbose:
                    print(f"plotting boundary for metadata file {meta_file}")
                for a, bnwdth in zip(self.axis, self.bin_width):
                    fig, ax = plot_boundary_investigation(
                        self.plot_data["plot_boundary_investigation"][
                            "df_truth"
                        ],
                        self.plot_data["plot_boundary_investigation"][
                            "df_picked"
                        ],
                        meta_file,
                        self.job_types,
                        bnwdth,
                        a,
                    )
                    outfilename = os.path.join(
                        self.plot_dir,
                        f"{meta_basename.split('.')[0]}_boundary_{a}.png",
                    )
                    self._save_plot(fig, ax, outfilename)
        else:
            raise TypeError(
                "One of df_truth and df_picked is not a pd.DataFrame!"
            )

    def _save_plot(self, fig, ax, outfilename):
        # save the plot
        fig.savefig(outfilename, dpi=self.dpi, bbox_inches="tight")
        if self.pdf:
            fig.savefig(
                outfilename.replace(".png", ".pdf"), bbox_inches="tight"
            )
        fig.clf()


def plot_boundary_investigation(
    df_truth: pd.DataFrame,
    df_picked: pd.DataFrame,
    metadata_filename: str,
    job_types: Dict[str, str],
    bin_width: int = 100,
    axis: str = "x",
):
    if isinstance(metadata_filename, list):
        metadata_filename = metadata_filename[0]

    particles_per_ugraph = (
        df_truth.groupby("ugraph_filename").size().reset_index()
    )
    particles_per_ugraph.rename(
        columns={0: "particles_per_ugraph"}, inplace=True
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

    fig, ax = plt.subplots(figsize=(3.5, 3.5))
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
    ax.set_xlabel(f"{axis} position (Angstroms)", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(job_types[metadata_filename], fontsize=16)
    return fig, ax


class plotOverlap(plotDataFrame):
    def __init__(
        self,
        job_types: dict[str, str],
        plot_data: dict[str, dict[str, pd.DataFrame]] | None = None,
        plot_dir: str = "",
        dpi: int = 300,
        pdf: bool = False,
        verbose: bool = False,
    ) -> None:
        super().__init__(plot_data)

        if plot_data:
            self.plot_data = plot_data

        self.job_types = job_types
        self.plot_dir = plot_dir
        self.dpi = dpi
        self.pdf = pdf
        self.verbose = verbose

    def setup_plot_data(
        self,
        df_overlap: pd.DataFrame,
    ):
        self.plot_data = {"plot_overlap": {"df_overlap": df_overlap}}

    def setup_plot_data_empty(self):
        self.plot_data = {"plot_overlap": {"df_overlap": None}}

    def make_and_save_plots(
        self,
        overwrite_data=False,
    ):
        # save/overwrite data file
        self.save_dataframes(self.plot_dir, overwrite_data)

        if isinstance(
            self.plot_data["plot_overlap"]["df_overlap"], pd.DataFrame
        ):
            if self.verbose:
                print("plotting overlap...")
                for meta_file in self.plot_data["plot_overlap"]["df_overlap"][
                    "metadata_filename"
                ].unique():
                    meta_basename = os.path.basename(meta_file)
                    fig, ax = plot_overlap_investigation(
                        self.plot_data["plot_overlap"]["df_overlap"],
                        meta_file,
                    )
                    outfilename = os.path.join(
                        self.plot_dir,
                        f"{meta_basename.split('.')[0]}_overlap.png",
                    )
                    self._save_plot(fig, ax, outfilename)

                fig, ax = plot_overlap_investigation(
                    self.plot_data["plot_overlap"]["df_overlap"],
                    None,
                    self.job_types,
                )  # plot all
                outfilename = os.path.join(self.plot_dir, "overlap.png")
                self._save_plot(fig, ax, outfilename)
        else:
            raise TypeError("plot_overlap is not a pd.DataFrame!")

    def _save_plot(self, fig, ax, outfilename):
        # save the plot
        fig.savefig(outfilename, dpi=self.dpi, bbox_inches="tight")
        if self.pdf:
            fig.savefig(
                outfilename.replace(".png", ".pdf"), bbox_inches="tight"
            )
        fig.clf()


def plot_overlap_investigation(
    df_overlap: pd.DataFrame,
    metadata_filename: str | list[str] | None = None,
    job_types: Dict[str, str] | None = None,
):
    if metadata_filename is None:
        # plot all metadata files in one plot
        fig, ax = plt.subplots(figsize=(7, 3.5))
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
        handles = handles[: len(handles) // 2]
        labels = labels[: len(labels) // 2]
        if job_types is not None:
            labels = [job_types[label] for label in labels]
        ax.legend(
            handles,
            labels,
            fontsize=12,
        )
        ax.set_ylabel("# Overlaps", fontsize=14)
        fig.tight_layout()
        return fig, ax

    else:
        if isinstance(metadata_filename, list):
            metadata_filename = metadata_filename[0]
        print(f"plotting overlap for {metadata_filename}")
        # make a plot for each metadata file
        fig, ax = plt.subplots(figsize=(7, 3.5))
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
        ax.set_ylabel("# Overlaps", fontsize=14)
        fig.tight_layout()
        return fig, ax


def main(args):
    """This analysis tool makes plots of the picked and ground-truth particles
    in a number of micrographs. It then makes quantitative comparisons between
    the two.
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

    for plot_type in args.plot_types:
        if plot_type == "label_truth":  # plot the ground-truth particles
            label_truth = plotLabelTruth(
                args.mrc_dir,
                plot_dir=args.plot_dir,
                num_ugraphs=args.num_ugraphs,
                box_width=args.box_width,
                box_height=args.box_height,
                dpi=args.dpi,
                pdf=args.pdf,
                verbose=args.verbose,
            )
            label_truth.setup_plot_data(df_truth)
            label_truth.make_and_save_plots(overwrite_data=True)

        if plot_type == "label_picked":  # plot the picked particles
            label_picked = plotLabelPicked(
                args.mrc_dir,
                plot_dir=args.plot_dir,
                num_ugraphs=args.num_ugraphs,
                box_width=args.box_width,
                box_height=args.box_height,
                dpi=args.dpi,
                pdf=args.pdf,
                verbose=args.verbose,
            )
            label_picked.setup_plot_data(df_picked)
            label_picked.make_and_save_plots(overwrite_data=True)

        if plot_type == "label_truth_and_picked":
            label_truth_and_picked = plotLabelTruthAndPicked(
                args.mrc_dir,
                plot_dir=args.plot_dir,
                num_ugraphs=args.num_ugraphs,
                box_width=args.box_width,
                box_height=args.box_height,
                dpi=args.dpi,
                pdf=args.pdf,
                verbose=args.verbose,
            )
            label_truth_and_picked.setup_plot_data(df_truth, df_picked)
            label_truth_and_picked.make_and_save_plots(overwrite_data=True)

        if plot_type == "label_matched_and_unmatched":
            label_matched_and_unmatched = plotMatchedAndUnmatched(
                args.mrc_dir,
                plot_dir=args.plot_dir,
                num_ugraphs=args.num_ugraphs,
                box_width=args.box_width,
                box_height=args.box_height,
                dpi=args.dpi,
                pdf=args.pdf,
                verbose=args.verbose,
            )
            label_matched_and_unmatched.setup_plot_data(analysis)
            label_matched_and_unmatched.make_and_save_plots(
                overwrite_data=True
            )

        if plot_type == "precision":
            # first need to compute the precision statistics
            df_precision, _ = analysis.compute_precision(
                df_picked, df_truth, verbose=args.verbose
            )

            plot_precision = plotPrecision(
                job_types,
                order,
                plot_dir=args.plot_dir,
                dpi=args.dpi,
                pdf=args.pdf,
                verbose=args.verbose,
            )
            plot_precision.setup_plot_data(df_precision)
            plot_precision.make_and_save_plots(overwrite_data=True)

        if plot_type == "boundary":
            bin_width = [100, 100, 10]  # bin width for x, y, z
            axis = ["x", "y", "z"]
            plot_boundary = plotBoundaryInvestigation(
                job_types,
                bin_width,
                axis,
                plot_dir=args.plot_dir,
                dpi=args.dpi,
                pdf=args.pdf,
                verbose=args.verbose,
            )
            plot_boundary.setup_plot_data(
                df_truth,
                df_picked,
            )
            plot_boundary.make_and_save_plots(overwrite_data=True)

        if plot_type == "overlap":
            df_overlap = analysis.compute_overlap(
                df_picked, df_truth, verbose=args.verbose
            )

            plot_overlap = plotOverlap(
                job_types,
                plot_dir=args.plot_dir,
                dpi=args.dpi,
                pdf=args.pdf,
                verbose=args.verbose,
            )
            plot_overlap.setup_plot_data(df_overlap)
            plot_overlap.make_and_save_plots(overwrite_data=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
