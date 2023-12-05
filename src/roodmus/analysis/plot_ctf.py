"""Plot a comparison between the estimated CTF parameters and the
true values used in data generation.

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
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
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
        help=(
            "Directory with .mrc files. Assumed to be the same as"
            " 'config-dir'by default"
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "-N",
        "--num_ugraphs",
        help="Number of micrographs to consider in analyses. Default 'all'",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--meta_file",
        help=(
            "Particle metadata file. Can be .star (RELION) or .cs"
            " (CryoSPARC)"
        ),
        type=str,
    )
    parser.add_argument(
        "--plot_dir",
        help="Directory to output ctf file(s)",
        type=str,
        default="ctf_plots",
    )
    parser.add_argument(
        "--plot_types",
        help="types of plots to generate",
        type=str,
        nargs="+",
        default=["scatter"],
        choices=["scatter", "per-particle-scatter", "ctf"],
    )
    parser.add_argument(
        "--verbose", help="increase output verbosity", action="store_true"
    )
    parser.add_argument(
        "--tqdm", help="use tqdm progress bar", action="store_true"
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
    return "plot_ctf"


class plotPerParticleDefocusScatter(plotDataFrame):
    def __init__(
        self,
        meta_file: str,
        plot_data: dict[str, dict[str, pd.DataFrame]] | None = None,
        plot_dir: str = "",
        dpi: int = 300,
        pdf: bool = False,
    ) -> None:
        super().__init__(plot_data)

        if plot_data:
            self.plot_data = plot_data

        self.meta_file = meta_file
        self.plot_dir = plot_dir
        self.dpi = dpi
        self.pdf = pdf

    def setup_plot_data(
        self,
        df_truth: pd.DataFrame,
        df_picked: pd.DataFrame,
        df_matched_picked: pd.DataFrame,
        df_matched_truth: pd.DataFrame,
        df_pp_defoci: pd.DataFrame,
    ):
        self.plot_data = {"per_particle_defoci": {}}
        self.plot_data["per_particle_defoci"]["df_truth"] = df_truth
        self.plot_data["per_particle_defoci"]["df_picked"] = df_picked
        self.plot_data["per_particle_defoci"][
            "df_matched_picked"
        ] = df_matched_picked
        self.plot_data["per_particle_defoci"][
            "df_matched_truth"
        ] = df_matched_truth
        self.plot_data["per_particle_defoci"]["df_pp_defoci"] = df_pp_defoci

    def setup_plot_data_empty(self):
        self.plot_data = {"per_particle_defoci": {}}
        self.plot_data["per_particle_defoci"]["df_truth"] = None
        self.plot_data["per_particle_defoci"]["df_picked"] = None
        self.plot_data["per_particle_defoci"]["df_matched_picked"] = None
        self.plot_data["per_particle_defoci"]["df_matched_truth"] = None
        self.plot_data["per_particle_defoci"]["df_pp_defoci"] = None

    def make_and_save_plots(
        self,
        overwrite_data: bool = False,
    ):
        self.save_dataframes(self.plot_dir, overwrite_data)

        if isinstance(
            self.plot_data["per_particle_defoci"]["df_pp_defoci"],
            pd.DataFrame,
        ):
            fig, ax = plot_per_particle_defocus_scatter(
                self.plot_data["per_particle_defoci"]["df_pp_defoci"],
                self.meta_file,
            )
            outfilename = os.path.join(
                self.plot_dir, "ctf_per_particle_scatter.png"
            )
            self._save_plot(fig, ax, outfilename)
        else:
            TypeError(
                "One of df_truth, df_picked or df_pp_defoci is not"
                " a pd.Dataframe"
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


def plot_per_particle_defocus_scatter(
    pp_df: pd.DataFrame,
    metadata_filename: str,
    palette="BuGn",
):
    # plot the results
    plt.rcParams["font.size"] = 24
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    sns.scatterplot(
        x="defocus_truth",
        y="defocusU",
        data=pp_df,
        ax=ax[0],
        hue="ugraph_filename",
        palette=palette,
        legend=False,
        marker="+",
    )
    sns.scatterplot(
        x="defocus_truth",
        y="defocusV",
        data=pp_df,
        ax=ax[1],
        hue="ugraph_filename",
        palette=palette,
        legend=False,
        marker="+",
    )
    # add identity line
    min_defocus_truth = pp_df["defocus_truth"].min()
    max_defocus_truth = pp_df["defocus_truth"].max()
    ax[0].plot(
        [min_defocus_truth, max_defocus_truth],
        [min_defocus_truth, max_defocus_truth],
        color="black",
        linestyle="--",
        alpha=0.5,
    )
    ax[1].plot(
        [min_defocus_truth, max_defocus_truth],
        [min_defocus_truth, max_defocus_truth],
        color="black",
        linestyle="--",
        alpha=0.5,
    )

    # add labels
    ax[0].set_xlabel("defocus truth [$\u212B$]")
    ax[1].set_xlabel("defocus truth [$\u212B$]")
    ax[0].set_ylabel("defocusU estimated [$\u212B$]")
    ax[0].set_title("defocusU")
    ax[1].set_title("defocusV")
    ax[0].grid(False)
    ax[1].grid(False)
    # add colorbar legend
    sm = plt.cm.ScalarMappable(
        cmap=palette,
        norm=plt.Normalize(
            vmin=0, vmax=len(np.unique(pp_df["ugraph_filename"])) - 1
        ),
    )
    sm._A = []
    cbar = plt.colorbar(sm)
    cbar.set_label("micrograph")
    fig.tight_layout()
    return fig, ax


class plotDefocusScatter(plotDataFrame):
    def __init__(
        self,
        meta_file: str,
        plot_data: dict[str, dict[str, pd.DataFrame]] | None = None,
        plot_dir: str = "",
        dpi: int = 300,
        pdf: bool = False,
    ) -> None:
        super().__init__(plot_data)

        if plot_data:
            self.plot_data = plot_data

        self.meta_file = meta_file
        self.plot_dir = plot_dir
        self.dpi = dpi
        self.pdf = pdf

    def setup_plot_data(
        self,
        df_truth: pd.DataFrame,
        df_picked: pd.DataFrame,
    ):
        self.plot_data = {"defocus_scatter": {}}
        self.plot_data["defocus_scatter"]["df_truth"] = df_truth
        self.plot_data["defocus_scatter"]["df_picked"] = df_picked

    def setup_plot_data_empty(
        self,
    ):
        self.plot_data = {"defocus_scatter": {}}
        self.plot_data["defocus_scatter"]["df_truth"] = None
        self.plot_data["defocus_scatter"]["df_picked"] = None

    def make_and_save_plots(
        self,
        overwrite_data: bool = False,
    ):
        self.save_dataframes(self.plot_dir, overwrite_data)

        if isinstance(
            self.plot_data["defocus_scatter"]["df_truth"],
            pd.DataFrame,
        ) and isinstance(
            self.plot_data["defocus_scatter"]["df_picked"],
            pd.DataFrame,
        ):
            outfilename = os.path.join(self.plot_dir, "ctf_scatter.png")
            fig, ax = plot_defocus_scatter(
                df_picked=self.plot_data["defocus_scatter"]["df_picked"],
                metadata_filename=self.meta_file,
                df_truth=self.plot_data["defocus_scatter"]["df_truth"],
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


def plot_defocus_scatter(
    df_picked,
    metadata_filename,
    df_truth,
    palette="BuGn",
):
    # extract the group from the picked data frame
    if isinstance(metadata_filename, list):
        metadata_filename = metadata_filename[0]
    df_picked_groupby = df_picked.groupby("metadata_filename").get_group(
        metadata_filename
    )

    # used for plotting
    results = {
        "ugraph_filename": [],
        "defocusU": [],
        "defocusV": [],
        "defocus_truth": [],
    }

    # from the data frames, extract the defocus values on per-ugraph level
    df_picked_grouped = df_picked_groupby.groupby("ugraph_filename")
    df_truth_grouped = df_truth.groupby("ugraph_filename")
    for groupname in df_picked_grouped.groups.keys():
        defocusU = np.abs(
            df_picked_grouped.get_group(groupname)["defocusU"].mean()
        )
        defocusV = np.abs(
            df_picked_grouped.get_group(groupname)["defocusV"].mean()
        )
        defocus_truth = np.abs(
            df_truth_grouped.get_group(groupname)["defocus"].mean()
        )

        results["ugraph_filename"].append(groupname)
        results["defocusU"].append(defocusU)
        results["defocusV"].append(defocusV)
        results["defocus_truth"].append(defocus_truth)

    df = pd.DataFrame(results)

    # plot the results
    plt.rcParams["font.size"] = 24
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(1, 2, figsize=(16, 8), sharey=True)
    sns.scatterplot(
        x="defocus_truth",
        y="defocusU",
        data=df,
        ax=ax[0],
        hue="ugraph_filename",
        palette=palette,
        legend=False,
        marker="+",
    )
    sns.scatterplot(
        x="defocus_truth",
        y="defocusV",
        data=df,
        ax=ax[1],
        hue="ugraph_filename",
        palette=palette,
        legend=False,
        marker="+",
    )
    # add identity line
    min_defocus_truth = df["defocus_truth"].min()
    max_defocus_truth = df["defocus_truth"].max()
    ax[0].plot(
        [min_defocus_truth, max_defocus_truth],
        [min_defocus_truth, max_defocus_truth],
        color="black",
        linestyle="--",
        alpha=0.5,
    )
    ax[1].plot(
        [min_defocus_truth, max_defocus_truth],
        [min_defocus_truth, max_defocus_truth],
        color="black",
        linestyle="--",
        alpha=0.5,
    )

    # add labels
    ax[0].set_xlabel("defocus truth [$\u212B$]")
    ax[1].set_xlabel("defocus truth [$\u212B$]")
    ax[0].set_ylabel("defocusU estimated [$\u212B$]")
    ax[0].set_title("defocusU")
    ax[1].set_title("defocusV")
    ax[0].grid(False)
    ax[1].grid(False)
    # add colorbar legend
    sm = plt.cm.ScalarMappable(
        cmap=palette,
        norm=plt.Normalize(
            vmin=0, vmax=len(np.unique(df["ugraph_filename"])) - 1
        ),
    )
    sm._A = []
    cbar = plt.colorbar(sm)
    cbar.set_label("micrograph")
    fig.tight_layout()
    return fig, ax


def _relativistic_lambda(voltage):
    # returns the relativistic wavelength in Angstrom
    # voltage should be in volts
    return 12.2643247 / np.sqrt(voltage * (1 + voltage * 0.978466e-6))


def _simulate_CTF_curve(defocus, amp, Cs, B, voltage, k):
    wavelength = _relativistic_lambda(voltage)
    gamma = (-np.pi / 2) * Cs * np.power(wavelength, 3) * np.power(
        k, 4
    ) + np.pi * wavelength * defocus * np.power(k, 2)
    CTF = -1 * np.sin(amp + gamma)
    if B != 0:
        CTF *= np.exp(-B * k**2)

    return CTF


def _convert_1d_ctf_to_2d_ctf(ctf_1d, freq_1d, freq_2d):
    ctf_interpolate = interpolate.interp1d(
        freq_1d, ctf_1d, fill_value="extrapolate"
    )
    ctf_2d = ctf_interpolate(freq_2d)
    return ctf_2d


def plot_CTF(
    df_picked,
    metadata_filename,
    df_truth,
    mrc_dir,
    ugraph_index=0,
    amp=0.1,
    Cs=2.7,
    Bfac=0,
    kV=300,
):
    # group the picked data frame
    if metadata_filename is not None:
        if isinstance(metadata_filename, list):
            metadata_filename = metadata_filename[0]
        df_picked = df_picked.groupby("metadata_filename").get_group(
            metadata_filename
        )

    # get the micrograph name
    ugraph_filename = np.unique(df_picked["ugraph_filename"])[ugraph_index]
    print(f"plotted index {ugraph_index}; micrograph: {ugraph_filename}")

    ugraph_path = os.path.join(mrc_dir, ugraph_filename)
    ugraph = mrcfile.open(ugraph_path).data[0, :, :]
    ugraph_ft = np.fft.fftshift(np.fft.fft2(ugraph))
    magnitude_spectrum = 20 * np.log(np.abs(ugraph_ft))

    # for contrast, only plot the middle 80% of the spectrum
    vmin = np.nanpercentile(magnitude_spectrum, 10)
    vmax = np.nanpercentile(magnitude_spectrum, 90)

    # get the CTF values for the micrograph from the dataframe
    data_ugraph = df_picked.groupby("ugraph_filename").get_group(
        ugraph_filename
    )
    defocusU = np.array(data_ugraph["defocusU"])[
        0
    ]  # assume all particles have the same defocusU

    # extract the frequency values from the micrograph
    rows, cols = ugraph.shape
    freq_rows = np.fft.fftfreq(rows, d=1)
    freq_cols = np.fft.fftfreq(cols, d=1)
    mesh_freq_cols, mesh_freq_rows = np.meshgrid(freq_cols, freq_rows)
    mesh_freq = np.sqrt(mesh_freq_cols**2 + mesh_freq_rows**2)

    # compute the 1D CTF curve
    ctf_estimated_1D = _simulate_CTF_curve(
        defocusU, amp, Cs, Bfac, kV, freq_rows
    )

    # compute the 2D CTF curve
    ctf_estimated_2D, freq_2d = _convert_1d_ctf_to_2d_ctf(
        ctf_estimated_1D,
        freq_rows,
        mesh_freq,
    )
    ctf_estimated_2D = np.abs(ctf_estimated_2D)

    # x = np.linspace(0, 1, ctf_estimated_2D.shape[0])
    # X, Y = np.meshgrid(freq, freq)
    # # crop out the centre
    # ctf_estimated_2D = ctf_estimated_2D[
    #     ctf_estimated_2D.shape[0] // 4 : -ctf_estimated_2D.shape[0] // 4,
    #     ctf_estimated_2D.shape[1] // 4 : -ctf_estimated_2D.shape[1] // 4,
    # ]
    # circle_mask = ((X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 0.125).astype(int)
    # circle_mask = (freq_2d <= 0.375).astype(int)
    # ctf_estimated_2D *= circle_mask
    # # turn the first 3 quadrants to nan
    # ctf_estimated_2D[ctf_estimated_2D.shape[0] // 2 :, :] = np.nan
    # ctf_estimated_2D[:, ctf_estimated_2D.shape[1] // 2 :] = np.nan
    # ctf_estimated_2D[ctf_estimated_2D == 0] = np.nan
    vmin_ctf = np.nanpercentile(ctf_estimated_2D, 5) * 0.5
    vmax_ctf = np.nanpercentile(ctf_estimated_2D, 99.99) * 1.5

    # get the truth values
    defocus_truth = np.abs(
        df_truth.groupby("ugraph_filename")
        .get_group(ugraph_filename)["defocus"]
        .values[0]
    )

    ctf_truth_1d = _simulate_CTF_curve(
        defocus_truth,
        amp,
        Cs,
        Bfac,
        kV,
        max_freq=0.05,
        num_points=2000 // 2,
    )
    ctf_truth_2d, _ = _convert_1d_ctf_to_2d_ctf(ctf_truth_1d[1])
    ctf_truth_2d = np.abs(ctf_truth_2d)
    x = np.linspace(0, 1, ctf_truth_2d.shape[0])
    X, Y = np.meshgrid(x, x)
    circle_mask = ((X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 0.125).astype(int)
    ctf_truth_2d *= circle_mask
    # turn the first 3 quadrants to nan
    ctf_truth_2d[ctf_truth_2d.shape[0] // 2 :, :] = np.nan
    ctf_truth_2d[:, : ctf_truth_2d.shape[1] // 2] = np.nan
    ctf_truth_2d[ctf_truth_2d == 0] = np.nan
    # vmin_ctf_truth = np.nanpercentile(ctf_truth_2d, 5) * 0.5
    # vmax_ctf_truth = np.nanpercentile(ctf_truth_2d, 99.99) * 1.5

    # plot the power spectrum
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(magnitude_spectrum, cmap="gray", vmin=vmin, vmax=vmax)
    ax.imshow(ctf_estimated_2D, cmap="gray", vmin=vmin_ctf, vmax=vmax_ctf)
    # ax.imshow(
    #     ctf_truth_2d, cmap="gray", vmin=vmin_ctf_truth, vmax=vmax_ctf_truth
    # )
    # add text to show the estimated and truth values in units of Angstrom
    ax.text(
        0.05,
        0.95,
        s=f"defocusU: {defocusU:.2f} $\u212B$",
        fontsize=20,
        color="white",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax.text(
        0.55,
        0.95,
        s=f"true defocus: {defocus_truth:.2f} $\u212B$",
        fontsize=20,
        color="white",
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
    )
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax


def main(args):
    """The script loads the metadata file and extracts the ctf parameters
    for each particle. It is possible that the CTF values for each particle
    are the same if the CTF estimation was done on the entire micrograph.
    In that case, the script will plot the CTF values for each micrograph.
    Next, the script loads the config parakeet file and gets the CTF values
    for each micrograph.
    For each micrograph that is present in the metadata, the script will plot
    the estimated CTF values against the ground-truth CTF values.
    """

    # create output directory if it does not exist
    if not os.path.isdir(args.plot_dir):
        os.makedirs(args.plot_dir)

    # verbose outputs
    if args.verbose:
        tt = time.time()
        print("loading particles ...")

    # load data from file(s)
    mrc_dir = args.mrc_dir if args.mrc_dir else args.config_dir
    analysis = load_data(
        args.meta_file,
        args.config_dir,
        particle_diameter=100,
        verbose=args.verbose,
        enable_tqdm=args.tqdm,
    )
    df_picked = pd.DataFrame(analysis.results_picking)
    df_truth = pd.DataFrame(analysis.results_truth)

    # only include first --num_ugraphs micrographs
    # select subset of ugraphs from truth df
    if args.num_ugraphs:
        ugraph_identifiers = sorted(df_truth["ugraph_filename"].unique())[
            : args.num_ugraphs
        ]

        # remove all rows not corresponding to selected ugraphs
        df_picked = df_picked[
            df_picked["ugraph_filename"].isin(ugraph_identifiers)
        ]
        df_truth = df_truth[
            df_truth["ugraph_filename"].isin(ugraph_identifiers)
        ]

    if args.verbose:
        print(
            "Loaded {} particles from {}. starting plotting ...".format(
                len(df_picked), args.meta_file
            )
        )
        print(f"time taken: {time.time()-tt:.2f} seconds")

    # create the plots, which are:
    # 1. single scatter plot of estimated vs truth defoci
    # 2. per micrograph ctf plots
    for plot_type in args.plot_types:
        if plot_type.lower() == "scatter":
            if args.verbose:
                tt = time.time()
                print("Plotting defocus scatter plot ...")
            defocus_scatter = plotDefocusScatter(
                meta_file=args.meta_file,
                plot_dir=args.plot_dir,
                dpi=args.dpi,
                pdf=args.pdf,
            )
            defocus_scatter.setup_plot_data(
                df_truth,
                df_picked,
            )
            defocus_scatter.make_and_save_plots(overwrite_data=True)
            if args.verbose:
                print(f"Time taken: {time.time()-tt:.2f} seconds")

        if plot_type.lower() == "per-particle-scatter":
            # limiting num_ugraphs is done in the function below
            if args.verbose:
                tt = time.time()
                print("Plotting defocus per-particle scatter plot ...")

            # extract the group from the picked data frame
            metadata_filename = args.meta_file
            if isinstance(metadata_filename, list):
                metadata_filename = metadata_filename[0]
            df_picked_groupby = df_picked.groupby(
                "metadata_filename"
            ).get_group(metadata_filename)

            # used for plotting
            results = {
                "ugraph_filename": [],
                "defocusU": [],
                "defocusV": [],
                "defocus_truth": [],
            }

            # match the picked particles to closest truth particle
            mp_df, mt_df, _, _ = analysis._match_particles(
                metadata_filename,
                df_picked_groupby,
                df_truth,
                verbose=False,
            )

            # only use the successfully matched particles
            results["ugraph_filename"].extend(
                mp_df["ugraph_filename"].tolist()
            )
            results["defocusU"].extend(mp_df["defocusU"].tolist())
            results["defocusV"].extend(mp_df["defocusV"].tolist())
            # combine the particle position with the ugraph defocus value!
            results["defocus_truth"].extend(
                (mt_df["defocus"] + mt_df["position_z"]).abs().tolist()
            )
            """
            could add a functionality to save or print matched particles
            for i in range(len(results["ugraph_filename"])):
                print(
                    "{}\t{}\t{}\t{}".format(
                        results["ugraph_filename"][i],
                        results["defocusU"][i],
                        results["defocusV"][i],
                        results["defocus_truth"][i],
                    )
                )
            """
            pp_df = pd.DataFrame(results)
            per_particle_defocus = plotPerParticleDefocusScatter(
                meta_file=args.meta_file,
                plot_dir=args.plot_dir,
                dpi=args.dpi,
                pdf=args.pdf,
            )
            per_particle_defocus.setup_plot_data(
                df_truth,
                df_picked,
                mp_df,
                mt_df,
                pp_df,
            )
            per_particle_defocus.make_and_save_plots(overwrite_data=True)
            if args.verbose:
                print(f"Time taken: {time.time()-tt:.2f} seconds")

        if plot_type.lower() == "ctf":
            if args.num_ugraphs is None:
                print("Plotting CTF for all micrographs ...")
            else:
                print(f"Plotting CTF for {args.num_ugraphs} micrographs ...")

            for ugraph_index in np.unique(df_picked["ugraph_filename"])[
                : args.num_ugraphs
            ]:
                if args.verbose:
                    tt = time.time()
                    print(f"plotting CTF for micrograph {ugraph_index} ...")
                ugraph_index = int(ugraph_index.strip(".mrc"))
                filename = os.path.join(
                    args.plot_dir, f"ctf_{ugraph_index}.png"
                )

                fig, ax = plot_CTF(
                    df_picked=df_picked,
                    metadata_filename=args.meta_file,
                    df_truth=df_truth,
                    mrc_dir=mrc_dir,
                    ugraph_index=ugraph_index,
                )
                fig.savefig(filename, dpi=args.dpi, bbox_inches="tight")
                if args.pdf:
                    fig.savefig(
                        filename.replace(".png", ".pdf"),
                        bbox_inches="tight",
                    )
                plt.close(fig)
                if args.verbose:
                    print(f"time taken: {time.time()-tt:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
