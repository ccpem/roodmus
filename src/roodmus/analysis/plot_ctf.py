"""Script to plot a comparison between the estimated CTF parameters and the
true values used in data generation
"""

import argparse
import os
import time

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import mrcfile

from .analyse_ctf import ctf_estimation


def add_arguments(parser):
    parser.add_argument(
        "--config-dir",
        help="Directory with .mrc files and .yaml config files",
        type=str,
    )
    parser.add_argument(
        "--mrc-dir",
        help=(
            "Directory with .mrc files. Assumed to be the same as"
            " 'config-dir'by default",
        ),
        type=str,
        default=None,
    )
    parser.add_argument(
        "-N",
        "--num-ugraphs",
        help="Number of micrographs to consider in analyses. Default 'all'",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--meta-file",
        help=(
            "Particle metadata file. Can be .star (RELION) or .cs"
            " (CryoSPARC)"
        ),
        type=str,
    )
    parser.add_argument(
        "--plot-dir",
        help="output file name",
        type=str,
        default="ctf.png",
    )
    parser.add_argument(
        "--plot-types",
        help="types of plots to generate",
        type=str,
        nargs="+",
        default=["scatter"],
        choices=["scatter", "CTF"],
    )
    parser.add_argument(
        "--verbose", help="increase output verbosity", action="store_true"
    )
    return parser


def get_name():
    return "plot_ctf"


def plot_defocus_scatter(df, palette="BuGn"):
    # df_grouped = df.groupby("ugraph_filename")

    # plot the results
    plt.rcParams["font.size"] = 24
    plt.style.use("seaborn-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 10))
    # for name, group in df_grouped:
    #     ax.scatter(group["defocus_truth"], group["defocusU"], label=name)
    sns.scatterplot(
        x="defocus_truth",
        y="defocusU",
        data=df,
        ax=ax,
        hue="ugraph_filename",
        palette=palette,
        legend=False,
        marker="+",
    )
    # add identity line
    min_defocusU_truth = df["defocus_truth"].min()
    max_defocusU_truth = df["defocus_truth"].max()
    ax.plot(
        [min_defocusU_truth, max_defocusU_truth],
        [min_defocusU_truth, max_defocusU_truth],
        color="black",
        linestyle="--",
        alpha=0.5,
    )
    # add labels
    ax.set_xlabel("defocus truth [$\u212B$]")
    ax.set_ylabel("defocusU estimated [$\u212B$]")
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
    return fig, ax


def _relativistic_lambda(voltage):
    return 12.2643247 / np.sqrt(voltage * (1 + voltage * 0.978466e-6))


def _simulate_CTF_curve(
    defocus, amp, Cs, B, voltage, max_freq=0.5, num_points=1024 // 2
):
    k = np.linspace(0, max_freq, num_points)
    wavelength = _relativistic_lambda(voltage)
    gamma = (-np.pi / 2) * Cs * np.power(wavelength, 3) * np.power(
        k, 4
    ) + np.pi * wavelength * defocus * np.power(k, 2)
    CTF = -1 * np.sin(amp + gamma)
    if B != 0:
        CTF *= np.exp(-B * k**2)

    return k, CTF


def _convert_1d_ctf_to_2d_ctf(ctf_1d):
    from scipy import interpolate

    n = len(ctf_1d)
    apix = 1
    freq = np.linspace(1 / (n * apix), 1 / (2 * apix), n, endpoint=True)
    assert len(freq) == len(
        ctf_1d
    ), "Frequency and CTF must have the same size"
    ctf_interpolate = interpolate.interp1d(
        freq, ctf_1d, fill_value="extrapolate"
    )

    freq_x, freq_y = np.meshgrid(
        np.linspace(-1 / (2 * apix), 1 / (2 * apix), 2 * n, endpoint=True),
        np.linspace(-1 / (2 * apix), 1 / (2 * apix), 2 * n, endpoint=True),
    )
    # generate a 2d CTF grid
    freq_2d = np.sqrt(freq_x**2 + freq_y**2)
    # mask freq_2d
    circular_mask = (freq_2d <= 1).astype(int)
    freq_2d = freq_2d * circular_mask
    ctf_2d = ctf_interpolate(freq_2d)
    # apply the CTF to the image
    return ctf_2d


def plot_CTF(df, mrc_dir, ugraph_index=0):
    # get the micrograph name
    ugraph_filename = np.unique(df["ugraph_filename"])[ugraph_index]
    print(f"plotted index {ugraph_index}; micrograph: {ugraph_filename}")
    ugraph_path = os.path.join(mrc_dir, ugraph_filename)
    ugraph = mrcfile.open(ugraph_path).data[0, :, :]

    ugraph_ft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(ugraph))))
    L = ugraph_ft.shape[0] // 4
    ugraph_ft_crop = ugraph_ft[L:-L, L:-L]
    vmin = np.percentile(ugraph_ft_crop, 5)
    vmax = np.percentile(ugraph_ft_crop, 99.99)

    # get the CTF values for the micrograph from the dataframe
    data_ugraph = df.groupby("ugraph_filename").get_group(ugraph_filename)
    defocusU = np.array(data_ugraph["defocusU"])[
        0
    ]  # assume all particles have the same defocusU
    amp = np.array(data_ugraph["amp"])[
        0
    ]  # assume all particles have the same amp
    Cs = np.array(data_ugraph["Cs"])[
        0
    ]  # assume all particles have the same Cs
    Bfac = np.array(data_ugraph["Bfac"])[
        0
    ]  # assume all particles have the same Bfac
    kV = np.array(data_ugraph["kV"])[
        0
    ]  # assume all particles have the same kV

    ctf_estimated_1D = _simulate_CTF_curve(
        defocusU, amp, Cs, Bfac, kV, max_freq=0.05, num_points=2000 // 2
    )
    ctf_estimated_2D = _convert_1d_ctf_to_2d_ctf(ctf_estimated_1D[1])
    ctf_estimated_2D = np.abs(ctf_estimated_2D)
    x = np.linspace(0, 1, ctf_estimated_2D.shape[0])
    X, Y = np.meshgrid(x, x)
    circle_mask = ((X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 0.125).astype(int)
    ctf_estimated_2D *= circle_mask
    # turn the first 3 quadrants to nan
    ctf_estimated_2D[ctf_estimated_2D.shape[0] // 2 :, :] = np.nan
    ctf_estimated_2D[:, ctf_estimated_2D.shape[1] // 2 :] = np.nan
    ctf_estimated_2D[ctf_estimated_2D == 0] = np.nan
    vmin_ctf = np.nanpercentile(ctf_estimated_2D, 5) * 0.5
    vmax_ctf = np.nanpercentile(ctf_estimated_2D, 99.99) * 1.5

    # get the truth values
    defocus_truth = np.array(data_ugraph["defocus_truth"])[
        0
    ]  # assume all particles have the same defocus
    amp_truth = np.zeros_like(
        defocus_truth
    )  # amplitude is 0 for all particles
    Cs_truth = np.array(data_ugraph["Cs_truth"])[
        0
    ]  # assume all particles have the same Cs
    Bfac_truth = np.zeros_like(
        defocus_truth
    )  # B-factor is 0 for all particles
    kV_truth = np.array(data_ugraph["kV_truth"])[
        0
    ]  # assume all particles have the same kV

    ctf_truth_1d = _simulate_CTF_curve(
        defocus_truth,
        amp_truth,
        Cs_truth,
        Bfac_truth,
        kV_truth,
        max_freq=0.05,
        num_points=2000 // 2,
    )
    ctf_truth_2d = _convert_1d_ctf_to_2d_ctf(ctf_truth_1d[1])
    ctf_truth_2d = np.abs(ctf_truth_2d)
    x = np.linspace(0, 1, ctf_truth_2d.shape[0])
    X, Y = np.meshgrid(x, x)
    circle_mask = ((X - 0.5) ** 2 + (Y - 0.5) ** 2 <= 0.125).astype(int)
    ctf_truth_2d *= circle_mask
    # turn the first 3 quadrants to nan
    ctf_truth_2d[ctf_truth_2d.shape[0] // 2 :, :] = np.nan
    ctf_truth_2d[:, : ctf_truth_2d.shape[1] // 2] = np.nan
    ctf_truth_2d[ctf_truth_2d == 0] = np.nan
    vmin_ctf_truth = np.nanpercentile(ctf_truth_2d, 5) * 0.5
    vmax_ctf_truth = np.nanpercentile(ctf_truth_2d, 99.99) * 1.5

    # plot the power spectrum
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(ugraph_ft_crop, cmap="gray", vmin=vmin, vmax=vmax)
    ax.imshow(ctf_estimated_2D, cmap="gray", vmin=vmin_ctf, vmax=vmax_ctf)
    ax.imshow(
        ctf_truth_2d, cmap="gray", vmin=vmin_ctf_truth, vmax=vmax_ctf_truth
    )
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
    if not os.path.exists(args.plot_dir):
        os.mkdir(args.plot_dir)

    if args.verbose:
        tt = time.time()
        print("loading particles ...")
    mrc_dir = args.mrc_dir if args.mrc_dir else args.config_dir
    picked_particles = ctf_estimation(
        args.meta_file, args.config_dir, verbose=args.verbose
    )
    picked_particles = pd.DataFrame(picked_particles.results)
    if args.verbose:
        print(
            "Loaded {} particles from {}. starting plotting ...".format(
                len(picked_particles), args.meta_file
            )
        )
        print(f"time taken: {time.time()-tt:.2f} seconds")

    # create the plots
    for plot_type in args.plot_types:
        if plot_type == "scatter":
            if args.verbose:
                tt = time.time()
                print("Plotting defocus scatter plot ...")
            filename = os.path.join(args.plot_dir, "ctf_scatter.png")
            fig, ax = plot_defocus_scatter(picked_particles)
            fig.savefig(filename, dpi=300, bbox_inches="tight")
            plt.close(fig)
            if args.verbose:
                print(f"Time taken: {time.time()-tt:.2f} seconds")

        if plot_type == "CTF":
            if args.num_ugraphs is None:
                print("Plotting CTF for all micrographs ...")
            else:
                print(f"Plotting CTF for {args.num_ugraphs} micrographs ...")

            for ugraph_index in np.unique(picked_particles["ugraph_filename"])[
                : args.num_ugraphs
            ]:
                if args.verbose:
                    tt = time.time()
                    print(f"plotting CTF for micrograph {ugraph_index} ...")
                ugraph_index = int(ugraph_index.strip(".mrc"))
                filename = os.path.join(
                    args.plot_dir, f"ctf_{ugraph_index}.png"
                )
                fig, ax = plot_CTF(picked_particles, mrc_dir, ugraph_index)
                fig.savefig(filename, dpi=300, bbox_inches="tight")
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
