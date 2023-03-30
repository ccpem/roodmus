import os
import argparse

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from roodmus.analysis.analyse_ctf import ctf_estimation
from roodmus.analysis.plot_ctf import plot_CTF, plot_defocus_scatter
from roodmus.analysis.analyse_picking import particle_picking
from roodmus.analysis.plot_picking import (
    label_micrograph_picked,
    label_micrograph_truth,
    label_micrograph_truth_and_picked,
    plot_precision,
    plot_recall,
    plot_boundary_investigation,
    plot_overlap_investigation,
)


def load_paths():
    config_dir = "/mnt/parakeet_storage4/roodmus_tests_data/mrc"
    meta_file = (
        "/mnt/parakeet_storage4/roodmus_tests_data/cryoSPARC/"
        "J293_picked_particles.cs"
    )

    # figures may new new
    plot_dir = "/mnt/parakeet_storage4/roodmus_tests_data/figures"
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    return config_dir, meta_file, plot_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--verbose",
        help="Output increased verbosity",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--ugraph_index",
        help="Micrograph index to use for testing. Defaults to 0",
        type=int,
        default=0,
        required=False,
    )

    parser.add_argument(
        "--ugraph_x_pixels",
        help=("Pixels along x axis of ugraph"),
        type=int,
        default=4000,
        required=False,
    )

    parser.add_argument(
        "--ugraph_y_pixels",
        help=("Pixels along x axis of ugraph"),
        type=int,
        default=4000,
        required=False,
    )

    parser.add_argument(
        "--ang_per_pixel",
        help=("Angstroms per pixel"),
        type=float,
        default=1.0,
        required=False,
    )

    parser.add_argument(
        "--particle_diameter",
        help=("Particle diameter in A. Defaults to 100"),
        type=float,
        default=100,
        required=False,
    )

    parser.add_argument(
        "--analyse_ctf_estimation",
        help="load and print ctf estimation data(frame)",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--particle_labelling",
        help="label and save picked/truth particle locations on image",
        default=False,
        action="store_true",
    )

    args = parser.parse_args()

    # load/set the default (minimal) parameters
    config_dir, meta_file, plot_dir = load_paths()

    # testing individual utilities (cells) in the notebook

    # look at the ctf_estimation df
    if args.analyse_ctf_estimation:
        analysis_ctf = ctf_estimation(
            meta_file, config_dir, verbose=args.verbose
        )
        analysis_ctf.compute()
        df_ctf = pd.DataFrame(analysis_ctf.results)
        print(df_ctf)

        # plot defocus scatter
        fig, ax = plot_defocus_scatter(df_ctf)
        plt.rcParams["font.size"] = 15
        plt.savefig(os.path.join(plot_dir, "defocus_scatter.png"))
        plt.savefig(os.path.join(plot_dir, "defocus_scatter.pdf"))
        plt.clf()

        # plot exaple ctf
        fig, ax = plot_CTF(df_ctf, config_dir, args.ugraph_index)
        plt.savefig(
            os.path.join(plot_dir, "ctf_plot_{}.png".format(args.ugraph_index))
        )
        plt.savefig(
            os.path.join(plot_dir, "ctf_plot_{}.pdf".format(args.ugraph_index))
        )
        plt.clf()

        # find max mismatch ctf image and save it
        delta_defocus = 0.0
        for _, row in df_ctf.iterrows():
            defocusU = row["defocusU"]
            defocus_truth = row["defocus_truth"]
            ab_dif = np.abs(defocusU - defocus_truth)
            if ab_dif > delta_defocus:
                delta_defocus = ab_dif
                max_error_index = int(row["ugraph_filename"].strip(".mrc"))

        fig, ax = plot_CTF(df_ctf, config_dir, max_error_index)
        plt.savefig(os.path.join(plot_dir, "ctf_max_mismatch.png"))
        plt.savefig(os.path.join(plot_dir, "ctf_max_mismatch.pdf"))
        plt.clf()

        print("Finished minimal ctf estimation analysis")

    if args.particle_labelling:
        # choose the default values
        meta_files = [meta_file]
        jobtypes = {meta_file: "blob picking"}
        particle_diameter = args.particle_diameter
        ugraph_shape = (args.ugraph_x_pixels, args.ugraph_y_pixels)

        # load up the picked particles
        analysis_picking = particle_picking(
            meta_file,
            config_dir,
            particle_diameter,
            ugraph_shape=ugraph_shape,
            verbose=args.verbose,
        )

        # print the picked adn truth particles in df
        df_picked = pd.DataFrame(analysis_picking.results_picking)
        df_truth = pd.DataFrame(analysis_picking.results_truth)

        print("Picked particles: {}".format(df_picked))
        print("\n\nTruth particles: {}\n\n".format(df_truth))

        # plot the picked particles
        metadata_index = args.ugraph_index
        fig, ax = label_micrograph_picked(
            df_picked.groupby("metadata_filename").get_group(
                meta_files[metadata_index]
            ),
            args.ugraph_index,
            config_dir,
            box_width=particle_diameter,
            box_height=particle_diameter,
            verbose=args.verbose,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.set_size_inches(7, 7)
        plt.savefig(os.path.join(plot_dir, "picked_particles.png"))
        plt.savefig(os.path.join(plot_dir, "picked_particles.pdf"))
        plt.clf()

        # plot the truth particles
        fig, ax = label_micrograph_truth(
            df_truth,
            args.ugraph_index,
            config_dir,
            box_width=particle_diameter,
            box_height=particle_diameter,
            verbose=args.verbose,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.set_size_inches(7, 7)
        plt.savefig(os.path.join(plot_dir, "truth_particles.png"))
        plt.savefig(os.path.join(plot_dir, "truth_particles.pdf"))
        plt.clf()

        # plot both sets of particles
        fig, ax = label_micrograph_truth_and_picked(
            df_picked.groupby("metadata_filename").get_group(
                meta_files[metadata_index]
            ),
            df_truth,
            args.ugraph_index,
            config_dir,
            box_width=particle_diameter,
            box_height=particle_diameter,
            verbose=args.verbose,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        fig.tight_layout()
        fig.set_size_inches(7, 7)
        plt.savefig(os.path.join(plot_dir, "picked_with_truth.png"))
        plt.savefig(os.path.join(plot_dir, "picked_with_truth.pdf"))
        plt.clf()

        # need to copy over my plotting of unpicked
        # particles utility to roodmus

        # determine (calculate) and show the picking precision
        df_precision, df_picked = analysis_picking.compute_precision(
            df_picked, df_truth, verbose=args.verbose
        )
        print("Picking precision: {}".format(df_precision))

        # make boxplots for precision and recall
        fig, ax = plot_precision(df_precision, jobtypes)
        fig.set_size_inches([10, 10])
        plt.savefig(os.path.join(plot_dir, "precision_boxplot.png"))
        plt.savefig(os.path.join(plot_dir, "precision_boxplot.pdf"))
        plt.clf()
        fig, ax = plot_recall(df_precision, jobtypes)
        fig.set_size_inches([10, 10])
        plt.savefig(os.path.join(plot_dir, "recall_boxplot.png"))
        plt.savefig(os.path.join(plot_dir, "recall_boxplot.pdf"))
        plt.clf()

        # make boxplot of both precision and recall (for each 'method')
        df = df_precision.melt(
            id_vars=[
                "metadata_filename",
                "ugraph_filename",
                "defocus",
                "TP",
                "FP",
                "FN",
                "multiplicity",
                "num_particles_picked",
                "num_particles_truth",
                "class2D",
            ]
        )
        plt.rcParams["font.size"] = 10
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.boxplot(
            x="metadata_filename",
            y="value",
            data=df,
            ax=ax,
            fliersize=0,
            palette="RdYlBu",
            hue="variable",
        )
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticklabels([jobtypes[meta_file] for meta_file in meta_files])
        plt.setp(
            ax.get_xticklabels(),
            rotation=45,
            ha="right",
            rotation_mode="anchor",
        )
        ax.legend().set_visible(False)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=1,
            bbox_to_anchor=(1.1, 0.85),
        )
        fig.tight_layout()
        plt.savefig(
            os.path.join(plot_dir, "precision_with_recall_boxplot.png")
        )
        plt.savefig(
            os.path.join(plot_dir, "precision_with_recall_boxplot.pdf")
        )
        plt.clf()

        # plot the distribution of the particles in the
        # ugraphs in x, y, and z directions
        bin_width = [100, 100, 10]
        axis = ["x", "y", "z"]
        metadata_filename = meta_files[metadata_index]
        for a, bnwdth in zip(axis, bin_width):
            fig, ax = plot_boundary_investigation(
                df_truth, df_picked, metadata_filename, bnwdth, axis=a
            )
            plt.savefig(
                os.path.join(
                    plot_dir, "particle_{}_distribution.png".format(a)
                )
            )
            plt.savefig(
                os.path.join(
                    plot_dir, "particle_{}_distribution.pdf".format(a)
                )
            )
            plt.clf()

        # compute the overlap between picked and truth particles
        df_overlap = analysis_picking.compute_overlap(
            df_picked, df_truth, verbose=args.verbose
        )
        print("Particle overlaps: {}".format(df_overlap))

        # plot the overlaps
        fig, ax = plot_overlap_investigation(
            df_overlap, metadata_filename, jobtypes=jobtypes
        )
        plt.savefig(os.path.join(plot_dir, "particle_overlaps.png"))
        plt.savefig(os.path.join(plot_dir, "particle_overlaps.pdf"))
        plt.clf()

        # plot the distribution of trajectory frames in a metadata file
        df_picked["closest_pdb_index"] = df_picked["closest_pdb"].apply(
            lambda x: int(x.split("_")[-1].split(".")[0])
        )
        df_picked.loc[
            df_picked["closest_dist"] > particle_diameter, "closest_pdb_index"
        ] = np.nan
        df_truth["pdb_index"] = df_truth["pdb_filename"].apply(
            lambda x: int(x.split("_")[-1].split(".")[0])
        )
        plt.rcParams["font.size"] = 10
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.histplot(
            df_picked.groupby("metadata_filename").get_group(
                meta_files[metadata_index]
            )["closest_pdb_index"],
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
        ax.set_title(jobtypes[meta_files[metadata_index]])
        fig.tight_layout()
        fig.legend(
            ["picked", "truth"],
            loc="lower center",
            ncol=1,
            bbox_to_anchor=(1.1, 0.85),
        )
        plt.savefig(os.path.join(plot_dir, "frame_distribution.png"))
        plt.savefig(os.path.join(plot_dir, "frame_distribution.pdf"))
        plt.clf()

        print("Finished minimal particle labelling analysis")


if __name__ == "__main__":
    main()
