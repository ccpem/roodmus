"""Load in 2 or more pkl'ed sklearn cluster objects and compute metrics to
compare the performance of each to the other. Save results in a human-readable
.csv file

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
import pickle
import os
import random

import numpy as np
import pandas as pd
from MDAnalysis.analysis.encore.clustering.ClusteringMethod import (
    AffinityPropagationNative,
)
from MDAnalysis.analysis.encore.dimensionality_reduction.DimensionalityReductionMethod import (  # noqa: E501
    StochasticProximityEmbeddingNative,
)
import MDAnalysis as mda
import seaborn as sns
import matplotlib.pyplot as plt

from roodmus.analysis.utils import load_data
from roodmus.heterogeneity.het_metrics import (
    get_pdb_list,
    select_confs,
)

# import os
# import pandas as pd
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import jensenshannon, directed_hausdorff

# look for appropriate clustering metrics (not necessarily required)


def add_arguments(parser: argparse.ArgumentParser):
    """Parse arguments

    Args:
        parser (argparse.ArgumentParser): _description_

    Returns:
        _type_: _description_
    """
    parser.add_argument(
        "--config_dir",
        help="Directory with .mrc files and .yaml config files",
        type=str,
    )

    parser.add_argument(
        "--meta_file",
        help=(
            "Particle metadata file. Can be .star (RELION) or .cs (CryoSPARC)"
        ),
        type=str,
    )

    parser.add_argument(
        "--conformations_dir",
        help=("Directory with .pdb files of the conformations"),
        type=str,
    )

    parser.add_argument(
        "--clusters",
        "-c",
        help=".pkl file containing clustered conformations."
        " Provide 2 pkls with clusters so similarity can be"
        " calculated!",
        nargs="+",
        type=str,
        default=[""],
    )

    parser.add_argument(
        "--clusters_source",
        "-ct",
        help="Whether .pkl file contained clustered conformations from"
        " MD trajectory or from reconstruction metadata. Use"
        " `--clusters_source MD reco` for an MD and reconstruction-derived"
        " .pkl file respectively. Provide 1 per .pkl file in the correct"
        " order.",
        nargs="+",
        type=str,
        default=[""],
    )

    # TODO remove this
    parser.add_argument(
        "--js",
        help="Compute jensen-shannon metric between ensembles",
        action="store_true",
    )

    # TODO remove this
    parser.add_argument(
        "--hd",
        help="Compute hausdorff metric between ensembles",
        action="store_true",
    )

    parser.add_argument(
        "--verbose", help="increase output verbosity", action="store_true"
    )

    parser.add_argument(
        "--output_dir",
        help="Directory to save results and intermediate results",
        type=str,
        default="het_metrics",
        required=False,
    )

    parser.add_argument(
        "--digits",
        help=(
            "Number of digits (including leading zeros)"
            " in the conformation filenames"
        ),
        type=int,
        default=6,
        required=False,
    )

    parser.add_argument(
        "--file_ext",
        help="File extension of the conformation files. Default is .pdb",
        type=str,
        default=".pdb",
    )

    parser.add_argument(
        "--timestep",
        help="Time step between conformations (ps)."
        " Default is 1 picosecond",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--time_offset",
        help="Time offset for first conformation (ps)."
        " Default is 0 picosecond",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--save_rmsd",
        help="Save the RMSD matrix in npz format for future use",
        action="store_true",
    )

    parser.add_argument(
        "--rmsd_precalc",
        help="RMSD for these conformations has been precalculated for reuse."
        " Takes precedence over recalculation of RMSD",
        type=str,
        default="",
        required=False,
    )

    parser.add_argument(
        "--preference",
        help="Preference value to use for affinity propagation. Default is"
        " -1. Rough range to explore is -100. to -1.",
        type=float,
        default=-1.0,
    )

    parser.add_argument(
        "--dimensions",
        help="Dimensionality to use for stochastic proximity embedding in"
        " DRES. Defaults to 3",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--estimate_error",
        help="Estimate error on JS-divergence",
        action="store_true",
    )

    parser.add_argument(
        "--bootstrapping_samples",
        help="How many times to sample whilst bootstrapping JS-divergence"
        " calculations (estimating error). Defaults to 10",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--ncores",
        help="How many CPU cores to use for (RMSD) calculations",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--select",
        help="choice of atoms to compute RMSD using. Default is name CA`",
        type=str,
        default="name CA",
    )

    parser.add_argument(
        "--dpi",
        help="choose dots per inch in png plots, Default is 100",
        type=int,
        default=100,
        required=False,
    )

    parser.add_argument("--pdf", help="save plot as pdf", action="store_true")

    parser.add_argument(
        "--pkl_universes",
        help="Save the mda.Universes in the pkl file. WARNING:"
        " may make pkl VERY large",
        action="store_true",
    )

    parser.add_argument(
        "--n_confs",
        help="Limit to <n_confs> conformations from conformations_dir",
        type=int,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--contiguous_confs",
        help="Set the sampled conformations to be contiguous instead of"
        " uniformly sampled in time",
        action="store_true",
    )

    parser.add_argument(
        "--first_conf",
        help="Set an index (used after alphanumeric sorting) to set first"
        " conformation to be sampled from",
        type=int,
        default=0,
        required=False,
    )

    parser.add_argument(
        "--overwrite",
        help="Overwrite previous pkl files with same filepath",
        action="store_true",
    )

    parser.add_argument(
        "--ensemble_subset_size",
        help="Ensemble subset size limit to use for bootstrapping of large"
        " datasets.",
        type=int,
        default=1000,
        required=False,
    )

    parser.add_argument(
        "--n_subsets",
        help="Number of ensemble subsets to calculate JS divergence for",
        type=int,
        default=None,
        required=False,
    )

    return parser


def get_name():
    return "het_ensemblecomparison"


def js_analysis(clusters: list, output_file: str) -> None:
    # load cluster objects and grab the predicted labels
    predictions = []
    for clustering in clusters:
        predictions.append(pickle.load(clustering).labels_.tolist())

    # extract the cluster alg info from pkl filename

    # convert to np.ndarray
    predictions_matrix = np.asarray(predictions, dtype=int)

    # create a csv to relate clustering alg indices to clustering alg
    # save a pairwise np array of js coeffs. Indices can be interpreted
    # from csv relating index to clustering alg

    # compute js
    compute_js(predictions_matrix)

    # save csv with the pairwise comparison


def compute_js(
    clusters: np.ndarray,
) -> np.ndarray:
    # base will default to scipy.stats.entropy base
    # which is e according to https://docs.scipy.org/doc/scipy/reference/
    # generated/scipy.stats.entropy.html

    # or maybe https://docs.scipy.org/doc/scipy/reference/generated/
    # scipy.spatial.distance.jensenshannon.html
    # which should limit range from 0 to ln(2) (0-0.69314718)
    print("note implemetned {}".format(clusters))
    return np.ndarray((-1, -1))


def compute_hd(
    clusters: np.ndarray,
) -> np.ndarray:
    # base will default to https://docs.scipy.org/doc/scipy/reference/
    # generated/scipy.spatial.distance.directed_hausdorff.html#scipy.spatial.
    # distance.directed_hausdorff
    # which is not symmetric!!!
    print("note implemetned {}".format(clusters))
    return np.ndarray((-1, -1))


def get_index_from_conformation_filename(
    conformation_files: list[str],
    digits: int = 6,
    file_ext: str = ".pdb",
    verbose: bool = False,
) -> list[str]:
    indices = []
    slice_2 = -len(file_ext)
    slice_1 = slice_2 - digits
    for conf_file in conformation_files:
        indices.append(conf_file[slice_1:slice_2])
        if verbose:
            print(
                "{} becomes {}".format(
                    conf_file,
                    conf_file[slice_1:slice_2],
                ),
            )
    return indices


class JSDivergence(object):
    def __init__(
        self,
        pkl_filepath: str,
        dpi: int = 300,
        pdf: bool = True,
    ) -> None:
        """Class for holding the results which result from comparing two or
        more clustering workflows for a given dataset

        Args:
            pkl_filepath (str): _description_
        """
        self.pkl_filepath = pkl_filepath
        self.dpi = dpi
        self.pdf = pdf

        # CES details
        self.ces: list[np.ndarray] | None = None
        self.ces_details: list[np.ndarray] | None = None
        self.ces_avg: np.ndarray | None = None
        self.ces_stddev: np.ndarray | None = None

        # DRES details
        self.dres: list[np.ndarray] | None = None
        self.dres_details: list[np.ndarray] | None = None
        self.dres_avg: np.ndarray | None = None
        self.dres_stddev: np.ndarray | None = None

        # as we have to use subset(s) of ensembles, keep track of settings
        self.args: argparse.ArgumentParser | None = None

        # ensembles which are dict[dict[str, mda.Universe|int]]
        self.ensembles_list: list[mda.Universe] | None = []
        self.ensembles_indices: dict[str, dict[str, int]] | None = None

    def compute_ces_stats(self):
        if self.ces:
            self.ces_avg = np.array(
                np.average(np.array(self.ces, dtype=float), axis=0)[0][:][:]
            )
            self.ces_stddev = np.array(
                np.std(np.array(self.ces, dtype=float), axis=0)[0][:][:]
            )

    def compute_dres_stats(self):
        if self.dres:
            self.dres_avg = np.array(
                np.average(np.array(self.dres, dtype=float), axis=0)[0][:][:]
            )
            self.dres_stddev = np.array(
                np.std(np.array(self.dres, dtype=float), axis=0)[0][:][:]
            )

    def plot_ces_results(self):
        cw = []
        cw_ensemble_index = []
        ensembles_list_index = []

        for k1, v1 in self.ensembles_indices.items():
            for k2, v2 in v1.items():
                cw.append(k1)
                cw_ensemble_index.append(k2)
                ensembles_list_index.append(v2)

        # now we can rearrange the lists by the indices
        # to get labels in the same order as the ces matrix indices
        cw = [cw[i] for i in ensembles_list_index]
        cw_ensemble_index = [
            cw_ensemble_index[i] for i in ensembles_list_index
        ]

        # lets create labels that combine cw and ensemble index
        combined_label = []
        for cluster_workflow, ensemble_index in zip(cw, cw_ensemble_index):
            combined_label.append(
                cluster_workflow + "_ensemble_" + ensemble_index
            )

        # we now have labels to put on x,y axes (they are the same)
        # so can sns.heatmap plot
        # if we have a list, grab the np.ndarray inside (will be 1 long
        # unless there were multiple ces clustering algs used)
        for j, subset in enumerate(self.ces):
            if isinstance(subset, list):
                for i, clustering_approach in enumerate(subset):
                    plot_heatmap(
                        clustering_approach,
                        xlabels=combined_label,
                        ylabels=combined_label,
                        filename=os.path.join(
                            os.path.dirname(self.pkl_filepath),
                            "ces_jsd_{}_subset_{}.png".format(i, j),
                        ),
                        dpi=self.dpi,
                        pdf=self.pdf,
                    )
            else:
                plot_heatmap(
                    subset,
                    xlabels=combined_label,
                    ylabels=combined_label,
                    filename=os.path.join(
                        os.path.dirname(self.pkl_filepath),
                        "ces_jsd_subset_{}.png".format(j),
                    ),
                    dpi=self.dpi,
                    pdf=self.pdf,
                )

        # plot the dres avg and stddev heatmaps
        if self.ces_avg is not None:
            if isinstance(self.ces_avg, list):
                plot_heatmap(
                    self.ces_avg[0],
                    xlabels=combined_label,
                    ylabels=combined_label,
                    filename=os.path.join(
                        os.path.dirname(self.pkl_filepath),
                        "ces_jsd_avg.png",
                    ),
                    dpi=self.dpi,
                    pdf=self.pdf,
                )
            else:
                plot_heatmap(
                    self.ces_avg,
                    xlabels=combined_label,
                    ylabels=combined_label,
                    filename=os.path.join(
                        os.path.dirname(self.pkl_filepath),
                        "ces_jsd_avg.png",
                    ),
                    dpi=self.dpi,
                    pdf=self.pdf,
                )
        if self.ces_stddev is not None:
            if isinstance(self.ces_stddev, list):
                plot_heatmap(
                    self.ces_stddev[0],
                    xlabels=combined_label,
                    ylabels=combined_label,
                    filename=os.path.join(
                        os.path.dirname(self.pkl_filepath),
                        "ces_jsd_stddev.png",
                    ),
                    dpi=self.dpi,
                    pdf=self.pdf,
                )
            else:
                plot_heatmap(
                    self.ces_stddev,
                    xlabels=combined_label,
                    ylabels=combined_label,
                    filename=os.path.join(
                        os.path.dirname(self.pkl_filepath),
                        "ces_jsd_stddev.png",
                    ),
                    dpi=self.dpi,
                    pdf=self.pdf,
                )

        # TODO alter heatmap to only compare one clustering alg to another
        # this may be useful but not necessarily required
        """
        # loop through one side of ces array and ignore diagonal
        for i in range(len(self.ces[0])):
            for j in range(len(self.ces[1]-1)):
                # use the i,j value(s) to grab the correct
                # ces_jsd values
                # and identify the correct cluster workflows which
                # were compared to get it (both pkl file and cw index)
        """

    def plot_dres_results(self):
        cw = []
        cw_ensemble_index = []
        ensembles_list_index = []

        for k1, v1 in self.ensembles_indices.items():
            for k2, v2 in v1.items():
                cw.append(k1)
                cw_ensemble_index.append(k2)
                ensembles_list_index.append(v2)

        # now we can rearrange the lists by the indices
        # to get labels in the same order as the ces matrix indices
        cw = [cw[i] for i in ensembles_list_index]
        cw_ensemble_index = [
            cw_ensemble_index[i] for i in ensembles_list_index
        ]

        # lets create labels that combine cw and ensemble index
        combined_label = []
        for cluster_workflow, ensemble_index in zip(cw, cw_ensemble_index):
            combined_label.append(
                cluster_workflow + "_ensemble_" + ensemble_index
            )

        # we now have labels to put on x,y axes (they are the same)
        # so can sns.heatmap plot
        # if we have a list, grab the np.ndarray inside (will be 1 long
        # unless there were multiple dres clustering algs used)
        for j, subset in enumerate(self.dres):
            if isinstance(subset, list):
                for i, clustering_approach in enumerate(subset):
                    plot_heatmap(
                        clustering_approach,
                        xlabels=combined_label,
                        ylabels=combined_label,
                        filename=os.path.join(
                            os.path.dirname(self.pkl_filepath),
                            "dres_jsd_{}_subset_{}.png".format(i, j),
                        ),
                        dpi=self.dpi,
                        pdf=self.pdf,
                    )
            else:
                plot_heatmap(
                    subset,
                    xlabels=combined_label,
                    ylabels=combined_label,
                    filename=os.path.join(
                        os.path.dirname(self.pkl_filepath),
                        "dres_jsd_subset_{}.png".format(j),
                    ),
                    dpi=self.dpi,
                    pdf=self.pdf,
                )
        if self.dres_avg is not None:
            # plot the dres avg and stddev heatmaps
            if isinstance(self.dres_avg, list):
                plot_heatmap(
                    self.dres_avg[0],
                    xlabels=combined_label,
                    ylabels=combined_label,
                    filename=os.path.join(
                        os.path.dirname(self.pkl_filepath),
                        "dres_jsd_avg.png",
                    ),
                    dpi=self.dpi,
                    pdf=self.pdf,
                )
            else:
                plot_heatmap(
                    self.dres_avg,
                    xlabels=combined_label,
                    ylabels=combined_label,
                    filename=os.path.join(
                        os.path.dirname(self.pkl_filepath),
                        "dres_jsd_avg.png",
                    ),
                    dpi=self.dpi,
                    pdf=self.pdf,
                )
        if self.dres_details is not None:
            if isinstance(self.dres_stddev):
                plot_heatmap(
                    self.dres_stddev[0],
                    xlabels=combined_label,
                    ylabels=combined_label,
                    filename=os.path.join(
                        os.path.dirname(self.pkl_filepath),
                        "dres_jsd_stddev.png",
                    ),
                    dpi=self.dpi,
                    pdf=self.pdf,
                )
            else:
                plot_heatmap(
                    self.dres_stddev,
                    xlabels=combined_label,
                    ylabels=combined_label,
                    filename=os.path.join(
                        os.path.dirname(self.pkl_filepath),
                        "dres_jsd_stddev.png",
                    ),
                    dpi=self.dpi,
                    pdf=self.pdf,
                )

        # TODO alter heatmap to only compare one clustering alg to another
        # this may be useful but not necessarily required
        """
        # loop through one side of ces array and ignore diagonal
        for i in range(len(self.ces[0])):
            for j in range(len(self.ces[1]-1)):
                # use the i,j value(s) to grab the correct
                # ces_jsd values
                # and identify the correct cluster workflows which
                # were compared to get it (both pkl file and cw index)
        """

    def save_jsd(
        self,
        overwrite: bool = False,
    ):
        if os.path.isfile(self.pkl_filepath):
            assert overwrite, "{} exists. Instruct to overwrite if required!"
        with open(self.pkl_filepath, "wb") as f:
            pickle.dump(self, f)


def plot_heatmap(
    data: np.ndarray,
    xlabels: list[str],
    ylabels: list[str],
    filename: str,
    dpi: int = 300,
    pdf: bool = True,
):
    """Plot a heatmap for a square numpy array. Must has 2 dimensions

    Args:
        data (np.ndarray): _description_
        filename (str): _description_
    """
    assert len(data.shape) == 2

    # plot the heatmap
    ax = sns.heatmap(
        data,
        xticklabels=xlabels,
        yticklabels=ylabels,
        annot=True,
        linewidths=0.5,
        cbar=True,
        # vmin=0.,
        # vmax=0.7,
    )
    ax.xaxis.tick_bottom()
    ax.yaxis.tick_left()

    # save to disk
    if filename[-4:] == ".png":
        figname = filename
    else:
        figname = filename + ".png"
    plt.savefig(figname, dpi=dpi)
    if pdf:
        plt.savefig(
            figname.replace(".png", ".pdf"),
        )
    plt.clf()


def main(args):
    """Load in pkl files and compare similarity via jensen-shannon

    Args:
        args (_type_): _description_
    """

    # check that 2 cluster pkls were provided
    assert len(args.clusters) == 2
    # check the clusters sources are correctly provided for the pkl files
    assert len(args.clusters_source) == len(args.clusters_source)
    for source in args.clusters_source:
        assert source == "MD" or source == "latent"

    # get the conformation filenames
    conformation_filenames = get_pdb_list(
        args.conformations_dir,
        args.file_ext,
    )
    conformation_filenames = select_confs(
        conformation_filenames,
        args.n_confs,
        args.first_conf,
        args.contiguous_confs,
    )

    # TODO replace/update setting of these values
    particle_diameter = 100  # approximate particle diameter in Angstroms
    ugraph_shape = (
        4000,
        4000,
    )
    # above is shape of the micrograph in pixels. Only needs to be given
    # if the metadata file is a .star file
    ignore_missing_files = True
    enable_tqdm = True

    # use these values to grab and get closest pdb_index
    if args.verbose:
        print("Computing closest truth particle indices")
    analysis = load_data(
        args.meta_file,
        args.config_dir,
        particle_diameter,
        ugraph_shape=ugraph_shape,
        verbose=args.verbose,
        enable_tqdm=enable_tqdm,
        ignore_missing_files=ignore_missing_files,
    )  # creates the class
    df_picked = pd.DataFrame(analysis.results_picking)
    df_truth = pd.DataFrame(analysis.results_truth)
    _, df_picked = analysis.compute_precision(
        df_picked,
        df_truth,
        verbose=args.verbose,
    )

    # extract the cluster indices from the pkl files
    # easiest to use a dict for keeping track
    cluster_indices: dict[str, pd.DataFrame] = {}
    # open the pkls
    for cluster_file, pkl_source in zip(args.clusters, args.clusters_source):
        workflow = pickle.load(open(cluster_file, "rb"))
        # MD-derived pkl files have clusterlabels ordered from first
        # conformation to last conformation. May be subject to change
        if pkl_source == "MD":
            # this automatically sorts the conformations to be
            # in the same order as the cluster indices array output
            # by the het_metrics script by using the same loading func
            conformation_indices = get_index_from_conformation_filename(
                conformation_filenames,
                digits=args.digits,
                file_ext=args.file_ext,
                verbose=args.verbose,
            )
            if args.verbose:
                print(
                    "Conformation indices:\n{}\nThere are {} entries".format(
                        conformation_indices,
                        len(conformation_indices),
                    )
                )
                print(
                    "Cluster labels:\n{}\nThere are {} entries".format(
                        workflow.ca_obj.labels_,
                        len(workflow.ca_obj.labels_),
                    )
                )
        # reconstruction metadata-derived pkl files have cluster labels
        # ordered from first entry in metadata file to last, so need to
        # reorder the labels to be from conformation 0 to conformation X
        elif pkl_source == "latent":
            # have the `closest_pdb` column filled which allows
            # each label entry to have a pdb_index associated with it.
            # This works because the cluster indices output by the
            # latent_clustering script should be in the same
            # order as the particles in the metadata file!

            # so get the digits-length conformation indices as str as above
            conformation_indices = get_index_from_conformation_filename(
                df_picked["closest_pdb"].tolist(),
                digits=args.digits,
                file_ext=args.file_ext,
                verbose=args.verbose,
            )
        else:
            raise ValueError(
                "Must use either `MD` or `latent` as the pkl_source values"
            )

        # we already optionally limited n_confs from the conformations_dir
        # now we need to make sure that it is only these indices which
        # we carry forward into calculation of JD-divergence values
        # To do that, grab the indices of conformations which are present
        # in the list conformation_filenames
        keep = []
        for this_index in conformation_indices:
            present = False
            for conf_filename in conformation_filenames:
                if this_index in conf_filename:
                    present = True
            keep.append(present)
        keep = np.array(keep, dtype=bool)

        if args.verbose:
            print(
                "shape conformation_indices:{}\n{}\n\n".format(
                    np.array(conformation_indices, dtype=str).shape,
                    np.array(conformation_indices, dtype=str),
                )
            )
            print(
                "shape conformation_filenames:{}\n{}\n\n".format(
                    np.array(conformation_filenames, dtype=str).shape,
                    np.array(conformation_filenames, dtype=str),
                )
            )
            print(
                "labels.shape:{}\nlabels:{}\n\n".format(
                    workflow.ca_obj.labels_.shape, workflow.ca_obj.labels_
                )
            )
            print("shape keep:{}\n{}\n\n".format(keep.shape, keep))
            print(
                "kept clusters have indices with this subset:{} of {}".format(
                    np.unique(workflow.ca_obj.labels_[keep]),
                    np.unique(workflow.ca_obj.labels_),
                )
            )
        conformation_indices_subset = np.array(
            conformation_indices, dtype=str
        )[keep].tolist()
        cluster_index_subset = workflow.ca_obj.labels_[keep]

        # create the pd dataframe
        # if args.n_confs is not used, these are full sets, not subsets
        cluster_indices[cluster_file] = pd.DataFrame(
            {
                "conformation_index": conformation_indices_subset,
                "cluster_index": cluster_index_subset,
            }
        )

    # now we have the conformation index and cluster index labels,
    # load up all the conformations from conformations_dir into an
    # MDAnalysis Universe and then create a sub-universe for each cluster
    # md_trajectory = mda_load_universe(args)

    # extract the ensembles for each clustering workflow
    """
    ensembles: dict[str, dict[str, mda.Universe]] = {}
    for cw, ci_df in cluster_indices.items():
        cluster_conformations: dict[str, mda.Universe] = {}

        # for each cluster index, get an mda.Universe
        for cluster_index in np.unique(ci_df["cluster_index"]):
            # grab the conformation indices corresponding to this cluster
            select_conformations = ci_df.loc[
                ci_df["cluster_index"] == cluster_index
            ]

            # need each ensemble to be a mda.Universe, so
            # for clustering, extract the list of filenames of conformations
            # (pdb files) which are each in ensemble/cluster. IT is assumed
            # that they may be repeated any number of times from 1 to N
            ensemble_conformations: list[str] = []
            for conformation_index in select_conformations[
                "conformation_index"
            ].tolist():
                conformation_file: list[str] = []
                # grab the filename that this str(X).zfill(6) string is
                # present in
                for conformation_filename in conformation_filenames:
                    if conformation_index in conformation_filename:
                        conformation_file.append(conformation_filename)

                # check that the indexing is correct
                # as there should only be one entry which matches
                if len(conformation_file) != 1:
                    print(
                        "Matching string index to single conformation"
                        " failed! \n{}\nare found for {}".format(
                            conformation_file, conformation_index
                        )
                    )
                assert len(conformation_file) == 1

                # now that we have a len 1 list with the pdb file path,
                # append the ensemble conformations
                ensemble_conformations += conformation_file

            # now we have the list of conformations for this ensemble/cluster
            # we create an ensemble and add it to the cluster_conformations
            # dictionary
            # Do not sort the ensemble during this
            cluster_conformations[str(cluster_index)] = mda.Universe(
                conformation_filenames[0],
                ensemble_conformations,
            )
            # TODO check if these conformations need aligning before ces/dres
            # AND HOW TO ALIGN MULTIPLE ENSEMBLES???
            # ANSW: THEY SHOULD BE ALGINED BY CES RMSD CALC!!!!
            # mda.analysis.align.AlignTraj(
            #     cluster_conformations[cw][str(cluster_index)],
            #     cluster_conformations[cw][str(cluster_index)],
            #     select="name CA",
            #     in_memory=True,
            # ).run()
        # add these ensembles to the ensembles dict
        ensembles[cw] = cluster_conformations
        # create a list of mda.Universes and a list of indices to keep track

    """

    # we now how a dict holding a dict for each clustering workflow
    # The subdict holds one mda.Universe for each cluster index
    # Therefore, we create a list of ALL ensembles and a dict which
    # is the same as ensembles var except it holds index of list
    # for each of mda.Universe
    """
    ensembles_indices: dict[str, dict[str, int]] = {}
    ensembles_list = []
    counter = 0
    for k, v in ensembles.items():
        cluster_list_indices: dict[str, int] = {}
        for cluster_index, universe in v.items():
            cluster_list_indices[str(cluster_index)] = counter
            # ensembles_list.append(universe)
            counter += 1
        ensembles_indices[k] = cluster_list_indices
    """

    ensembles_indices: dict[str, dict[str, int]] = {}
    # ensembles_list = []
    counter = 0
    for cw, ci_df in cluster_indices.items():
        cluster_list_indices: dict[str, int] = {}
        for cluster_index in np.unique(ci_df["cluster_index"]):
            cluster_list_indices[str(cluster_index)] = counter
            counter += 1
        ensembles_indices[cw] = cluster_list_indices

    # now need to add the following objects to the JS-divergence.pkl class
    # ces
    # details
    # ensembles_indices
    js_divergence = JSDivergence(
        os.path.join(args.output_dir, "js_divergence.pkl"),
        dpi=args.dpi,
        pdf=args.pdf,
    )
    js_divergence_ces = []
    js_divergence_ces_details = []
    js_divergence_dres = []
    js_divergence_dres_details = []

    # because of compute limits on RMSD calculations for all confs in all
    # conformations there are 2 solutions.
    # First is to calc RMSD between all conformations in the dataset and then
    # to use arrangement of ensembles to fill in values for repeated
    # comparisons. However, for a large number of unique conformations and
    # a large enough molecule, this calculation is still prohibitively slow
    # Second option (implemented) is to bootstrap the JS-divergence
    # calculation between ensembles. Total of around 1000 conformations
    # takes around 1 hr to compute JS-divergence (ie: RMSD and CES) for covid
    # spike trimer. Therefore using ensemble_subset_size and n_subsets to
    # determine number of conformations to extract from each ensemble and
    # how many times to compare subsets

    # loop through all the subsets to make, randomly select the subset of
    # confs for each ensemble, calculate the CES JS-D and append to the pkl
    # Number of ensembles is len(ensembles_list)
    for subset_i in range(args.n_subsets):
        subset_ensembles: dict[str, dict[str, mda.Universe]] = {}
        for cw, ci_df in cluster_indices.items():
            cluster_conformations: dict[str, mda.Universe] = {}

            # for each cluster index, get an mda.Universe
            for cluster_index in np.unique(ci_df["cluster_index"]):
                # grab the conformation indices corresponding to this cluster
                select_conformations = ci_df.loc[
                    ci_df["cluster_index"] == cluster_index
                ]

                # this time randomly select a subset of the confs
                subset_confs = random.sample(
                    select_conformations["conformation_index"].tolist(),
                    args.ensemble_subset_size,
                )

                # need each ensemble to be a mda.Universe, so
                # for clustering, extract the list of filenames of
                # conformations
                # (pdb files) which are each in ensemble/cluster. IT is assume
                # that they may be repeated any number of times from 1 to N
                ensemble_conformations: list[str] = []
                for conformation_index in subset_confs:
                    conformation_file: list[str] = []
                    # grab the filename that this str(X).zfill(6) string is
                    # present in
                    for conformation_filename in conformation_filenames:
                        if conformation_index in conformation_filename:
                            conformation_file.append(conformation_filename)

                    # check that the indexing is correct
                    # as there should only be one entry which matches
                    if len(conformation_file) != 1:
                        print(
                            "Matching string index to single conformation"
                            " failed! \n{}\nare found for {}".format(
                                conformation_file, conformation_index
                            )
                        )
                    assert len(conformation_file) == 1

                    # now that we have a len 1 list with the pdb file path,
                    # append the ensemble conformations
                    ensemble_conformations += conformation_file

                # now we have the list of conformations for this
                # ensemble/cluster
                # we create an ensemble and add it to the
                # cluster_conformations dictionary
                # Do not sort the ensemble during this
                cluster_conformations[str(cluster_index)] = mda.Universe(
                    conformation_filenames[0],
                    ensemble_conformations,
                )

            subset_ensembles[cw] = cluster_conformations

        # grab the subset
        ensembles_list_subset = []
        for k, v in subset_ensembles.items():
            for cluster_index, universe in v.items():
                ensembles_list_subset.append(universe)

        # grab the JS-D
        ces, ces_details = mda.analysis.encore.similarity.ces(
            ensembles_list_subset,
            select=args.select,
            clustering_method=AffinityPropagationNative(
                preference=args.preference,
            ),
            # distance_matrix=rmsd_matrix,
            estimate_error=args.estimate_error,
            bootstrapping_samples=args.bootstrapping_samples,
            ncores=args.ncores,
            calc_diagonal=True,
            allow_collapsed_result=False,
        )

        js_divergence_ces.append(ces)
        js_divergence_ces_details.append(ces_details)

        dres, dres_details = mda.analysis.encore.similarity.dres(
            ensembles_list_subset,
            select=args.select,
            dimensionality_reduction_method=StochasticProximityEmbeddingNative(
                dimension=args.dimensions,
                distance_cutoff=1.5,
                min_lam=0.1,
                max_lam=2.0,
                ncycle=100,
                nstep=10000,
            ),
            # distance_matrix=rmsd_matrix,
            nsamples=len(ensembles_list_subset),
            estimate_error=args.estimate_error,
            bootstrapping_samples=args.bootstrapping_samples,
            ncores=args.ncores,
            calc_diagonal=True,
            allow_collapsed_result=False,
        )

        js_divergence_dres.append(dres)
        js_divergence_dres_details.append(dres_details)

        print("Completed subset {}".format(subset_i))

    # create output dir
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        if args.verbose:
            print("Created {}".format(args.output_dir))
    # make the csv file with input/output tracking from pd.DataFrame
    # track the pkl files and the type of clustering workflow
    metadata = {}
    for i, k in enumerate(ensembles_indices):
        # gives index i and key k
        metadata["input_file_{}".format(i)] = [k]
        metadata["input_file_{}_type".format(i)] = [args.clusters_source[i]]

    # get the rmsd input or output file path
    if args.rmsd_precalc:
        metadata["rmsd_calculation"] = args.rmsd_precalc
    elif args.save_rmsd:
        metadata["rmsd_calculation"] = os.path.join(
            args.output_dir, "rmsd.npz"
        )

    # get the filepath of the class pkl which will be created to hold
    # the heatmap which results from calculating the JS-divergence
    # as well as the indexing details of the output np.array
    # and the ces and details objects
    metadata["JS-divergence"] = [
        os.path.join(args.output_dir, "JS-divergence.pkl")
    ]
    # metadata dict is now complete! so save it to disk
    pd.DataFrame.from_dict(metadata).to_csv(
        os.path.join(args.output_dir, "JS-divergence-metadata.csv")
    )

    # if args.pkl_universes:
    #     js_divergence.ensembles_list = ensembles_list
    js_divergence.ensembles_indices = ensembles_indices
    js_divergence.args = args
    js_divergence.ces = js_divergence_ces
    js_divergence.ces_details = js_divergence_ces_details
    js_divergence.compute_ces_stats()
    js_divergence.dres = js_divergence_dres
    js_divergence.dres_details = js_divergence_dres_details
    js_divergence.compute_dres_stats()
    # save the object to file
    js_divergence.save_jsd(args.overwrite)
    # use list of indices to create cluster workflow vs clustering
    # workflow heatmap of JS-divergences
    js_divergence.plot_ces_results()
    js_divergence.plot_dres_results()

    """
    # Now we load or calc the RMSD matrix before using it as an input to
    # CES()
    # load up the rmsd if it was precalc'ed
    if args.rmsd_precalc:
        if args.verbose:
            print(
                "Loading rmsd matrix from: {}".format(
                    metadata["rmsd_calculation"]
                )
            )
        rmsd_matrix = mda.analysis.encore.get_distance_matrix(
            mda.analysis.encore.utils.merge_universes(ensembles_list),
            load_matrix=metadata["rmsd_calculation"],
        )
    # otherwise calculate it and (optionally) save to disk
    else:
        if args.save_rmsd:
            if args.verbose:
                print(
                    "Calculating distance matrix and saving to {}".format(
                        metadata["rmsd_calculation"]
                    )
                )
            # calc the rmsd and save to disk
            rmsd_matrix = mda.analysis.encore.get_distance_matrix(
                mda.analysis.encore.utils.merge_universes(ensembles_list),
                save_matrix=metadata["rmsd_calculation"],
            )
        else:
            if args.verbose:
                print("Calculating distance matrix but not saving it")
            # calc the rmsd
            rmsd_matrix = mda.analysis.encore.get_distance_matrix(
                mda.analysis.encore.utils.merge_universes(ensembles_list),
                save_matrix=metadata["rmsd_calculation"],
            )

    ces, details = mda.analysis.encore.similarity.ces(
        ensembles_list,
        select=args.select,
        clustering_method=AffinityPropagationNative(
            preference=args.preference,
        ),
        distance_matrix=rmsd_matrix,
        estimate_error=args.estimate_error,
        bootstrapping_samples=args.bootstrapping_samples,
        ncores=args.ncores,
        allow_collapsed_result=False,
    )

    # TODO
    # now need to add the following objects to the JS-divergence.pkl class
    # ces
    # details
    # ensembles_list
    # ensembles_indices
    # then add utilities to the class which allows you to plot heatmaps
    # for all vs all
    # and for each clustering workflow vs each other clustering workflow
    js_divergence = JSDivergence(
        os.path.join(args.output_dir, "js_divergence.pkl"),
        dpi=args.dpi,
        pdf=args.pdf,
    )
    js_divergence.ces = ces
    js_divergence.ces_details = details
    if args.pkl_universes:
        js_divergence.ensembles_list = ensembles_list
    js_divergence.ensembles_indices = ensembles_indices

    # save the object to file
    js_divergence.save_jsd(args.overwrite)
    # use list of indices to create cluster workflow vs clustering
    # workflow heatmap of JS-divergences
    js_divergence.plot_ces_results()
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
