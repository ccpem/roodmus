"""Load in conformations (pdb) and compute distance
metrics and perform clustering

User specifies a directory which holds conformations and dimension reduction,
distance metric and cluster algorithm parameter(s) to use.

1. Load in conformations (.pdb supported) sorted in alphanumeric order,
based on the file name

2. Compute all possible configurations of supplied clustering workflow using
parameters specified as user input (otherwise default are used)

3. Save workflow parameters (and pkl file locations holding the resulting
python objects at each stage) in a csv file

4. Do dimension reduction on trajectory xyz info (excludes hydrogen/helium)

5. Compute distance metric on dimension reduced (or original) tarj xyz

6. Apply clustering on distance matrix

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
import pickle
import logging
import glob2 as glob

import MDAnalysis as mda
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import mdtraj
import scipy.cluster.hierarchy

from itertools import combinations
from MDAnalysis.analysis import align, diffusionmap


from scipy.spatial.distance import squareform, pdist
from sklearn.decomposition import PCA, IncrementalPCA, FastICA, KernelPCA
from sklearn.manifold import (
    Isomap,
    LocallyLinearEmbedding,
    SpectralEmbedding,
    MDS,
    TSNE,
)
import umap
from sklearn.cluster import (
    KMeans,
    MiniBatchKMeans,
    AffinityPropagation,
    MeanShift,
    SpectralClustering,
    AgglomerativeClustering,
    DBSCAN,
    HDBSCAN,
    OPTICS,
)
from sklearn.mixture import GaussianMixture

"""
from MDAnalysis.analysis import encore
from MDAnalysis.analysis.encore.clustering import ClusteringMethod as clm
from MDAnalysis.analysis.encore.dimensionality_reduction import (
    DimensionalityReductionMethod as drm,
)
"""
# from scipy.spatial.distance import pdist
# import matplotlib.cm as cm

# setup file to write warning and logs to
logging.basicConfig(filename="het_metrics.log", level=logging.DEBUG)
logging.captureWarnings(True)


def add_arguments(parser: argparse.ArgumentParser):
    """Parse arguments for performing one or more clustering workflows
    (as configured by provided argument permutations)

    Args:
        parser (argparse.ArgumentParser): _description_

    Returns:
        _type_: _description_
    """

    parser.add_argument(
        "--conformations_dir",
        "-c",
        help="Directory with .pdb files",
        type=str,
    )
    """
    parser.add_argument(
        "--topfile_path",
        help="The pdb holding the structure of molecule (no solvent)",
        type=str,
        default="",
    )
    """

    parser.add_argument(
        "--n_confs",
        help="Limit to <n_confs> conformations",
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
        "--verbose", help="increase output verbosity", action="store_true"
    )

    parser.add_argument(
        "--overwrite",
        help="Overwrite previous pkl files with same filepath",
        action="store_true",
    )

    parser.add_argument(
        "--dpi",
        help="choose dots per inch in png plots, Default tis 100",
        type=int,
        default=100,
        required=False,
    )

    parser.add_argument(
        "--output_dir",
        help="Directory to save results and intermediate results",
        type=str,
        default="het_metrics",
        required=False,
    )

    parser.add_argument(
        "--workflows_filename",
        help="Filename for csv which keeps track of workflows and locations"
        " of the pkl file containing the clusters from each workflow."
        " Defaults to workflows.csv",
        type=str,
        default="workflows.csv",
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
        "--n_cores",
        help="Number of compute cores to use." " Default is 1",
        type=int,
        default=1,
    )

    # distance metrics
    parser.add_argument(
        "--distance_metric",
        help="Distance metric to use in workflow." " Defaults to none.",
        nargs="+",
        choices=["", "approx_rmsd", "rmsd"],
        type=str,
        default=[""],
    )

    # dimension reduction
    parser.add_argument(
        "--dimension_reduction",
        help="Dimensionality reduction technique to apply to distance"
        " metric. Defaults to none",
        nargs="+",
        choices=[
            "",
            "pca",
            "ipca",
            "ica",
            "kernelpca",
            "isomap",
            "lle",
            "mlle",
            "hlle",
            "ltsa",
            "spectral",
            "mds_metric",
            "mds_nonmetric",
            "tsne",
            "umap",
        ],
        type=str,
        default=[""],
    )

    # dimensions to reduce to
    parser.add_argument(
        "--dimensions",
        help="Number of dimensions to reduce the pairwise distance metric."
        " If number of args provided is not same as number of uses of"
        " dimension_reduction argument then: if no --dimensions arg (or a"
        " single) argument is supplied made, the default (2) will be used"
        " for all. Else an error will be raised.",
        nargs="+",
        type=int,
        default=[2],
    )

    # clustering alg
    parser.add_argument(
        "--cluster_alg",
        help="Algorithm to use for clustering of (optionally dimension"
        " reduced) distance metric.",
        nargs="+",
        choices=[
            "kmeans",
            "mbkmeans",
            "affinity_euclid",
            "affinity_precomputed",
            "meanshift",
            "spectral",
            # "spectral_precomputed",
            "ward",
            "hier_avg",
            "hier_complete",
            "hier_single",
            "dbscan",
            "hdbscan",
            "optics",
            # "gmm",
        ],
        type=str,
        default=["kmeans"],
    )

    # nclusters
    parser.add_argument(
        "--n_clusters",
        help="Number of clusters to use in clustering",
        nargs="+",
        type=int,
        default=[2],
    )

    # alignment alg
    parser.add_argument(
        "--alignment",
        help="Alignment algorithm to use. Defaults to mdtraj.superpose()",
        nargs="+",
        choices=["", "superpose"],
        type=str,
        default=["superpose"],
    )

    # store output pkl files
    parser.add_argument(
        "--nopkl",
        help="Disable saving of pkl files containing workflow outputs",
        action="store_true",
    )

    parser.add_argument("--pdf", help="save plot as pdf", action="store_true")

    return parser


def get_name():
    return "het_metrics"


def get_pdb_list(
    conformations_dir: str,
    file_ext: str = ".pdb",
) -> list[str]:
    confs_path = os.path.join(conformations_dir, "*{}".format(file_ext))
    return glob.glob(confs_path)


def select_confs(
    conf_files: list[str],
    n_confs: int,
    first_conf: int = 0,
    contiguous_confs: bool = False,
) -> list[str]:
    conf_files = sorted(conf_files)
    # limit number of conformations if desired
    if n_confs and contiguous_confs:
        if n_confs > len(conf_files) - first_conf:
            raise ValueError(
                "Trying to sample {} confs from {} files starting at"
                " index {}".format(
                    n_confs,
                    len(conf_files),
                    first_conf,
                )
            )
        conf_files = conf_files[first_conf : first_conf + n_confs]
    elif n_confs:
        # get every nth sample depending on n_confs requested
        if (len(conf_files) - first_conf) < n_confs:
            raise ValueError(
                "Trying to sample {} confs from {} remaining! Error!".format(
                    n_confs,
                    len(conf_files) - first_conf,
                )
            )
        else:
            sample_indices = np.arange(
                first_conf,
                len(conf_files),
                (len(conf_files) - first_conf) / n_confs,
            ).astype(int)
            conf_files = np.array(conf_files, dtype=str)[
                sample_indices
            ].tolist()
    assert len(conf_files) > 0
    return conf_files


def mda_load_universe(args) -> mda.Universe:
    conf_files = get_pdb_list(
        args.conformations_dir,
        args.file_ext,
    )
    # TODO support option to load pdb files from .star or .npz (cs)?
    # sort alphanumerically

    conf_files = select_confs(
        conf_files,
        args.n_confs,
        args.first_conf,
        args.contiguous_confs,
    )
    # add the first pdb as a topology file
    topfile = conf_files[0]
    # turn list of pdbs into an mda.Universe
    # should be supported accoring to table in
    # https://userguide.mdanalysis.org/1.1.1/formats/index.html#formats
    # default timestep is 1ps and default time_offset is 0
    conformations = mda.Universe(
        topfile,
        conf_files,
        dt=args.timestep,
        time_offset=args.time_offset,
    )
    return conformations


def mdtraj_load_traj(args) -> mdtraj.Trajectory:
    conf_files = get_pdb_list(
        args.conformations_dir,
        args.file_ext,
    )
    # TODO support option to load pdb files from .star or .npz (cs)?
    conf_files = select_confs(
        conf_files,
        args.n_confs,
        args.first_conf,
        args.contiguous_confs,
    )
    # add the first pdb as a topology file
    topfile = conf_files[0]
    traj = mdtraj.load(conf_files, top=topfile)
    if args.verbose:
        print(traj)
    return traj


def mdtraj_pairwise_rmsd(
    traj: mdtraj.Trajectory,
) -> np.ndarray:
    distances = np.empty((traj.n_frames, traj.n_frames))
    for i in range(traj.n_frames):
        distances[i] = mdtraj.rmsd(traj, traj, i)
    return distances


def plot_mdtraj_pairwise_rmsd(
    args,
    distances: np.ndarray,
) -> None:
    mdtraj_rmsd_figname = os.path.join(args.output_dir, "mdt_rmsd")
    plt.imshow(
        distances,
        cmap="viridis",
    )
    plt.xlabel("Conformation #")
    plt.ylabel("Conformation #")
    plt.colorbar(label=r"RMSD ($\AA$)")
    plt.savefig(mdtraj_rmsd_figname + ".png", dpi=args.dpi)
    if args.pdf:
        plt.savefig(
            mdtraj_rmsd_figname + ".pdf",
        )
    plt.clf()


def avg_linkage_cluster(
    pairwise_metric_matrix: np.ndarray,
    verbose: bool = False,
) -> np.ndarray:
    if verbose:
        sub_transpose = pairwise_metric_matrix - pairwise_metric_matrix.T
        print(sub_transpose[sub_transpose > 1e-6])
    # check that matrix is (pretty much) symmetric
    assert np.all(pairwise_metric_matrix - pairwise_metric_matrix.T < 1e-5)
    reduced_distances = squareform(pairwise_metric_matrix, checks=False)
    linkage = scipy.cluster.hierarchy.linkage(
        reduced_distances, method="average"
    )
    return linkage


def plot_rmsd_dendrogram(
    args,
    linkage: np.ndarray,
) -> None:
    # get dendrogram
    plt.title("RMSD Average linkage hierarchical clustering")
    # rmsd_avglink_dendro
    _ = scipy.cluster.hierarchy.dendrogram(
        linkage, no_labels=True, count_sort="descendent"
    )
    dendrogram_fpath = os.path.join(
        args.output_dir, "scipy_rmsd_avglink_dendro"
    )
    plt.savefig(dendrogram_fpath + ".png", dpi=args.dpi)
    if args.pdf:
        plt.savefig(
            dendrogram_fpath + ".pdf",
        )
    plt.clf()


def mdtraj_apply_pca(
    n_components: int = 2,
) -> PCA:
    # get pca ou of mdtraj inputs
    pca1 = PCA(n_components=n_components)
    return pca1


def mdtraj_plot_2d_pca(
    args,
    pca_fit: np.ndarray,
    traj: mdtraj.Trajectory,
) -> None:
    plt.scatter(pca_fit[:, 0], pca_fit[:, 1], marker="x", c=traj.time)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Cartesian coordinate PCA")
    plt.colorbar()
    mdtraj_2dpca_figname = os.path.join(args.output_dir, "mdtraj_2dpca")
    plt.savefig(mdtraj_2dpca_figname + ".png", dpi=args.dpi)
    if args.pdf:
        plt.savefig(
            mdtraj_2dpca_figname + ".pdf",
        )
    plt.clf()


def mdtraj_pairwise_distances(
    args,
    traj: mdtraj.Trajectory,
    n_atoms: int | None = None,
) -> np.ndarray:
    atom_pairs = list(combinations(range(traj.n_atoms), 2))
    if n_atoms:
        atom_pairs = list(combinations(range(n_atoms), 2))
    pairwise_distances = mdtraj.geometry.compute_distances(traj, atom_pairs)
    if args.verbose:
        print(pairwise_distances.shape)
    return pairwise_distances


def mda_distancematrix(
    conformations: mda.Universe,
    select: str = "name CA",
    in_memory: bool = True,
) -> mda.analysis.diffusionmap.DistanceMatrix:
    align.AlignTraj(
        conformations,
        conformations,
        select="name CA",
        in_memory=True,
    ).run()
    # then calc a distance matrix
    matrix = diffusionmap.DistanceMatrix(
        conformations,
        select="name CA",
    ).run()
    return matrix


def plot_mda_distancematrix(
    args,
    matrix_2d: np.ndarray,
) -> None:
    # create output dir and rmsd outputs into it
    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    dist_matrix_fpath = os.path.join(args.output_dir, "dist_matrix")
    plt.imshow(
        matrix_2d,
        cmap="viridis",
    )
    plt.xlabel("Conformation #")
    plt.ylabel("Conformation #")
    plt.colorbar(label=r"RMSD ($\AA$)")
    plt.savefig(dist_matrix_fpath + ".png", dpi=args.dpi)
    if args.pdf:
        plt.savefig(
            dist_matrix_fpath + ".pdf",
        )
    plt.clf()


class ensembleComparison(object):
    """Class to perform comparison of clusters. Clusters to compare are
    specified via CLI args from the csv (or otherwise) which keeps
    track of the clustering workflows.
    Adds a file location of computed metrics into the tracking csv file.
    This contains pairwise computation of (symmetric) hausdorf metric
    and Shannon-Jensen comparison of the clusters (from the 2 workflows).
    Run on the same workflow the get pairwise comparison between itself
    instead.

    Args:
        object (_type_): _description_

    Returns:
        _type_: _description_
    """

    def __init__(self) -> None:
        pass


def plot_2d_embedding(
    embedding_array: np.ndarray,
    color_scatter: np.ndarray,
    filename: str,
    metric: str,
    xlabel: str = "z0",
    ylabel: str = "z1",
    dpi: int = 300,
    pdf: bool = False,
) -> None:
    assert embedding_array.shape[1] == 2
    plt.scatter(
        embedding_array[:, 0],
        embedding_array[:, 1],
        marker="o",
        c=color_scatter,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.colorbar(label=r"{}".format(metric))
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


def plot_distancematrix(
    distance_matrix: np.ndarray,
    filename: str,
    metric: str,
    dpi: int = 300,
    pdf: bool = False,
) -> None:
    square_matrix = squareform(distance_matrix)
    if filename[-4:] == ".png":
        dist_matrix_fpath = filename
    else:
        dist_matrix_fpath = filename + ".png"
    plt.imshow(
        square_matrix,
        cmap="viridis",
    )
    plt.xlabel("Conformation #")
    plt.ylabel("Conformation #")
    plt.colorbar(label=r"{} ($\AA$)".format(metric))
    plt.savefig(dist_matrix_fpath, dpi=dpi)
    if pdf:
        plt.savefig(
            dist_matrix_fpath.replace(".png", ".pdf"),
        )
    plt.clf()


def determine_workflow_permutations(
    alignment: list[str],
    dimension_reduction: list[str],
    dimensions: list[int | float | None],
    distance_metric: list[str],
    cluster_alg: list[str],
    clusters: list[int],
    verbose: bool = False,
) -> pd.DataFrame:
    """Turn permutations of workflow into a dataframe

    Args:
        alignment (list[str], optional): _description_.
        Defaults to ["superpose"]
        dimension_reduction (list[str]): _description_
        dimensions (list[int|float]): _description_
        distance_metric (list[str]): list of distance metrics
        cluster_alg (list[str]): _description_
        clusters (list[int]): _description_.

    Returns:
        pd.DataFrame: _description_
    """
    workflow_labels = [
        "alignment",
        "dimension_reduction",
        "dimensions",
        "distance_metric",
        "cluster_alg",
        "clusters",
    ]
    if verbose:
        print("alignment: {}".format(alignment))
        print("dimension_reduction: {}".format(dimension_reduction))
        print("dimensions: {}".format(dimensions))
        print("distance_metric: {}".format(distance_metric))
        print("cluster_alg: {}".format(cluster_alg))
        print("clusters: {}".format(clusters))

    workflows = []
    for al in alignment:
        for dr in dimension_reduction:
            for dims in dimensions:
                for dm in distance_metric:
                    for ca in cluster_alg:
                        for cn in clusters:
                            workflows.append([al, dr, dims, dm, ca, cn])

    workflows_df = pd.DataFrame(workflows, columns=workflow_labels)
    if verbose:
        print("workflow permutations:\n{}".format(workflows_df))
    return workflows_df


def none_to_empty_str(check_var):
    if check_var is None:
        return ""
    else:
        return str(check_var)


def construct_object_pkl_locations(
    directory: str,
    alignment: str,
    dimension_reduction: str,
    dimensions: int | float | None,
    distance_metric: str,
    cluster_alg: str,
    clusters: str,
):
    # preserve workflow in filename
    pkl_filename = "{}_{}_{}_{}_{}_{}.pkl".format(
        alignment,
        dimension_reduction,
        none_to_empty_str(dimensions),
        distance_metric,
        cluster_alg,
        clusters,
    )
    pkl_path = os.path.join(directory, pkl_filename)
    return pkl_path


def determine_workflow_pkl_locations(
    results_dir: str,
    workflows: pd.DataFrame,
):
    pkl_files = []
    # note that pkl files are put in same dir as workflows.csv
    for workflow_index in range(len(workflows)):
        workflow = workflows.iloc[workflow_index]
        # dimension reduction pkl (if any)
        pkl_files.append(
            construct_object_pkl_locations(
                results_dir,
                workflow["alignment"],
                workflow["dimension_reduction"],
                none_to_empty_str(workflow["dimensions"]),
                workflow["distance_metric"],
                workflow["cluster_alg"],
                workflow["clusters"],
            )
        )

    # add new column to df for each output object
    print(pkl_files)
    workflows["pkl"] = pkl_files
    return workflows


class Workflow(object):
    def __init__(
        self,
        pkl_filepath: str,
    ) -> None:
        """Class for holding the outputs from (optional) dimension reduction,
        distance matrix calculation and clustering.

        Args:
            pkl_filepath (_type_): _description_
        """
        self.pkl_filepath: str = pkl_filepath
        self.alignment: str = ""
        self.dr_obj = None
        self.dr_transformed: None | np.ndarray = None
        self.dimensions: None | int = None
        self.dm: np.ndarray | None = None
        self.ca_obj = None


def save_workflow(
    workflow: Workflow,
    overwrite: bool = False,
) -> None:
    if os.path.isfile(workflow.pkl_filepath):
        assert overwrite, "{} exists. Instruct to overwrite if required!"
    with open(workflow.pkl_filepath, "wb") as f:
        pickle.dump(workflow, f)


class ensembleClustering(object):
    """Class to take in args from cmd line and assemble a clustering workflow
    up to the creation of clusters. Comparison is done in ensembleComparison
    class.
    Formed of calls to other classes and functions for each stage of the
    processing.
    Each CLI call performs a clustering workflow making use of ALL
    permutations of the specified worflow steps. Call several times
    via CLI to perform a specific workflow. Each workflow is tracked via csv
    file which contains the workflow info and a link to the output clustering
    of the embedded distance matrix (saved as a np.array pkl).

    Args:
        object (_type_): _description_
    """

    def __init__(
        self,
        trajectory: mdtraj.Trajectory,
        results_dir: str,
        workflows_filename: str,
        alignment: list[str],
        distance_metric: list[str],
        dimension_reduction: list[str],
        dimensions: list[int | float | None],
        cluster_alg: list[str],
        clusters: list[int],
        dpi: int = 300,
        pdf: bool = False,
        overwrite: bool = False,
        nopkl: bool = False,
        verbose: bool = False,
    ) -> None:
        self.trajectory = trajectory
        self.dpi = dpi
        self.pdf = pdf
        self.nopkl = nopkl
        self.verbose = verbose
        self.overwrite = overwrite
        self.results_dir = results_dir
        self.workflows_filename = os.path.join(
            self.results_dir,
            workflows_filename,
        )

        self.workflows = determine_workflow_permutations(
            alignment,
            dimension_reduction,
            dimensions,
            distance_metric,
            cluster_alg,
            clusters,
            verbose=self.verbose,
        )
        # determine locations for pkl files to save/load the
        # distance matrix objects
        # determine locations for pkl files to save/load the
        # dimensionality reduced objects
        # determine locations for pkl files to save/load the
        # cluster objects
        self.workflows = determine_workflow_pkl_locations(
            self.results_dir,
            self.workflows,
        )

        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)
        # save the workflows csv from df to file
        self.workflows.to_csv(self.workflows_filename)

    def run_all_workflows(self):
        # defaults cause first workflow to always run all steps
        # rather than reuse 1 or more previous results
        self.last_align = "runfirst"
        self.last_dr = ("runfirst",)
        self.last_dr_dims = ("runfirst",)
        self.last_dm = ("runfirst",)
        self.last_ca = ("runfirst",)
        self.last_ca_nc = "runfirst"
        for workflow_index in range(len(self.workflows)):
            workflow = self.workflows.iloc[workflow_index]
            self.run_workflow(workflow, workflow_index)

    def run_workflow(self, workflow, workflow_index: int):
        """
        if workflow["alignment"] is not None:
            self.run_alignment(workflow["alignment"])
        """
        # create the class to save workflow results as pkl
        wf = Workflow(workflow["pkl"])
        upstream_changed = False

        # ---------------

        # run alignment
        if workflow["alignment"] != self.last_align:
            upstream_changed = True
            self.aligned = self.run_alignment(workflow["alignment"])
        # remaining case is that self.aligned already holds the
        # aligned trajectory, in which case we don't need to do anything

        # note the alignment alg used (assumed too big to save to pkl)
        wf.alignment = workflow["alignment"]
        self.last_align = workflow["alignment"]

        # ---------------

        # run dimension reduction
        if (
            workflow["dimension_reduction"] != self.last_dr
            or workflow["dimensions"] != self.last_dr_dims
            or upstream_changed
        ):
            upstream_changed = True

            (
                self.dr,
                self.transformed_dimensions,
            ) = self.run_dimensionality_reduction(
                dimensionality_reduction=workflow["dimension_reduction"],
                dimensions=workflow["dimensions"],
            )
            if self.verbose:
                print(
                    "Dimensions reduced to {}".format(
                        self.transformed_dimensions.shape
                    )
                )

            self.last_dr = workflow["dimension_reduction"]
            self.last_dr_dims = workflow["dimensions"]

        # visualise embedding
        # in 2d, make a plot
        if self.transformed_dimensions.shape[-1] > 1:
            plot_2d_embedding(
                self.transformed_dimensions[:, 0:2],
                np.arange(self.transformed_dimensions.shape[0]),
                workflow["pkl"].replace(".pkl", "_dr.png"),
                "conformation #",
                dpi=self.dpi,
                pdf=self.pdf,
            )
        wf.dr_obj = self.dr
        wf.dr_transformed = self.transformed_dimensions
        wf.dimensions = workflow["dimensions"]

        # save workflow after each step
        if ~self.nopkl:
            save_workflow(wf, self.overwrite)

        # ---------------

        # apply the distance matrix (or none!)
        if workflow["distance_metric"] != self.last_dm or upstream_changed:
            upstream_changed = True

            if workflow["dimension_reduction"] is not None:
                self.distance_metric = self.run_distance_metric(
                    workflow["distance_metric"],
                    self.transformed_dimensions,
                )
            else:
                self.distance_metric = self.run_distance_metric(
                    workflow["distance_metric"],
                    self.aligned.xyz.reshape(
                        self.aligned.n_frames,
                        self.aligned.n_atoms * 3,
                    ),
                )

            if self.verbose:
                print(
                    "Distance metric has shape: {}".format(
                        self.distance_metric.shape
                    )
                )

            self.last_dm = workflow["distance_metric"]

            # visualise the distance metric
            # if a distance metric was computed
            if self.last_dm != "":
                plot_distancematrix(
                    self.distance_metric,
                    workflow["pkl"].replace(".pkl", "_dm.png"),
                    metric=workflow["distance_metric"],
                    dpi=self.dpi,
                    pdf=self.pdf,
                )

        wf.dm = self.distance_metric

        # save workflow after each step
        # already asserted that either file doesnt exist
        # or that overwrite is asserted
        if ~self.nopkl:
            save_workflow(wf, overwrite=True)

        # ---------------

        # apply clustering algorithm
        if (
            workflow["cluster_alg"] != self.last_ca
            or workflow["clusters"] != self.last_ca_nc
            or upstream_changed
        ):
            upstream_changed = True

            self.cluster_alg = self.run_cluster_alg(
                workflow["cluster_alg"],
                self.distance_metric,
                workflow["clusters"],
            )

            # visualise the clusters using 2 principal component
            # projection of reduced dimensions
            # also visualise cluster index vs conformation index
            # if dimensions were not reduced, only the latter
            if workflow["dimension_reduction"] != "" and self.cluster_alg:
                plot_2d_embedding(
                    self.transformed_dimensions[:, 0:2],
                    self.cluster_alg.labels_,
                    workflow["pkl"].replace(".pkl", "_ca.png"),
                    "cluster index",
                    dpi=self.dpi,
                    pdf=self.pdf,
                )

            self.last_ca = workflow["cluster_alg"]
            self.last_ca_nc = workflow["clusters"]

        wf.ca_obj = self.cluster_alg

        # save workflow after each step
        # already asserted that either file doesnt exist
        # or that overwrite is asserted
        if ~self.nopkl:
            save_workflow(wf, overwrite=True)

    def run_alignment(self, alignment) -> mdtraj.Trajectory:
        if alignment == "":
            aligned = self.trajectory

        if alignment == "superpose":
            # superposition aligns the trajectory on
            # the first frame of the trajectory
            aligned = self.trajectory.superpose(
                self.trajectory,
                0,
            )
        return aligned

    def run_dimensionality_reduction(
        self,
        dimensionality_reduction,
        dimensions,
    ):
        if dimensionality_reduction == "":
            dr_obj = None
            transformed_coords = self.aligned
        if dimensionality_reduction == "pca":
            dr_obj = PCA(n_components=dimensions)

            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            if self.verbose:
                print(
                    "PCA components variance explanation:\n{}".format(
                        dr_obj.explained_variance_ratio_
                    )
                )

            # normalise
            transformed_coords /= np.sqrt(self.aligned.n_atoms)

        # pca for large datasets (memory efficient)
        if dimensionality_reduction == "ipca":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply ipca"
            """
            dr_obj = IncrementalPCA(n_components=dimensions)

            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            if self.verbose:
                print(
                    "IPCA components variance explanation:\n{}".format(
                        dr_obj.explained_variance_ratio_
                    )
                )
            # normalise
            transformed_coords /= np.sqrt(self.aligned.n_atoms)

        # ICA (which autowhitens data as required)
        if dimensionality_reduction == "ica":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply ica"
            """
            dr_obj = FastICA(n_components=dimensions)
            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            # normalise
            # TODO check if normalisation makes sense for ICA
            transformed_coords /= np.sqrt(self.aligned.n_atoms)

        if dimensionality_reduction == "kernelpca":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply kernelpca"
            """
            dr_obj = KernelPCA(
                n_components=dimensions,
                eigen_solver="randomized",
            )

            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            # normalise
            # TODO check if normalisation makes sense for KPCA
            # transformed_coords/=np.sqrt(self.aligned.n_atoms)

        # now onto manifold (non-linear dimension reduction techniques)
        # isomap (tries to maintain geodesic distances)
        if dimensionality_reduction == "isomap":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply isomap"
            """
            dr_obj = Isomap(n_components=dimensions)
            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            # TODO normalise I think should not be done for non-linear
            # embedding, but need to check
            # transformed_coords/=np.sqrt(self.aligned.n_atoms)

        # Locally linear embedding (LLE) seeks a lower-dimensional projection
        # of the data which preserves distances within local neighborhoods
        if dimensionality_reduction == "lle":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply locally linear embedding"
            """
            dr_obj = LocallyLinearEmbedding(n_components=dimensions)
            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            # TODO normalise I think should not be done for non-linear
            # embedding, but need to check
            # transformed_coords/=np.sqrt(self.aligned.n_atoms)

        if dimensionality_reduction == "mlle":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply modified locally linear embedding"
            """
            dr_obj = LocallyLinearEmbedding(
                n_components=dimensions,
                method="modified",
            )
            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            # TODO normalise I think should not be done for non-linear
            # embedding, but need to check
            # transformed_coords/=np.sqrt(self.aligned.n_atoms)

        if dimensionality_reduction == "hlle":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply modified locally linear embedding"
            """
            dr_obj = LocallyLinearEmbedding(
                n_components=dimensions,
                method="hessian",
            )
            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            # TODO normalise I think should not be done for non-linear
            # embedding, but need to check
            # transformed_coords/=np.sqrt(self.aligned.n_atoms)

        if dimensionality_reduction == "ltsa":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply modified locally linear embedding"
            """
            dr_obj = LocallyLinearEmbedding(
                n_components=dimensions,
                method="ltsa",
            )
            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            # TODO normalise I think should not be done for non-linear
            # embedding, but need to check
            # transformed_coords/=np.sqrt(self.aligned.n_atoms)

        # preserves local distance
        if dimensionality_reduction == "spectral":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply modified locally linear embedding"
            """
            dr_obj = SpectralEmbedding(
                n_components=dimensions,
            )
            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            # TODO normalise I think should not be done for non-linear
            # embedding, but need to check
            # transformed_coords/=np.sqrt(self.aligned.n_atoms)

        # mds is for analyzing similarity or dissimilarity data. It attempts
        # to model similarity or dissimilarity data as distances in a
        # geometric spaces. The data can be ratings of similarity between
        # objects, interaction frequencies of molecules, or trade indices
        # between countries.
        if dimensionality_reduction == "mds_metric":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply modified locally linear embedding"
            """
            dr_obj = MDS(
                n_components=dimensions,
            )
            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            # TODO normalise I think should not be done for non-linear
            # embedding, but need to check
            # transformed_coords/=np.sqrt(self.aligned.n_atoms)

        if dimensionality_reduction == "mds_nonmetric":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply modified locally linear embedding"
            """
            dr_obj = MDS(
                n_components=dimensions,
                metric=False,
            )
            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            # TODO normalise I think should not be done for non-linear
            # embedding, but need to check
            # transformed_coords/=np.sqrt(self.aligned.n_atoms)

        # While Isomap, LLE and variants are best suited to unfold a single
        # continuous low dimensional manifold, t-SNE will focus on the local
        # structure of the data and will tend to extract clustered
        # local groups of samples
        # optimisation can be tricky, only using default hyperparams atm
        if dimensionality_reduction == "tsne":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply modified locally linear embedding"
            """
            dr_obj = TSNE(
                n_components=dimensions,
            )
            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            # TODO normalise I think should not be done for non-linear
            # embedding, but need to check
            # transformed_coords/=np.sqrt(self.aligned.n_atoms)

        if dimensionality_reduction == "umap":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply modified locally linear embedding"
            """
            dr_obj = umap.UMAP(
                n_components=dimensions,
            )
            transformed_coords = dr_obj.fit_transform(
                self.aligned.xyz.reshape(
                    self.aligned.n_frames,
                    self.aligned.n_atoms * 3,
                ),
            )
            # TODO normalise I think should not be done for non-linear
            # embedding, but need to check
            # transformed_coords/=np.sqrt(self.aligned.n_atoms)

        return dr_obj, transformed_coords

    def run_distance_metric(self, distance_metric, coords) -> np.ndarray:
        if distance_metric == "":
            distance_matrix = self.transformed_dimensions

        if distance_metric == "approx_rmsd":
            # create the distance matrix
            # distance_matrix = mdtraj_pairwise_rmsd(self.trajectory)
            # instead of rmsd on mdtraj.Trajectory, instead use pdist
            # as pca dimensions are not guaranteed applicable to
            # mdtraj.Trajectory
            distance_matrix = pdist(coords)
            if self.verbose:
                print(
                    "Distance matrix shape: {}".format(distance_matrix.shape)
                )

        if distance_metric == "rmsd":
            rmsd_distance_matrix = []
            # calc rmsd compared to all previous frames
            for frame in range(self.trajectory.n_frames - 1):
                rmsd_distance_matrix.append(
                    mdtraj.rmsd(
                        self.trajectory[frame + 1 :], self.trajectory[frame]
                    )
                )
            rmsd_distance_matrix = np.concatenate(rmsd_distance_matrix)
            distance_matrix = pdist(coords)
            if self.verbose:
                print(
                    "Distance matrix shape: {}".format(distance_matrix.shape)
                )

        return distance_matrix

    def run_cluster_alg(self, cluster_alg, distance_matrix, n_clusters):
        if cluster_alg == "":
            cluster_info = None

        if cluster_alg == "kmeans":
            # kmeans requires original or reduced dims
            # and must not have had a distance metric applied
            assert self.last_dm == "", (
                "To compute kmeans a distance "
                + "metric must not have been used! {} was used".format(
                    self.last_dm
                )
            )
            cluster_info = KMeans(n_clusters=n_clusters)
            cluster_info.fit(distance_matrix)

        if cluster_alg == "mbkmeans":
            # minibatch kmeans requires original or reduced dims
            # and must not have had a distance metric applied
            assert self.last_dm == "", (
                "To compute kmeans a distance"
                + " metric must not have been used! {} was used".format(
                    self.last_dm
                )
            )
            cluster_info = MiniBatchKMeans(n_clusters=n_clusters)
            cluster_info.fit(distance_matrix)

        if cluster_alg == "affinity_euclid":
            # affinity propagation chooses its own n_clusters
            print(
                "When computing affinity propagation the n_clusters"
                " is determined by the clustering algorithm!"
            )
            # using euclidean metric, distance metric MUST NOT
            # be precomputed
            assert self.last_dm == "", (
                "To compute euclidean affinity propagation a distance"
                " metric must not have been used! {} was used".format(
                    self.last_dm
                )
            )
            cluster_info = AffinityPropagation(affinity="euclidean")
            cluster_info.fit(distance_matrix)

        if cluster_alg == "affinity_precomputed":
            # affinity propagation chooses its own n_clusters
            print(
                "When computing affinity propagation the n_clusters"
                " is determined by the clustering algorithm!"
            )
            # using precomputed metric, distance metric MUST NOT
            # be precomputed
            assert self.last_dm != "", (
                "To compute precomputed affinity propagation a distance"
                " metric must have been used! {} was used".format(self.last_dm)
            )
            cluster_info = AffinityPropagation(affinity="precomputed")
            if len(distance_matrix.shape) < 2:
                cluster_info.fit(squareform(distance_matrix))
            else:
                cluster_info.fit(distance_matrix)

        if cluster_alg == "meanshift":
            print(
                "When computing meanshift the n_clusters"
                " is determined by the clustering algorithm!"
            )
            assert self.last_dm == "", (
                "To compute meanshift a distance"
                " metric must not have been used! {} was used".format(
                    self.last_dm
                )
            )
            cluster_info = MeanShift()
            cluster_info.fit(distance_matrix)

        if cluster_alg == "spectral":
            # using spectral clustering, distance metric MUST NOT
            # be precomputed
            assert self.last_dm == "", (
                "To compute spectral clustering a distance"
                " metric must not have been used! {} was used".format(
                    self.last_dm
                )
            )
            cluster_info = SpectralClustering(
                n_clusters=n_clusters,
                affinity="nearest_neighbors",
                n_neighbors=8,
                assign_labels="cluster_qr",
                n_jobs=8,
            )
            cluster_info.fit(distance_matrix)

        """
        if cluster_alg=="spectral_precomputed":
            pass
        """

        if cluster_alg == "ward":
            """
            cluster_info = scipy.cluster.hierarchy.linkage(
                distance_matrix,
                method="ward",
            )
            """
            if self.last_dm == "":
                cluster_info = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage="ward",
                )
                cluster_info.fit(distance_matrix)
            else:
                cluster_info = AgglomerativeClustering(
                    n_clusters,
                    # linkage="ward",
                    metric="precomputed",
                )
                cluster_info.fit(squareform(distance_matrix))

        if cluster_alg == "hier_avg":
            if self.last_dm == "":
                cluster_info = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage="average",
                )
                cluster_info.fit(distance_matrix)
            else:
                cluster_info = AgglomerativeClustering(
                    n_clusters, linkage="average", metric="precomputed"
                )
                cluster_info.fit(squareform(distance_matrix))

        if cluster_alg == "hier_complete":
            if self.last_dm == "":
                cluster_info = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage="complete",
                )
                cluster_info.fit(distance_matrix)
            else:
                cluster_info = AgglomerativeClustering(
                    n_clusters, linkage="complete", metric="precomputed"
                )
                cluster_info.fit(squareform(distance_matrix))

        if cluster_alg == "hier_single":
            if self.last_dm == "":
                cluster_info = AgglomerativeClustering(
                    n_clusters=n_clusters,
                    linkage="single",
                )
                cluster_info.fit(distance_matrix)
            else:
                cluster_info = AgglomerativeClustering(
                    n_clusters, linkage="single", metric="precomputed"
                )
                cluster_info.fit(squareform(distance_matrix))

        if cluster_alg == "dbscan":
            print(
                "When computing DBSCAN the n_clusters"
                " is determined by the clustering algorithm!"
            )
            # using dbscan, distance metric MUST NOT
            # be precomputed
            assert self.last_dm == "", (
                "To compute DBSCAN a distance"
                " metric must not have been used! {} was used".format(
                    self.last_dm
                )
            )
            cluster_info = DBSCAN()
            cluster_info.fit(distance_matrix)

        if cluster_alg == "hdbscan":
            print(
                "When computing HDBSCAN the n_clusters"
                " is determined by the clustering algorithm!"
            )
            # using dbscan, distance metric MUST NOT
            # be precomputed
            assert self.last_dm == "", (
                "To compute HDBSCAN a distance"
                " metric must not have been used! {} was used".format(
                    self.last_dm
                )
            )
            cluster_info = HDBSCAN()
            cluster_info.fit(distance_matrix)

        if cluster_alg == "optics":
            print(
                "When computing OPTICS the n_clusters"
                " is determined by the clustering algorithm!"
            )
            # using dbscan, distance metric MUST NOT
            # be precomputed
            assert self.last_dm == "", (
                "To compute OPTICS a distance"
                " metric must not have been used! {} was used".format(
                    self.last_dm
                )
            )
            cluster_info = OPTICS()
            cluster_info.fit(distance_matrix)

        # GMM
        # https://scikit-learn.org/stable/modules/mixture.html
        # init with kmeans++
        if cluster_alg == "gmm":
            assert self.last_dm == "", "To compute GMM a distance"
            " metric must not have been used! {} was used".format(self.last_dm)
            cluster_info = GaussianMixture(
                n_components=n_clusters,
                covariance_type="full",
                init_params="k-means++",
            )

        return cluster_info


def pilot_study(args):
    """Pilot study to establish sw library interfaces
    for aligning, distance metric, embedding, clustering
    trajectory loading: mdtraj
    aligning: mdtraj rmsd (or np pdist)
    embedding: 2D pca sklearn
    clustering: k-means sklearn
    """
    # using mdtraj to compute RMSD clustering of traj
    # and subsequently pairwise distance clustering of traj???
    # load traj
    traj = mdtraj_load_traj(args)
    # select the non-hydrogen/helium atoms
    traj = traj.atom_slice(traj.topology.select("mass>2.0"))

    # align first -> now inside ensembleClustering
    # traj.superpose(traj, 0)

    # init ensembleClustering
    # determine_workflow_permutations is called
    # determine_workflow_pkl_locations is also called
    ensemble_clustering = ensembleClustering(
        trajectory=traj,
        results_dir=args.output_dir,
        workflows_filename=args.workflows_filename,
        alignment=args.alignment,
        distance_metric=args.distance_metric,
        dimension_reduction=args.dimension_reduction,
        dimensions=args.dimensions,
        cluster_alg=args.cluster_alg,
        clusters=args.n_clusters,
        overwrite=args.overwrite,
        nopkl=args.nopkl,
        verbose=args.verbose,
    )

    # run all workflows
    # this should run all the workflows and save pkl objects
    # at the dimension reduction, distance matrix and cluster stages
    ensemble_clustering.run_all_workflows()

    # do dimension reduction

    """
    # get pairwise rmsd and plot
    pairwise_rmsd = mdtraj_pairwise_rmsd(traj)
    print("Max pairwise rmsd: %f nm" % np.max(pairwise_rmsd))
    plot_mdtraj_pairwise_rmsd(args, pairwise_rmsd)

    # get 2d pca of mdtraj rmsd
    reduced_cartesian = mdtraj_apply_pca(
        traj.xyz.reshape(traj.n_frames, traj.n_atoms * 3),
        2,
        args.verbose,
    )
    # plot 2d pca of mdtraj rmsd
    mdtraj_plot_2d_pca(
        args,
        reduced_cartesian,
        traj,
    )
    """


def avg_linkage_clustering(args, distance_metric: np.ndarray):
    # cluster with average linkage using RMSD distance metric as an example
    # Clustering only accepts reduced form. Squareform's checks are too
    # stringent
    linkage = avg_linkage_cluster(distance_metric, args.verbose)

    # plot dendrogram of avg linkage clustering
    plot_rmsd_dendrogram(args, linkage)


def main(args):
    """Load in conformations (pdb) and compute heterogeneity
    related metrics and (optimal) clustering"""
    pilot_study(args)

    # calc pairwise rmsd with mda standard rmsd
    # first you gotta align the conformations
    # more info on in_mem/file here:
    # https://userguide.mdanalysis.org/1.1.1/examples/analysis/
    # alignment_and_rms/aligning_trajectory.html#Aligning-a-trajectory
    # -with-AlignTraj
    # distancematrix = mda_distancematrix(conformations)
    # plot the distancematrix
    # default metric is rmsd
    # looks like there is no
    # dimension reduction (that is implemented in diffusionmap instead)

    # TODO calc rmsf?

    # TODO apply elbow for optimal number of clusters (sklearn inertia)

    # TODO calc SSAP

    # TODO calc lDDT

    # TODO calc protein cluster conformers metric

    # TODO some pca and analysis ofconvergence?

    # TODO add output csv at minimum allowing matching of HRA latent
    # space to <alg> cluster index


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
