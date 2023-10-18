"""Load in conformations (pdb) and compute heterogeneity
related metrics and (optimal) clustering

Approach 1 - Using No Dimensionality Reduction
1. Calc distance matrix
2. Cluster with distance matrix-applicable clustering algorithms
like k-medioids or agglomerative hierarchical clustering
2.1 Investigate methods which will allow error calculations... (which
might include GMM or bootstrapping of other methods). And look at
Jensen-Shannon divergence matrices
3. Get optimal n clusters with inertia/elbow method (if possible)
4. Organise distance matrix by distance from cluster centre on 1 axis and
plot against frame index to show boundaries between clusters. Color based
on cluster index OR make 1 distance plot per cluster and color based on
cluster index?
5. Calc FSC metric? in addition to distance metrics?

Approach 2 - Using Dimensionality Reduction
1. Apply dimensionality reduction (3d PCA on 3d data does what????)
2. Cluster embedding with algs
3.

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

from itertools import combinations

import glob2 as glob
import MDAnalysis as mda
from MDAnalysis.analysis import align, diffusionmap
from MDAnalysis.analysis import encore
from MDAnalysis.analysis.encore.clustering import ClusteringMethod as clm
from MDAnalysis.analysis.encore.dimensionality_reduction import (
    DimensionalityReductionMethod as drm,
)
import matplotlib.pyplot as plt

# import matplotlib.cm as cm
import numpy as np
import mdtraj
import scipy.cluster.hierarchy

# from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA


def add_arguments(parser):
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
        help="Limit to first <n_confs> conformations",
        type=int,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--verbose", help="increase output verbosity", action="store_true"
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


def mda_load_universe(args) -> mda.Universe:
    conf_files = get_pdb_list(
        args.conformations_dir,
        args.file_ext,
    )
    # TODO support option to load pdb files from .star or .npz (cs)?
    # sort alphanumerically
    conf_files = sorted(conf_files)
    # limit number of conformations if desired
    if args.n_confs:
        conf_files = conf_files[: args.n_confs]
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
    # sort alphanumerically
    conf_files = sorted(conf_files)
    # limit number of conformations if desired
    if args.n_confs:
        conf_files = conf_files[: args.n_confs]
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
    args,
    vector_1_3: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    # get pca ou of mdtraj inputs
    pca1 = PCA(n_components=2)
    reduced_cartesian = pca1.fit_transform(vector_1_3)
    if args.verbose:
        print(reduced_cartesian.shape)
    return reduced_cartesian


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
) -> mda.analysis.DistanceMatrix:
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


def main(args):
    """Load in conformations (pdb) and compute heterogeneity
    related metrics and (optimal) clustering"""

    # load the trajectory as an mda.Universe
    conformations = mda_load_universe(args)

    # using mdtraj to compute RMSD clustering of traj
    # and subsequently pairwise distance clustering of traj???
    # load traj
    traj = mdtraj_load_traj(args)

    # get pairwise rmsd and plot
    pairwise_rmsd = mdtraj_pairwise_rmsd(traj)
    print("Max pairwise rmsd: %f nm" % np.max(pairwise_rmsd))
    plot_mdtraj_pairwise_rmsd(args, pairwise_rmsd)

    # cluster with average linkage using RMSD distance metric as an example
    # Clustering only accepts reduced form. Squareform's checks are too
    # stringent
    linkage = avg_linkage_cluster(pairwise_rmsd, args.verbose)

    # plot dendrogram of avg linkage clustering
    plot_rmsd_dendrogram(args, linkage)

    # get 2d pca of mdtraj rmsd
    # align first
    traj.superpose(traj, 0)
    reduced_cartesian = mdtraj_apply_pca(
        args,
        traj.xyz.reshape(traj.n_frames, traj.n_atoms * 3),
        2,
    )
    # plot 2d pca of mdtraj rmsd
    mdtraj_plot_2d_pca(
        args,
        reduced_cartesian,
        traj,
    )

    # now also look at using pairwise distances as a distance metric
    traj.superpose(traj, 0)
    pairwise_distances = mdtraj_pairwise_distances(
        args,
        traj,
        10000,
    )
    # apply 2d pca
    reduced_distances = mdtraj_apply_pca(
        args,
        pairwise_distances,
        2,
    )
    mdtraj_plot_2d_pca(
        args,
        reduced_distances,
        traj,
    )
    """
    labels are generic in the plot function for now
    TODO improve plot label customisation
    plt.scatter(
        reduced_distances[:, 0],
        reduced_distances[:,1],
        marker='x',
        c=traj.time
    )
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Pairwise distance PCA')
    cbar = plt.colorbar()
    mdtraj_2dpca_pdist_figname = os.path.join(
        args.output_dir,
        "mdtraj_2dpca_pdist"
    )
    plt.savefig(
        mdtraj_2dpca_pdist_figname+".png",
        dpi=args.dpi
    )
    if args.pdf:
        plt.savefig(
        mdtraj_2dpca_pdist_figname+".pdf",
    )
    """

    # calc pairwise rmsd with mda standard rmsd
    # first you gotta align the conformations
    # more info on in_mem/file here:
    # https://userguide.mdanalysis.org/1.1.1/examples/analysis/
    # alignment_and_rms/aligning_trajectory.html#Aligning-a-trajectory
    # -with-AlignTraj
    distancematrix = mda_distancematrix(conformations)
    # plot the distancematrix
    # default metric is rmsd
    # looks like there is no
    # dimension reduction (that is implemented in diffusionmap instead)
    plot_mda_distancematrix(
        args,
        distancematrix.dist_matrix,
    )

    # TODO calc rmsf?

    # POSSIBLE FULL TRAJECTORY CLUSTERING -> DISTANCE PLOT ORGANISED BY
    # CLUSTER INDEX
    # use clm KMeans, AffinityPropagation??? and DBScan
    # to cluster trajectory into N clusters
    # rearrange distance matrix along y axis by increasing cluster index
    # with each cluster having its entries arranged by increasing
    # distance from the centre of the cluster
    # Leave x axis as frame index from 0 to -1

    # cluster the rmsd with k-means
    # use dimenstionality reduction method to cluster
    # using a range of different PCA dimensions
    # set up PCA dimension reduction algs
    pc2 = clm.KMeans(
        2,
    )
    ces0, details0 = encore.ces(
        [conformations],
        select="name CA",
        # clustering_method=[pc1, pc2, pc3],
        clustering_method=[pc2],
        ncores=args.n_cores,
    )
    # this aligns and calcs similarity matrix
    # as well as calculating the clusters
    # ces0 is the similarity matrix, with dimensions
    # [len(conformations)][len(clustering_methods)]
    # so can remove the above alignment and calculation code
    # details0 holds encore.clustering.ClusterCollection.ClusterCollection
    print(ces0)
    print(details0)
    print(details0["clustering"])
    print(details0["clustering"][0])
    print(details0["clustering"][0].clusters)
    cluster_list = details0["clustering"][0].clusters
    print(details0["clustering"][0].clusters[0].id)
    print(details0["clustering"][0].clusters[0].elements)

    colors = ["red", "blue", "yellow", "black"]
    pca_figname = os.path.join(args.output_dir, "pca_2d")
    pca_2d_clusters = zip(
        [cluster.id for cluster in cluster_list],
        [cluster.elements for cluster in cluster_list],
    )
    for cluster, elems in pca_2d_clusters:
        plt.plot(elems, color=colors[cluster])
    plt.savefig(
        pca_figname + ".png",
        dpi=args.dpi,
    )
    if args.pdf:
        plt.savefig(pca_figname + ".pdf")
    plt.clf()
    # in summary there was no dimensionality reduction here
    # so no good trying to plot 2d coords in space.
    # What you actually did was clusters trajectories into
    # 2 different clusters (because you set kmeans to have
    # 2 centroids)

    # POSSIBLE CLUSTERING OF MD TRAJECTORIES IN N DIMENSIONAL SPACE
    # VIA DIMENSIONALITY REDUCTION
    # reduce

    # set dimensionality reduction PCAs
    # pc1 = drm.PrincipalComponentAnalysis(dimension=1, svd_solver="auto")
    pc2 = drm.PrincipalComponentAnalysis(dimension=2, svd_solver="auto")
    # pc3 = drm.PrincipalComponentAnalysis(dimension=3, svd_solver="auto")
    dres0, dres_details0 = encore.dres(
        [conformations],
        select="name CA",
        # dimensionality_reduction_method=[pc1,pc2,pc3],
        dimensionality_reduction_method=[pc2],
        ncores=args.n_cores,
    )
    print(dres_details0)
    reduced_coords = dres_details0["reduced_coordinates"][0]
    print(reduced_coords)
    index = np.arange(len(reduced_coords[0]))
    plt.scatter(reduced_coords[0], reduced_coords[1], c=index, cmap="viridis")

    plt.colorbar()
    pca_figname = os.path.join(args.output_dir, "pca_2d")
    plt.savefig(
        pca_figname + ".png",
        dpi=args.dpi,
    )
    if args.pdf:
        plt.savefig(pca_figname + ".pdf")
    plt.clf()

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
