"""Load in latent space encodings from HRA and optionally compute dimension
reduction and perform clustering

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

import numpy as np

from scipy.spatial.distance import squareform
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

from roodmus.heterogeneity.het_metrics import (
    determine_workflow_permutations,
    determine_workflow_pkl_locations,
    Workflow,
    plot_2d_embedding,
    save_workflow,
)


def add_arguments(parser: argparse.ArgumentParser):
    """Parse arguments for performing one or more clustering workflows
    (as configured by provided argument permutations)

    Args:
        parser (argparse.ArgumentParser): _description_

    Returns:
        _type_: _description_
    """
    parser.add_argument(
        "--latent",
        help="latent coordinate file",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--latent_filetype",
        help="HRA used to produce latent coords file",
        choices=[
            "cryodrgn",
            "3dflex",
        ],
        type=str,
        default="cryodrgn",
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

    # store output pkl files
    parser.add_argument(
        "--nopkl",
        help="Disable saving of pkl files containing workflow outputs",
        action="store_true",
    )

    parser.add_argument("--pdf", help="save plot as pdf", action="store_true")

    return parser


def get_name():
    return "latent_clustering"


def load_latent_cryodrgn(latent_file: str):
    with open(latent_file, "rb") as lf:
        z = pickle.load(lf)
        # this should be np.ndarray shape [n_particles, n_latent_dims]
        ndim = z.shape[1]
        # now get all entries in this row so we can add a column to df
        latent = []
        for i in range(ndim):
            latent.append(z[:, i])
        latents = np.stack(latent, axis=1)
        return latents, ndim


def load_latent_cs(latent_file: str):
    latents = np.load(latent_file)
    ndim = len([r for r in latents.dtype.names if "value" in r])
    latent = []
    for i in range(ndim):
        latent.append(latents[f"components_mode_{i}/value"])
    latents = np.stack(latent, axis=1)
    return latents, ndim


def get_latent(latent: str, latent_filetype: str):
    if latent_filetype == "cryodrgn":
        latents, ndim = load_latent_cryodrgn(latent)
        return latents, ndim
    elif latent_filetype == "3dflex":
        latents, ndim = load_latent_cs(latent)
        return latents, ndim


class latentClustering(object):
    def __init__(
        self,
        latent_coords_file: str,
        latent_filetype: str,
        results_dir: str,
        workflows_filename: str,
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
        self.latent_coords_file = latent_coords_file
        self.latent_filetype = latent_filetype
        self.dpi = dpi
        self.pdf = pdf
        self.nopkl = nopkl
        self.verbose = verbose
        self.overwrite = overwrite
        self.results_dir = results_dir
        self.workflows_filename = os.path.join(
            self.results_dir, workflows_filename
        )

        self.workflows = determine_workflow_permutations(
            [""],  # alignment not required
            dimension_reduction,
            dimensions,
            [""],  # distance metric no required
            cluster_alg,
            clusters,
            verbose=self.verbose,
        )
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

        # open the latent space representation
        self.latent, self.latent_ndims = get_latent(
            self.latent_coords_file,
            self.latent_filetype,
        )

    def run_all_workflows(self):
        # defaults cause first workflow to always run all s
        self.last_align = ""
        self.last_dr = ("runfirst",)
        self.last_dr_dims = ("runfirst",)
        self.last_dm = ""
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
            # TODO figure out if
            # np.arange(self.transformed_dimensions.shape[0]),
            # can be replaced easily with an index from truth-mapped confs
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

        # apply clustering algorithm
        if (
            workflow["cluster_alg"] != self.last_ca
            or workflow["clusters"] != self.last_ca_nc
            or upstream_changed
        ):
            upstream_changed = True

            self.cluster_alg = self.run_cluster_alg(
                workflow["cluster_alg"],
                self.transformed_dimensions,
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

    def run_dimensionality_reduction(
        self,
        dimensionality_reduction,
        dimensions,
    ):
        if dimensionality_reduction == "":
            dr_obj = None
            transformed_coords = self.latent
        if dimensionality_reduction == "pca":
            dr_obj = PCA(n_components=dimensions)

            transformed_coords = dr_obj.fit_transform(
                self.latent,
            )
            if self.verbose:
                print(
                    "PCA components variance explanation:\n{}".format(
                        dr_obj.explained_variance_ratio_
                    )
                )

            # no normalisation of latent space by sqrt n_atoms

        # pca for large datasets (memory efficient)
        if dimensionality_reduction == "ipca":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply ipca"
            """
            dr_obj = IncrementalPCA(n_components=dimensions)

            transformed_coords = dr_obj.fit_transform(
                self.latent,
            )
            if self.verbose:
                print(
                    "IPCA components variance explanation:\n{}".format(
                        dr_obj.explained_variance_ratio_
                    )
                )
            # normalise
            # transformed_coords /= np.sqrt(self.aligned.n_atoms)

        # ICA (which autowhitens data as required)
        if dimensionality_reduction == "ica":
            """
            assert isinstance(dimensions, int), "Dimensions must be int"
            " to apply ica"
            """
            dr_obj = FastICA(n_components=dimensions)
            transformed_coords = dr_obj.fit_transform(
                self.latent,
            )
            # normalise
            # TODO check if normalisation makes sense for ICA
            # transformed_coords /= np.sqrt(self.aligned.n_atoms)

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
                self.latent,
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
                self.latent,
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
                self.latent,
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
                self.latent,
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
                self.latent,
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
                self.latent,
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
                self.latent,
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
                self.latent,
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
                self.latent,
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
                self.latent,
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
                self.latent,
            )
            # TODO normalise I think should not be done for non-linear
            # embedding, but need to check
            # transformed_coords/=np.sqrt(self.aligned.n_atoms)

        return dr_obj, transformed_coords

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
            # using precomputed metric, distance metric MUST
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


def pilot_study_latent(args):
    # init latentClustering
    # determine_workflow_permutations is called
    # determine_workflow_pkl_locations is also called
    latent_clustering = latentClustering(
        latent_coords_file=args.latent,
        latent_filetype=args.latent_filetype,
        results_dir=args.output_dir,
        workflows_filename=args.workflows_filename,
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
    latent_clustering.run_all_workflows()


def main(args):
    pilot_study_latent(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
