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

import numpy as np
import pandas as pd

from roodmus.analysis.utils import load_data
from roodmus.heterogeneity.het_metrics import get_pdb_list

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
        " Provide at least 2 twice so similarity can be calculated!",
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

    parser.add_argument(
        "--js",
        help="Compute jensen-shannon metric between ensembles",
        action="store_true",
    )

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

    return parser


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
    conformation_files: list[str], digits: int = 6, file_ext: str = ".pdb"
) -> list[str]:
    indices = []
    slice_2 = -len(file_ext)
    slice_1 = slice_2 - digits
    for conf_file in conformation_files:
        indices.append(conf_file[-slice_1:-slice_2])
    return indices


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

    # extract the cluster indices from the pkl files
    # easiest to use a dict for keeping track
    cluster_indices: dict[str, pd.DataFrame] = {}

    # get the conformation filenames
    conformation_filenames = get_pdb_list(args.config_dir)

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
            )
            # now we have str version of digits-length index
            # which can be used to match to a conformation
            cluster_indices[cluster_file] = pd.DataFrame(
                [conformation_indices, workflow.ca_obj.labels_],
                columns=["conformation_index", "cluster_index"],
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
            )
            cluster_indices[cluster_file] = pd.DataFrame(
                [conformation_indices, workflow.ca_obj.labels_],
                columns=["conformation_index", "cluster_index"],
            )

        else:
            raise ValueError(
                "Mist use either `MD` or `latent` as the pkl_source values"
            )

    # now we have the conformation index and cluster index labels,
    # load up all the conformations from conformations_dir into an
    # MDAnalysis Universe and then create a sub-universe for each cluster

    if args.js:
        # compute js between all the provided pkls
        compute_js()

    """
    if args.hd:
        # compute hausdorf (max distance) between all provided pkls
        compute_hd()
    """
    # save results as a human readable csv or np array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
