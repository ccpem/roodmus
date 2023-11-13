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
        "--clusters",
        "-c",
        help=".pkl file containing clustered conformations."
        " Provide at least 2 twice so similarity can be calculated!",
        nargs="+",
        type=str,
        default=[""],
    )

    parser.add_argument(
        "--js",
        help="Compute jensen=shannon metric between ensembles",
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


def main(args):
    """Load in pkl files and compare similarity via jensen-shannon

    Args:
        args (_type_): _description_
    """

    # check that at least 2 cluster pkls were provided
    assert len(args.clusters) >= 2

    if args.js:
        # compute js between all the provided pkls
        compute_js()
    if args.hd:
        # compute hausdorf (max distance) between all provided pkls
        compute_hd()

    # save results as a human readable csv or np array


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
