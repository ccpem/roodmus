"""
    API containing functions to perform analysis on a reconstruction workflow.
"""

import os
import time
from typing import Tuple

import numpy as np
from tqdm import tqdm
import pandas as pd
from scipy.spatial import cKDTree

from roodmus.analysis.utils import IO


class particle_picking(object):
    def __init__(
        self,
        meta_file: str,
        config_dir: str,
        particle_diameter: float,
        results_picking: dict | None = None,
        results_truth: dict | None = None,
        verbose: bool = False,
    ):
        self.meta_file = meta_file
        self.config_dir = config_dir
        self.particle_diameter = particle_diameter
        self.verbose = verbose
        if results_picking is not None:
            self.results_picking = results_picking
        else:
            self.results_picking = {
                "metadata_filename": [],
                "ugraph_filename": [],
                "position_x": [],
                "position_y": [],
                "ugraph_shape": [],
                "TP": [],
                "defocus": [],
            }

        if results_truth is not None:
            self.results_truth = results_truth
        else:
            self.results_truth = {
                "metadata_filename": [],
                "ugraph_filename": [],
                "pdb_filename": [],
                "position_x": [],
                "position_y": [],
                "multiplicity": [],
                "defocus": [],
            }
        # Misc attributes
        # stores the paths to the micrographs from the metadata file
        self.ugraph_paths: list[str] = []
        # stores the number of micrographs that have been loaded
        self.num_ugraphs_loaded = 0
        # stores the number of particles per micrograph
        self.particles_per_ugraph: list[int] = []

        self.compute()

    def compute(
        self,
        meta_file: str = "",
        config_dir: str = "",
        verbose: bool = False,
    ):
        """
        Processing of the particle positions from a .cs or .star file
         and from the Parakeet config files.
        """

        # if the user has provided new arguments, use them
        update_meta_file = False
        update_config_dir = False
        if meta_file is not None and meta_file is not self.meta_file:
            # user has specified a new metadata file to use
            self.meta_file = meta_file
            update_meta_file = True
        elif meta_file is None:
            # user has not specified a new metadata file to use,
            # but the current one is not valid
            update_meta_file = True
        if config_dir is not None and config_dir is not self.config_dir:
            # user has specified a new config directory to use
            self.config_dir = config_dir
            update_config_dir = True
        elif config_dir is None:
            # user has not specified a new config directory to use,
            # but the current one is not valid
            update_config_dir = True
        if verbose is not None:
            self.verbose = verbose

        # load the metadata
        if update_meta_file:
            if self.verbose:
                print(f"Loading metadata from {self.meta_file}...")
            metadata, file_type = self._load_metadata(
                self.meta_file, self.verbose
            )
            (
                ugraph_paths,
                positions,
                ctf_params,
                ugraph_shapes,
            ) = self._extract_from_metadata(metadata, file_type, self.verbose)
            if ctf_params is None:
                ctf_params = np.ones((len(positions), 1)) * np.nan
            self.ugraph_paths = list(
                np.unique(np.concatenate((self.ugraph_paths, ugraph_paths)))
            )
            if self.verbose:
                print(
                    "Found {} particles in {} micrographs".format(
                        len(positions),
                        len(np.unique(self.ugraph_paths)),
                    )
                )

            for ugraph_path, position, ctf_param, ugraph_shape in zip(
                ugraph_paths, positions, ctf_params, ugraph_shapes
            ):
                self.results_picking["metadata_filename"].append(
                    self.meta_file
                )
                self.results_picking["ugraph_filename"].append(ugraph_path)
                self.results_picking["position_x"].append(position[0])
                self.results_picking["position_y"].append(position[1])
                self.results_picking["ugraph_shape"].append(ugraph_shape)
                self.results_picking["defocus"].append(ctf_param[0])

            if self.verbose:
                print(
                    "Loaded {} micrographs from {}".format(
                        len(self.ugraph_paths),
                        self.meta_file,
                    )
                )
                print(
                    "Dictionaries now contain {} particles"
                    " and {} true particles".format(
                        len(self.results_picking["ugraph_filename"]),
                        len(self.results_truth["ugraph_filename"]),
                    )
                )

        # load the config files
        if (
            update_config_dir
            or len(self.ugraph_paths) > self.num_ugraphs_loaded
        ):
            # either the config dir has changed,
            # or the number of micrographs has increased
            progressbar = tqdm(
                total=len(self.ugraph_paths),
                desc="loading ground-truth particle positions",
            )
            num_particles = {}
            for ugraph_path in self.ugraph_paths:
                config = IO.load_config(
                    os.path.join(
                        self.config_dir, ugraph_path.replace(".mrc", ".yaml")
                    )
                )
                (
                    filenames,
                    positions,
                    defocus_values,
                ) = self._extract_from_config(config, self.verbose)
                num_particles[ugraph_path] = len(filenames)
                for filename, position, defocus_value in zip(
                    filenames, positions, defocus_values
                ):
                    self.results_truth["metadata_filename"].append(
                        self.meta_file
                    )
                    self.results_truth["ugraph_filename"].append(ugraph_path)
                    self.results_truth["pdb_filename"].append(filename)
                    self.results_truth["position_x"].append(position[0])
                    self.results_truth["position_y"].append(position[1])
                    self.results_truth["defocus"].append(defocus_value)

                _ = progressbar.update(1)
            progressbar.close()

            # update the number of micrographs loaded and the
            # number of particles per micrograph
            self.num_ugraphs_loaded = len(self.ugraph_paths)
            self.particles_per_ugraph = [
                num_particles[ugraph_path] for ugraph_path in self.ugraph_paths
            ]

            if self.verbose:
                print(
                    "Loaded ground-truth particle positions"
                    " from config files"
                )
                print(self.particles_per_ugraph)
                print(
                    "Dictionaries now contain {} particles"
                    " and {} true particles".format(
                        len(self.results_picking["ugraph_filename"]),
                        len(self.results_truth["ugraph_filename"]),
                    )
                )

        elif update_meta_file:
            # Same config dir, new metadata file.
            # Need to copy the values from before
            progressbar = tqdm(
                total=len(np.unique(ugraph_paths)),
                desc="loading ground-truth particle positions",
            )
            for ugraph_path in np.unique(
                ugraph_paths
            ):  # loop over the new set of micrographs
                pdb_filenames = [
                    self.results_truth["pdb_filename"][i]
                    for i in range(len(self.results_truth["pdb_filename"]))
                    if self.results_truth["ugraph_filename"][i] == ugraph_path
                ][
                    : self.particles_per_ugraph[
                        self.ugraph_paths.index(ugraph_path)
                    ]
                ]
                positions = np.array(
                    [
                        [
                            self.results_truth["position_x"][i],
                            self.results_truth["position_y"][i],
                        ]
                        for i in range(len(self.results_truth["position_x"]))
                        if self.results_truth["ugraph_filename"][i]
                        == ugraph_path
                    ]
                )[
                    : self.particles_per_ugraph[
                        self.ugraph_paths.index(ugraph_path)
                    ]
                ]
                defocus_values = [
                    self.results_truth["defocus"][i]
                    for i in range(len(self.results_truth["defocus"]))
                    if self.results_truth["ugraph_filename"][i] == ugraph_path
                ][
                    : self.particles_per_ugraph[
                        self.ugraph_paths.index(ugraph_path)
                    ]
                ]
                for pdb_filename, position, defocus_value in zip(
                    pdb_filenames, positions, defocus_values
                ):
                    self.results_truth["metadata_filename"].append(
                        self.meta_file
                    )
                    self.results_truth["ugraph_filename"].append(ugraph_path)
                    self.results_truth["pdb_filename"].append(pdb_filename)
                    self.results_truth["position_x"].append(position[0])
                    self.results_truth["position_y"].append(position[1])
                    self.results_truth["defocus"].append(defocus_value)

                _ = progressbar.update(1)
            progressbar.close()

            if self.verbose:
                print(
                    "Dictionaries now contain {} particles"
                    " and {} true particles".format(
                        len(self.results_picking["ugraph_filename"]),
                        len(self.results_truth["ugraph_filename"]),
                    )
                )

        # For each picked particle,
        # check if it is near any of the true particles
        # Only need to do this if the metadata file has changed
        if update_meta_file:
            if self.verbose:
                print(
                    "Calculating true positive picked particles"
                    " and unpicked truth particles..."
                )
            progressbar = tqdm(
                total=len(np.unique(self.ugraph_paths)),
                desc=(
                    "Calculating true positive picked particles and"
                    " unpicked truth particles"
                ),
            )
            # number of micrographs with no picked particles
            unpicked_ugraphs = 0
            for ugraph_path in np.unique(self.ugraph_paths):
                pos_picked_in_ugraph = []
                for i in range(len(self.results_picking["ugraph_filename"])):
                    if (
                        self.results_picking["ugraph_filename"][i]
                        == ugraph_path
                        and self.results_picking["metadata_filename"][i]
                        == self.meta_file
                    ):
                        pos_picked_in_ugraph.append(
                            [
                                self.results_picking["position_x"][i],
                                self.results_picking["position_y"][i],
                            ]
                        )
                if pos_picked_in_ugraph == []:
                    # print("no particles were picked in this micrograph")
                    unpicked_ugraphs += 1
                    _ = progressbar.update(1)
                    continue  # no particles were picked in this micrograph
                pos_picked_in_ugraph = np.array(pos_picked_in_ugraph)

                pos_truth_in_ugraph = []
                for i in range(len(self.results_truth["ugraph_filename"])):
                    if (
                        self.results_truth["ugraph_filename"][i] == ugraph_path
                        and self.results_truth["metadata_filename"][i]
                        == self.meta_file
                    ):
                        pos_truth_in_ugraph.append(
                            [
                                self.results_truth["position_x"][i],
                                self.results_truth["position_y"][i],
                            ]
                        )
                # if pos_truth_in_ugraph == []:
                #     print("no particles were picked in this micrograph")
                #     _ = progressbar.update(1)
                #     continue
                pos_truth_in_ugraph = np.array(pos_truth_in_ugraph)

                ugraph_shape = next(
                    val
                    for r, val in enumerate(
                        self.results_picking["ugraph_shape"]
                    )
                    if self.results_picking["ugraph_filename"][r]
                    == ugraph_path
                )

                dist_array = self._calc_dist_array(
                    pos_picked_in_ugraph, pos_truth_in_ugraph, ugraph_shape
                )

                # a picked particle is considered a true positive if it is
                # closer than 1 particle radius to any of the true particles
                for particle in range(len(pos_picked_in_ugraph)):
                    self.results_picking["TP"].append(
                        np.any(
                            dist_array[particle] < self.particle_diameter / 2
                        )
                    )

                # count the number of picked particles close to
                # a true particle (this is the true particle's multiplicity)
                for particle in range(len(pos_truth_in_ugraph)):
                    self.results_truth["multiplicity"].append(
                        np.sum(
                            dist_array[:, particle]
                            < self.particle_diameter / 2
                        )
                    )

                _ = progressbar.update(1)
            progressbar.close()

            if self.verbose:
                print(f"{unpicked_ugraphs} micrographs were not picked")

    def _load_metadata(
        self, meta_file: str, verbose: bool = False
    ) -> Tuple[dict, str]:
        if meta_file.endswith(".star"):
            metadata = IO.load_star(meta_file)
            file_type = "star"
        elif meta_file.endswith(".cs"):
            metadata = IO.load_cs(meta_file)
            file_type = "cs"
        else:
            raise ValueError(f"Unknown metadata file type: {meta_file}")

        if verbose:
            print(
                "Loaded metadata from {}."
                " Determined file type: {}".format(
                    meta_file,
                    file_type,
                )
            )
        return metadata, file_type

    def _extract_from_metadata(self, metadata, file_type, verbose=False):
        if file_type == "cs":
            ugraph_paths = IO.get_ugraph_cs(
                metadata
            )  # a list of all microraps in the metadata file
            positions = IO.get_positions_cs(
                metadata
            )  # an array of all the defocus values in the metadata file
            ugraph_shape = IO.get_ugraph_shape_cs(
                metadata
            )  # the shape of the micrograph
            ctf = IO.get_ctf_cs(
                metadata
            )  # an array of all the defocus values in the metadata file
        elif file_type == "star":
            ugraph_paths = IO.get_ugraph_star(
                metadata
            )  # a list of all microraps in the metadata file
            positions = IO.get_positions_star(
                metadata
            )  # an array of all the defocus values in the metadata file
        else:
            raise ValueError(f"unknown metadata file type: {file_type}")

        if verbose:
            print("Extracted ugraph paths and positions from metadata")
            print(positions.shape)
            print(len(ugraph_paths))
        return ugraph_paths, positions, ctf, ugraph_shape

    def _extract_from_config(self, config: dict, verbose: bool = False):
        defocus = config["microscope"]["lens"]["c_10"]
        pixel_size = config["microscope"]["detector"]["pixel_size"]
        positions = []
        filenames = []
        for molecules in config["sample"]["molecules"]["local"]:
            f = molecules["filename"]
            for instance in molecules["instances"]:
                position = instance["position"]
                positions.append(position)
                filenames.append(f)

        # convert to pixels
        positions = np.array(positions) / pixel_size
        defocus_list = [defocus] * len(positions)
        return filenames, positions, defocus_list

    def _calc_dist_array(self, pos_picked, pos_truth, image_shape):
        # calculates an array of distances between each pair
        # of truth and picked particles
        r = np.sqrt(
            np.power(
                float(image_shape[0]),
                2,
            )
            + np.power(
                float(image_shape[1]),
                2,
            )
        )
        truth_centres = self._grab_xy_array(pos_truth)
        picked_centres = self._grab_xy_array(pos_picked)
        sdm = (
            cKDTree(picked_centres)
            .sparse_distance_matrix(cKDTree(truth_centres), r)
            .toarray()
        )
        sdm[sdm < np.finfo(float).eps] = np.nan
        return sdm

    def _grab_xy_array(self, pos):
        # create a np array to ckdtree for number of neighbours
        # as function of distance
        particle_centres = np.empty(
            (int(len(pos)) + int(len(pos))),
            dtype=float,
        )
        particle_centres[0::2] = pos[:, 0]
        particle_centres[1::2] = pos[:, 1]
        particle_centres = particle_centres.reshape(
            int(len(particle_centres) / 2),
            2,
        )
        return particle_centres

    def compute_precision(
        self,
        results_picking: pd.DataFrame,
        results_truth: pd.DataFrame,
        verbose: bool = False,
    ):
        """Produces another data frame containing the number of
         true positives, false positives, false negatives
         and the precision, recall and multiplicity for each
         micrograph. The output then is a data frame with rows
         corresponding to each micrograph instead of each particle

        Args:
            results_picking (pandas.DataFrame): the results of
             the picking algorithm
            results_truth (pandas.DataFrame): the results of the
             truth algorithm
            verbose (bool, optional): print out information about
             the calculation. Defaults to False.

        Returns:
            pandas.DataFrame: a data frame containing the precision,
             recall and multiplicity for each micrograph
        """

        tt = time.time()
        # get the number of true positives, false positives and
        # false negatives for each micrograph for each metadata file
        ugraph_filename = results_picking.groupby(
            ["metadata_filename", "ugraph_filename"]
        )["ugraph_filename"].first()
        TP = results_picking.groupby(["metadata_filename", "ugraph_filename"])[
            "TP"
        ].sum()
        FP = (
            results_picking.groupby(["metadata_filename", "ugraph_filename"])[
                "TP"
            ].count()
            - TP
        )
        multiplicity = results_truth.groupby(
            ["metadata_filename", "ugraph_filename"]
        )["multiplicity"]
        # multiplicity is the number of times a truth particle is picked.
        # FN can be calculated from this by looking at the number of particles
        # with a multiplcity of 0
        FN = multiplicity.apply(lambda x: np.sum(x == 0))
        defocus = results_truth.groupby(
            ["metadata_filename", "ugraph_filename"]
        )["defocus"].mean()
        metadata_filename = results_picking.groupby(
            ["metadata_filename", "ugraph_filename"]
        )["metadata_filename"].first()

        """
        FP = results_picking.groupby(
            "ugraph_filename"
        )["TP"].count() - TP
        multiplicity = results_truth.groupby(
            "ugraph_filename"
        )["multiplicity"]
        # Multiplicity is the number of times a truth particle is picked.
        # FN can be calculated from this by looking at the number of particles
        # with a multiplcity of 0
        # FN = multiplicity.apply(lambda x: np.sum(x == 0))
        # defocus = results_picking.groupby(
            "ugraph_filename"
        )["defocus"].mean()
        # metadata_filename = results_picking.groupby(
            "ugraph_filename"
        )["metadata_filename"].first()
        """

        # calculate the precision, recall and multiplicity for each micrograph
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        average_multiplicity = results_truth.groupby(
            ["metadata_filename", "ugraph_filename"]
        )["multiplicity"].mean()

        # create a data frame containing the precision,
        # recall and multiplicity for each micrograph
        results = pd.DataFrame(
            {
                "metadata_filename": metadata_filename,
                "ugraph_filename": ugraph_filename,
                "precision": precision,
                "recall": recall,
                "avg_multiplicity": average_multiplicity,
                "TP": TP,
                "FP": FP,
                "FN": FN,
                "defocus": defocus,
            }
        )
        if verbose:
            print(results)
            print(
                "Computed precision, recall and multiplicity"
                " for each micrograph in {} seconds".format(
                    time.time() - tt,
                )
            )
        return results
