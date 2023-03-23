"""API containing functions to perform analysis on a reconstruction workflow.
"""

import os
import time
from collections.abc import KeysView as dict_keys

import numpy as np
from tqdm import tqdm
from scipy.spatial import cKDTree
import pandas as pd
from typing import Tuple, Any, List

from .utils import IO


class particle_picking(object):
    def __init__(
        self,
        meta_file: str,
        config_dir: str,
        particle_diameter: float,
        ugraph_shape: Tuple[int, int] = (4000, 4000),
        results_picking: dict | None = None,
        results_truth: dict | None = None,
        verbose: bool = False,
    ):
        self.meta_file = meta_file
        self.config_dir = config_dir
        self.particle_diameter = particle_diameter
        self.ugraph_shape = ugraph_shape
        self.verbose = verbose
        self.results_picking: dict[str, Any] = {
            "metadata_filename": [],
            "ugraph_filename": [],
            "position_x": [],
            "position_y": [],
            "ugraph_shape": [],
            "defocus": [],
            "class2D": [],
        }
        if results_picking is not None:
            self.results_picking = results_picking

        self.results_truth: dict[str, List[Any]] = {
            "ugraph_filename": [],
            "ice_thickness": [],
            "pdb_filename": [],
            "position_x": [],
            "position_y": [],
            "position_z": [],
            "defocus": [],
        }
        if results_truth is not None:
            self.results_truth = results_truth

        # Misc attributes
        # Stores the paths to the micrographs from the metadata file
        self.ugraph_paths: list[str] = []

        # If a new metadata file is given, the values in the picking results
        # need to be extracted from the new metadata file
        self._update_meta_file = True

        # compute the results
        self.compute()

    def compute(
        self,
        meta_file: str = "",
        config_dir: str = "",
        verbose: bool = False,
    ):
        """Processing of the particle positions from a .cs or .star file and
        from the Parakeet config files.
        """

        # The user can specify to use a new metadata file or load config
        # files from a new directory

        # If the user does not specify a new metadata file or config directory
        # the current one is used (an initial metadata file and config dir is
        # given when the class is instantiated)
        if meta_file and meta_file is not self.meta_file:
            # user has specified a new metadata file to use
            self.meta_file = meta_file
            self._update_meta_file = True
        if config_dir and config_dir is not self.config_dir:
            # user has specified a new config directory to use
            self.config_dir = config_dir
            self._update_config_dir = True

        # updates the level of verbosity
        if verbose is not None:
            self.verbose = verbose

        # If the user has specified a new metadata file,
        # the values in the picking results need to be extracted
        # from the new metadata file
        if self._update_meta_file:
            if self.verbose:
                print(f"loading metadata from {self.meta_file}...")
            metadata, file_type = self._load_metadata(
                self.meta_file, self.verbose
            )

            # Getting the values from the metadata file and
            # returning them as lists or nd.arrays

            # Adds the values to the picking results,
            # returns the number of particles added
            num_particles = self._extract_from_metadata(
                metadata, file_type, self.verbose
            )
            if self.verbose:
                print("\n")
                print(
                    "Dictionaries now contain"
                    " {} particles and {} true particles".format(
                        len(self.results_picking["ugraph_filename"]),
                        len(self.results_truth["ugraph_filename"]),
                    )
                )
                print(f"added {num_particles} particles from {self.meta_file}")
            self.results_picking["metadata_filename"].extend(
                [self.meta_file] * num_particles
            )  # add the metadata file to the picking results

        # next, check if any new ugraphs need to be loaded
        ugraphs_to_load = np.unique(
            [
                ugraph_path
                for ugraph_path in self.results_picking["ugraph_filename"]
                if ugraph_path not in self.ugraph_paths
            ]
        )
        if len(ugraphs_to_load) > 0:
            total_num_particles = 0
            progressbar = tqdm(
                total=len(ugraphs_to_load), desc="loading micrographs"
            )
            for ugraph_path in ugraphs_to_load:
                if not os.path.isfile(
                    os.path.join(
                        self.config_dir, ugraph_path.replace(".mrc", ".yaml")
                    )
                ):
                    print(f"WARNING: no config file found for {ugraph_path}")
                    continue
                config = IO.load_config(
                    os.path.join(
                        self.config_dir, ugraph_path.replace(".mrc", ".yaml")
                    )
                )

                # adds the values to the truth results,
                # returns the number of particles added
                num_particles = self._extract_from_config(config, self.verbose)
                total_num_particles += num_particles

                # add the micrograph path and the metadata file to the
                # truth results
                self.results_truth["ugraph_filename"].extend(
                    [ugraph_path] * num_particles
                )

                progressbar.update(1)
                progressbar.set_postfix({"micrograph": ugraph_path})
            progressbar.close()

            # update the list of loaded micrographs
            self.ugraph_paths.extend(ugraphs_to_load)

            if self.verbose:
                print(
                    "Loaded ground-truth particle positions from config files"
                )
                print(
                    "Dictionaries now contain"
                    " {} particles and {} true particles".format(
                        len(self.results_picking["ugraph_filename"]),
                        len(self.results_truth["ugraph_filename"]),
                    )
                )
                print(
                    "Added {} particles from {}".format(
                        total_num_particles, self.config_dir
                    )
                )
        return

    def _load_metadata(
        self, meta_file: str = "", verbose: bool = False
    ) -> Tuple[dict, str]:
        print(meta_file)
        if meta_file.endswith(".star"):
            metadata = IO.load_star(meta_file)
            file_type = "star"
        elif meta_file.endswith(".cs"):
            metadata = IO.load_cs(meta_file)
            file_type = "cs"
        else:
            raise ValueError(f"unknown metadata file type: {meta_file}")

        if verbose:
            print(
                "loaded metadata from"
                " {}. determined file type: {}".format(meta_file, file_type)
            )
        return metadata, file_type

    def _extract_from_metadata(self, metadata, file_type, verbose=False):
        """
        Extract the values from the metadata file. If a value other than the
        ugraph_filename is not present in the metadata file, it will be set
        to np.nan.
        The length of each list in self.results_picking is equal to the number
        of picked particles in the metadata file, determined by the length of
        the ugraph_filename list.
        """
        if file_type == "cs":
            ugraph_filename = IO.get_ugraph_cs(
                metadata
            )  # a list of all microraphs in the metadata file
            num_particles = len(ugraph_filename)
            self.results_picking["ugraph_filename"].extend(ugraph_filename)

            pos = IO.get_positions_cs(
                metadata
            )  # an array of all the x- and y-positions in the metadata file
            if pos is not None:
                self.results_picking["position_x"].extend(
                    pos[:, 0]
                )  # an array of all the x-positions in the metadata file
                self.results_picking["position_y"].extend(
                    pos[:, 1]
                )  # an array of all the y-positions in the metadata file
            else:
                self.results_picking["position_x"].extend(
                    [np.nan] * num_particles
                )
                self.results_picking["position_y"].extend(
                    [np.nan] * num_particles
                )

            ugraph_shape = IO.get_ugraph_shape_cs(
                metadata
            )  # the shape of the micrograph
            if ugraph_shape is not None:
                self.results_picking["ugraph_shape"].extend(ugraph_shape)
            else:
                self.results_picking["ugraph_shape"].extend(
                    [[np.nan, np.nan]] * num_particles
                )

            defocus = IO.get_ctf_cs(
                metadata
            )  # an array of all the defocus values in the metadata file
            if defocus is not None:
                self.results_picking["defocus"].extend(defocus[:, 0])
            else:
                self.results_picking["defocus"].extend(
                    [np.nan] * num_particles
                )

            # an array of all the class labels in the metadata file if present
            # otherwise None
            class2D = IO.get_class2D_cs(metadata)
            if class2D is not None:
                self.results_picking["class2D"].extend(class2D)
            else:
                self.results_picking["class2D"].extend(
                    [np.nan] * num_particles
                )

        elif file_type == "star":
            # a list of all microraps in the metadata file
            self.results_picking["ugraph_paths"] = IO.get_ugraph_star(metadata)
            # an array of all the defocus values in the metadata file
            self.results_picking["positions"] = IO.get_positions_star(metadata)
            # an array of all the defocus values in the metadata file
            self.results_picking["ctf"] = IO.get_ctf_star(metadata)
            # an array of all the class labels in the metadata file
            self.results_picking["class2D"] = IO.get_class2D_star(metadata)

        else:
            raise ValueError(f"unknown metadata file type: {file_type}")

        return num_particles

    def _extract_from_config(
        self, config: dict[str, Any], verbose: bool = False
    ):
        defocus = config["microscope"]["lens"]["c_10"]
        ice_thickness = config["sample"]["box"][2]
        pixel_size = config["microscope"]["detector"]["pixel_size"]
        positions_list: List = []
        filenames = []
        for molecules in config["sample"]["molecules"]["local"]:
            f = molecules["filename"]
            for instance in molecules["instances"]:
                position = instance["position"]
                positions_list.append(position)
                filenames.append(f)
        positions: np.ndarray = (
            np.array(positions_list) / pixel_size
        )  # convert to pixels

        # add to results
        print(len(filenames))
        self.results_truth["pdb_filename"] = np.append(
            self.results_truth["pdb_filename"], filenames
        )
        self.results_truth["position_x"] = np.append(
            self.results_truth["position_x"], positions[:, 0]
        )
        self.results_truth["position_y"] = np.append(
            self.results_truth["position_y"], positions[:, 1]
        )
        self.results_truth["position_z"] = np.append(
            self.results_truth["position_z"], positions[:, 2]
        )
        self.results_truth["defocus"] = np.append(
            self.results_truth["defocus"], [defocus] * len(positions)
        )
        self.results_truth["ice_thickness"] = np.append(
            self.results_truth["ice_thickness"],
            [ice_thickness] * len(positions),
        )
        num_particles = len(positions)
        return num_particles

    def _calc_dist_array(self, pos_picked, pos_truth, image_shape):
        # Calculates an array of distances between each pair of truth and
        # picked particles
        r = np.sqrt(
            np.power(float(image_shape[0]), 2)
            + np.power(float(image_shape[1]), 2)
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

    def _calc_neighbours(self, pos_picked, pos_truth, r):
        # Calculates the number of neighbours for each particle in the
        # truth set
        truth_centres = self._grab_xy_array(pos_truth)
        picked_centres = self._grab_xy_array(pos_picked)

        # get a vector of the number of neighbours, one entry per diameter
        neighbours = cKDTree(truth_centres).count_neighbors(
            cKDTree(truth_centres), r
        )
        # account for the fact each particle matches itself
        neighbours = neighbours - len(pos_truth)
        # account for fact each particle is matched 2 times! Once from
        # particle A->B and once from B->A
        neighbours = neighbours / 2

        # do the same but this time between the picked particles and the truth
        # particles
        picked_neighbours = cKDTree(picked_centres).count_neighbors(
            cKDTree(truth_centres), r
        )
        # account for fact each particle is matched 2 times!
        # Once from particle A->B and once from B->A
        picked_neighbours = picked_neighbours / 2

        return neighbours, picked_neighbours

    def _grab_xy_array(self, pos):
        # create a np array to ckdtree for number of neighbours as function
        # of distance
        particle_centres = np.empty(
            (int(len(pos)) + int(len(pos))), dtype=float
        )
        particle_centres[0::2] = pos[:, 0]
        particle_centres[1::2] = pos[:, 1]
        particle_centres = particle_centres.reshape(
            int(len(particle_centres) / 2), 2
        )
        return particle_centres

    def compute_precision(
        self,
        results_picking: pd.DataFrame,
        results_truth: pd.DataFrame,
        verbose: bool = False,
    ):
        """this function produces another data frame containing the number
        of true positives, false positives, false negatives
        and the precision, recall and multiplicity for each micrograph.
        The output then is a data frame with rows corresponding to each
        micrograph instead of each particle

        Args:
            results_picking (pandas.DataFrame): the results of the picking
            algorithm
            results_truth (pandas.DataFrame): the results of the truth
            algorithm
            verbose (bool, optional): print out information about the
            calculation. Defaults to False.

        Returns:
            pandas.DataFrame: a data frame containing the precision, recall
            and multiplicity for each micrograph
        """

        # define results data frame
        results_precision: dict[str, Any] = {
            "metadata_filename": [],
            "ugraph_filename": [],
            "defocus": [],
            "class2D": [],
            "num_particles_picked": [],
            "num_particles_truth": [],
            "TP": [],
            "FP": [],
            "FN": [],
            "precision": [],
            "recall": [],
            "multiplicity": [],
        }

        # stores for each picked particle if it is a true positive
        TP_all = []

        # stores for each picked particle the distance to the closest
        # truth particle
        closest_dist_all = []
        # stores for each picked particle the of the closest truth particle
        closest_particle_all = []
        # stores for each picked particle the pdb file of the
        # closest truth particle
        closest_pdb_all = []

        tt = time.time()
        print(
            "For each micrograph, for each metadata file, compute the"
            " precision, recall and multiplicity"
        )
        print(
            "Speed of computation depends on the number of particles in the"
            " micrograph. progressbar is not accurate"
        )

        df_picking_grouped: pd.core.groupby.generic.DataFrameGroupBy = (
            results_picking.groupby(["metadata_filename", "ugraph_filename"])
        )
        if verbose:
            print(
                "Total number of groups to loop over: {}".format(
                    len(df_picking_grouped.groups.keys())
                )
            )
            print(
                "Number of micgrographs: {}".format(
                    len(results_picking["ugraph_filename"].unique())
                )
            )
            print(
                "Number of metadata files: {}".format(
                    len(results_picking["metadata_filename"].unique())
                )
            )
            print("Starting loop over groups")
        df_truth_grouped: pd.core.groupby.generic.DataFrameGroupBy = (
            results_truth.groupby("ugraph_filename")
        )
        progressbar = tqdm(
            total=len(df_picking_grouped.groups.keys()),
            desc="computing precision",
            disable=not verbose,
        )
        groupnames: dict_keys = df_picking_grouped.groups.keys()
        for groupname in groupnames:
            if groupname[1] not in df_truth_grouped.groups.keys():
                print(f"WARNING: {groupname[1]} not found in truth data frame")
                continue
            picked_particles_in_ugraph = df_picking_grouped.get_group(
                groupname
            )
            ugraph_shape = picked_particles_in_ugraph["ugraph_shape"].values[0]
            pos_picked_in_ugraph = picked_particles_in_ugraph[
                ["position_x", "position_y"]
            ].values
            truth_particles_in_ugraph = df_truth_grouped.get_group(
                groupname[1]
            )
            pos_truth_in_ugraph = truth_particles_in_ugraph[
                ["position_x", "position_y"]
            ].values

            # Calculate the distance matrix between the picked
            # and truth particles
            sdm = self._calc_dist_array(
                pos_picked_in_ugraph, pos_truth_in_ugraph, ugraph_shape
            )

            # A picked particle is considered a true positive if it is closer
            # than 1 particle radius to any of the true particles
            # A picked particle is considered a false positive if it is not
            # closer than 1 particle radius to any of the true particles
            TP = 0  # the number of true positives in the current micrograph
            FP = 0  # the number of false positives in the current micrograph
            for particle in range(len(pos_picked_in_ugraph)):
                TP += np.any(sdm[particle] < self.particle_diameter / 2)
                FP += np.all(sdm[particle] > self.particle_diameter / 2)

                TP_all.append(
                    np.any(sdm[particle] < self.particle_diameter / 2)
                )  # append the TP and FP to the picked particle data frame
                closest_dist_all.append(np.min(sdm[particle]))
                closest_particle_all.append(np.argmin(sdm[particle]))
                closest_pdb_all.append(
                    truth_particles_in_ugraph["pdb_filename"].values[
                        np.argmin(sdm[particle])
                    ]
                )

            # A truth particle is considered a false negative if it is
            # not closer than 1 particle radius to any of the picked particles
            # The multiplicity is defined as the average number of times
            # a truth particle is picked
            FN = 0  # the number of false negatives in the current micrograph
            multiplicity = []  # the number of times a truth particle is picked
            for particle in range(len(pos_truth_in_ugraph)):
                FN += np.all(sdm[:, particle] > self.particle_diameter / 2)
                multiplicity.append(
                    np.sum(sdm[:, particle] < self.particle_diameter / 2)
                )
            multiplicity = np.mean(multiplicity)

            # calculate the precision, recall and multiplicity
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)

            # append the results to the results data frame
            results_precision["metadata_filename"].append(groupname[0])
            results_precision["ugraph_filename"].append(groupname[1])
            results_precision["defocus"].append(
                picked_particles_in_ugraph["defocus"].values[0]
            )
            results_precision["class2D"].append(
                picked_particles_in_ugraph["class2D"].values[0]
            )
            results_precision["num_particles_picked"].append(
                len(pos_picked_in_ugraph)
            )
            results_precision["num_particles_truth"].append(
                len(pos_truth_in_ugraph)
            )
            results_precision["TP"].append(TP)
            results_precision["FP"].append(FP)
            results_precision["FN"].append(FN)
            results_precision["precision"].append(precision)
            results_precision["recall"].append(recall)
            results_precision["multiplicity"].append(multiplicity)

            progressbar.update(1)
            progressbar.set_postfix(
                {
                    "precision": precision,
                    "recall": recall,
                    "multiplicity": multiplicity,
                }
            )
        progressbar.close()

        if verbose:
            print(
                "time taken to compute precision: {}".format(time.time() - tt)
            )

        # add the new values to the picked particle data frame
        results_picking["TP"] = TP_all
        results_picking["closest_dist"] = closest_dist_all
        results_picking["closest_particle"] = closest_particle_all
        results_picking["closest_pdb"] = closest_pdb_all
        results_picking["closest_pdb_index"] = results_picking[
            "closest_pdb"
        ].apply(lambda x: int(x.split("_")[-1].split(".")[0]))
        # set the closest_pdb_index to np.nan if the particle is not
        # closer to a truth particle thatn the particle diameter
        results_picking.loc[
            results_picking["closest_dist"] > self.particle_diameter,
            "closest_pdb_index",
        ] = np.nan

        # convert the results data frame to a pandas data frame
        return pd.DataFrame(results_precision), results_picking

    def compute_overlap(
        self,
        results_picking: pd.DataFrame,
        results_truth: pd.DataFrame,
        verbose: bool = False,
    ):
        """this function computes the number of picked particles that overlap
        with a truth particle at a range of radii.
        The output is a data frame with rows corresponding to each radius

        Args:
            results_picking (pandas.DataFrame): the results of the picking
            algorithm
            results_truth (pandas.DataFrame): the results of the truth
            algorithm
            verbose (bool, optional): print out information about the
            calculation. Defaults to False.

        Returns:
            pandas.DataFrame: a data frame containing the number of picked
            particles that overlap with a truth particle at a range of radii
        """

        results_overlap: dict[str, List[Any]] = {
            "metadata_filename": [],
            "ugraph_filename": [],
            "defocus": [],
            "radius": [],
            "neighbours_truth": [],
            "neighbours_picked": [],
        }

        r = np.array(np.arange(10.0, 401.0, 10.0)) / 2

        results_picking_grouped: pd.core.groupby.generic.DataFrameGroupBy = (
            results_picking.groupby(["metadata_filename", "ugraph_filename"])
        )
        results_truth_grouped = results_truth.groupby("ugraph_filename")

        progressbar = tqdm(
            total=len(results_picking_grouped.groups.keys()),
            desc="computing overlap",
            disable=not verbose,
        )
        keys: dict_keys = results_picking_grouped.groups.keys()
        for key in keys:
            picked_pos_x = results_picking_grouped.get_group(key)["position_x"]
            picked_pos_y = results_picking_grouped.get_group(key)["position_y"]
            truth_pos_x = results_truth_grouped.get_group(key[1])["position_x"]
            truth_pos_y = results_truth_grouped.get_group(key[1])["position_y"]
            defocus = results_truth_grouped.get_group(key[1])["defocus"].mean()

            neighbours, picked_neighbours = self._calc_neighbours(
                np.array([picked_pos_x, picked_pos_y]).T,
                np.array([truth_pos_x, truth_pos_y]).T,
                r,
            )

            for i in range(len(r)):
                results_overlap["metadata_filename"].append(key[0])
                results_overlap["ugraph_filename"].append(key[1])
                results_overlap["defocus"].append(defocus)
                results_overlap["radius"].append(r[i])
                results_overlap["neighbours_truth"].append(neighbours[i])
                results_overlap["neighbours_picked"].append(
                    picked_neighbours[i]
                )

            progressbar.update(1)
            progressbar.set_postfix(
                {
                    "neighbours_truth": neighbours[i],
                    "neighbours_picked": picked_neighbours[i],
                }
            )
        progressbar.close()

        return pd.DataFrame(data=results_overlap)
