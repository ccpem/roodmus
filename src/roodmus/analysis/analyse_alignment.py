"""
    API containing functions for the analysis of the 3D alignment results.
"""

import os

import numpy as np
from scipy.spatial.transform.rotation import Rotation as R
from tqdm import tqdm

from utils import IO


class alignment_3D(object):
    def __init__(
        self,
        meta_file: str,
        config_dir: str,
        results_picking: dict | None = None,
        results_truth: dict | None = None,
        verbose: bool = False,
    ):
        self.meta_file = meta_file
        self.config_dir = config_dir
        self.verbose = verbose
        if results_picking is not None:
            self.results_picking = results_picking
        else:
            self.results_picking = {
                "metadata_filename": [],
                "ugraph_filename": [],
                "defocus": [],
                "euler1": [],
                "euler2": [],
                "euler3": [],
                "particle_shape": [],
            }
        if results_truth is not None:
            self.results_truth = results_truth
        else:
            self.results_truth = {
                "metadata_filename": [],
                "ugraph_filename": [],
                "pdb_filename": [],
                "defocus": [],
                "euler1": [],
                "euler2": [],
                "euler3": [],
            }

        # Misc attributes
        # Stores the paths to the micrographs from the metadata file
        self.ugraph_paths: list[str] = []
        # Stores the number of micrographs that have been loaded
        self.num_ugraphs_loaded = 0
        # Stores the number of particles per micrograph
        self.particles_per_ugraph: list[int] = []

        self.compute()

    def compute(
        self,
        meta_file: str | None = None,
        config_dir: str | None = None,
        verbose: bool | None = None,
    ):
        """Load the 3D estimated poses from the metadata file and the true
         poses from the config files. Then compute the alignment error and
          store the results in a dict.

        Args:
            meta_file (str, optional): The metadata file.
             Defaults to None.
            config_dir (str, optional): The config directory.
             Defaults to None.
            verbose (bool, optional): Whether to print the results.
             Defaults to None.

        Raises:
            ValueError: If the metadata file type is unknown.

        Returns:
            dict: The results. Can be turned into a pandas DataFrame.
        """

        # if the user has provided new arguments, use them
        update_meta_file = False
        update_config_dir = False
        if meta_file is not None and meta_file is not self.meta_file:
            # user has specified a new metadata file to use
            self.meta_file = meta_file
            update_meta_file = True
        elif meta_file is None:
            # user has not specified a new metadata file to use
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
                print(f"loading metadata from {self.meta_file}...")
            metadata, file_type = self._load_metadata(
                self.meta_file,
                self.verbose,
            )
            (
                ugraph_paths,
                orientations,
                ctf_params,
                ugraph_shapes,
            ) = self._extract_from_metadata(metadata, file_type, self.verbose)
            if ctf_params is None:
                ctf_params = np.ones((len(orientations), 1)) * np.nan
            self.ugraph_paths = list(
                np.unique(np.concatenate((self.ugraph_paths, ugraph_paths)))
            )
            if self.verbose:
                print(
                    "Found {} particles in {} micrographs".format(
                        len(orientations),
                        len(np.unique(self.ugraph_paths)),
                    )
                )

            for ugraph_path, orientation, ctf_param, ugraph_shape in zip(
                ugraph_paths, orientations, ctf_params, ugraph_shapes
            ):
                self.results_picking["metadata_filename"].append(
                    self.meta_file
                )
                self.results_picking["ugraph_filename"].append(ugraph_path)
                self.results_picking["euler1"].append(orientation[0])
                self.results_picking["euler2"].append(orientation[1])
                self.results_picking["euler3"].append(orientation[2])
                self.results_picking["particle_shape"].append(ugraph_shape)
                self.results_picking["defocus"].append(ctf_param[0])

            if self.verbose:
                print(
                    "Loaded {} micrographs from {}".format(
                        len(self.ugraph_paths),
                        self.meta_file,
                    )
                )
                print(
                    "Dictionaries now contain {}"
                    " particles and {} true particles".format(
                        len(self.results_picking["ugraph_filename"]),
                        len(self.results_truth["ugraph_filename"]),
                    )
                )

        # Load the config files.
        # Either the config dir has changed,
        # or the number of micrographs has increased
        if (
            update_config_dir
            or len(self.ugraph_paths) > self.num_ugraphs_loaded
        ):
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
                    orientations,
                    defocus_values,
                    ice_thickness_values,
                ) = self._extract_from_config(config, self.verbose)
                num_particles[ugraph_path] = len(filenames)
                for (
                    filename,
                    orientation,
                    defocus_value,
                    ice_thickness_value,
                ) in zip(
                    filenames,
                    orientations,
                    defocus_values,
                    ice_thickness_values,
                ):
                    self.results_truth["metadata_filename"].append(
                        self.meta_file
                    )
                    self.results_truth["ugraph_filename"].append(ugraph_path)
                    self.results_truth["pdb_filename"].append(filename)
                    self.results_truth["euler1"].append(orientation[0])
                    self.results_truth["euler2"].append(orientation[1])
                    self.results_truth["euler3"].append(orientation[2])
                    self.results_truth["defocus"].append(defocus_value)

                _ = progressbar.update(1)
            progressbar.close()

            # update the number of micrographs loaded
            # and the number of particles per micrograph
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
                    "Dictionaries now contain {}"
                    " particles and {} true particles".format(
                        len(self.results_picking["ugraph_filename"]),
                        len(self.results_truth["ugraph_filename"]),
                    )
                )

    def _load_metadata(
        self, meta_file: str, verbose: bool = False
    ) -> tuple[dict, str]:
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
                "Loaded metadata from {}."
                " Determined file type: {}.".format(meta_file, file_type)
            )
        return metadata, file_type

    def _extract_from_metadata(self, metadata, file_type, verbose=False):
        if file_type == "cs":
            ugraph_paths = IO.get_ugraph_cs(
                metadata
            )  # a list of all microraps in the metadata file
            orientation = IO.get_orientations_cs(
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
        else:
            raise ValueError(f"unknown metadata file type: {file_type}")

        if verbose:
            print("Extracted ugraph paths and positions from metadata file")
            print(orientation.shape)
            print(len(ugraph_paths))
        return ugraph_paths, orientation, ctf, ugraph_shape

    def _extract_from_config(self, config: dict, verbose: bool = False):
        defocus = config["microscope"]["lens"]["c_10"]
        ice_thickness = config["sample"]["box"][2]
        pixel_size = config["microscope"]["detector"]["pixel_size"]
        orientations = []
        filenames = []
        for molecules in config["sample"]["molecules"]["local"]:
            f = molecules["filename"]
            for instance in molecules["instances"]:
                # position = instance["position"]
                orientation = instance["orientation"]  # as a rotation vector
                eulers = R.from_rotvec(orientation).as_euler(
                    "zyx", degrees=False
                )
                orientations.append(eulers)
                filenames.append(f)

        orientations = np.array(orientations) / pixel_size  # convert to pixels
        defocus_list = [defocus] * len(orientations)
        ice_thickness_list = [ice_thickness] * len(orientations)
        return filenames, orientations, defocus_list, ice_thickness_list
