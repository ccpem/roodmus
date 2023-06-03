import os
import time
from collections.abc import KeysView as dict_keys
from typing import Tuple, Any, List

import yaml
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm
import pandas as pd

from pipeliner.jobstar_reader import RelionStarFile


class IO(object):
    """Class containing several functions to load metadata from .star (RELION)
    and .cs (CryoSPARC) files and the config file generated during the
    Parakeet simulation
    """

    # Loading .cs files and parsing the ctf parameters,
    # the particle positions and orientations
    @classmethod
    def load_cs(self, cs_path):
        metadata = np.load(cs_path)
        return metadata

    @classmethod
    def get_ugraph_cs(self, metadata_cs: np.recarray, config_dir: str):
        if "location/micrograph_path" in metadata_cs.dtype.names:
            ugraph_paths = metadata_cs["location/micrograph_path"]
        elif "blob/path" in metadata_cs.dtype.names:
            ugraph_paths = metadata_cs["blob/path"]

        # make a mask to retain only the lines we have data files for
        mask = np.array(
            [
                1
                if os.path.exists(
                    os.path.join(
                        config_dir, path.decode("utf-8").split("_")[-1]
                    )
                )
                else 0
                for path in ugraph_paths
            ],
            dtype=bool,
        )

        ugraph_paths = ugraph_paths[mask]
        ugraph_paths = [
            os.path.basename(path).decode("utf-8").split("_")[-1]
            for path in ugraph_paths
        ]
        return ugraph_paths, mask

    @classmethod
    def get_uid_cs(self, metadata_cs: np.recarray):
        if "uid" in metadata_cs.dtype.names:
            uid = metadata_cs["uid"]
        else:
            uid = None
        return uid

    @classmethod
    def get_ctf_cs(
        self,
        metadata_cs: np.recarray,
        mask: np.ndarray,
    ):
        if "ctf/df1_A" in metadata_cs.dtype.names:
            defocusU = metadata_cs["ctf/df1_A"][mask]
            defocusV = metadata_cs["ctf/df2_A"][mask]
            kV = metadata_cs["ctf/accel_kv"][mask]
            Cs = metadata_cs["ctf/cs_mm"][mask]
            amp = metadata_cs["ctf/amp_contrast"][mask]
            Bfac = metadata_cs["ctf/bfactor"][mask]
            return np.stack([defocusU, defocusV, kV, Cs, amp, Bfac], axis=1)
        else:
            return None

    @classmethod
    def get_positions_cs(self, metadata_cs: np.recarray):
        if "location/center_x_frac" in metadata_cs.dtype.names:
            ugraph_shape = metadata_cs["location/micrograph_shape"]
            # print(type(ugraph_shape), len(ugraph_shape), ugraph_shape.shape)
            x = metadata_cs["location/center_x_frac"]
            y = metadata_cs["location/center_y_frac"]
            # convert to absolute coordinates
            x_abs = (
                x * ugraph_shape[0, 0]
            )  # assuming all micrographs have the same shape
            y_abs = y * ugraph_shape[0, 1]
            # conver to single array
            pos = np.stack([x_abs, y_abs], axis=1)
            return pos
        else:
            return None

    @classmethod
    def get_orientations_cs(self, metadata_cs: np.recarray):
        if "alignments3D/pose" in metadata_cs.dtype.names:
            pose = metadata_cs[
                "alignments3D/pose"
            ]  # orientations as rodriques vectors
            # convert to euler angles
            euler = [geom.rot2euler(geom.expmap(p)) for p in pose]
            euler = np.array(euler)
            # euler = R.from_rotvec(pose).as_euler(
            #     "zyx", degrees=False
            # )  # convert to euler angles
            return euler
        else:
            return None

    @classmethod
    def get_ugraph_shape_cs(self, metadata_cs: np.recarray):
        if "location/micrograph_shape" in metadata_cs.dtype.names:
            ugraph_shape = metadata_cs["location/micrograph_shape"]
        else:
            ugraph_shape = None
        return ugraph_shape

    @classmethod
    def get_class2D_cs(self, metadata_cs: np.recarray):
        if "alignments2D/class" in metadata_cs.dtype.names:
            class2d = metadata_cs["alignments2D/class"]
        else:
            class2d = None
        return class2d

    # Loading .star files and parsing the ctf parameters,
    # the particle positions and orientations
    @classmethod
    def load_star(self, star_path):
        return RelionStarFile(star_path)

    @classmethod
    def get_ugraph_star(self, metadata_star, config_dir: str):
        ugraph_paths = metadata_star.column_as_list(
            "particles", "_rlnMicrographName"
        )
        mask = np.array(
            [
                1 if os.path.exists(os.path.join(config_dir, path)) else 0
                for path in ugraph_paths
            ],
            dtype=bool,
        )
        ugraph_paths = ugraph_paths[mask]
        # convert to basename and remove index
        ugraph_paths = [
            os.path.basename(path).split("_")[-1] for path in ugraph_paths
        ]
        return ugraph_paths, mask

    @classmethod
    def get_ctf_star(
        self,
        metadata_star,
        mask: np.ndarray,
    ) -> np.ndarray:
        kV = [
            float(r)
            for r in metadata_star.column_as_list("optics", "_rlnVoltage")
        ]
        Cs = [
            float(r)
            for r in metadata_star.column_as_list(
                "optics", "_rlnSphericalAberration"
            )
        ]
        amp = [
            float(r)
            for r in metadata_star.column_as_list(
                "optics", "_rlnAmplitudeContrast"
            )
        ]

        defocusU = [
            float(r)
            for r in metadata_star.column_as_list("particles", "_rlnDefocusU")
        ][mask]
        defocusV = [
            float(r)
            for r in metadata_star.column_as_list("particles", "_rlnDefocusV")
        ][mask]
        Bfac = [0]  # not available in RELION star files
        return np.stack(
            [
                defocusU,
                defocusV,
                kV * len(defocusU),
                Cs * len(defocusU),
                amp * len(defocusU),
                Bfac * len(defocusU),
            ],
            axis=1,
        )

    @classmethod
    def get_positions_star(self, metadata_star) -> np.ndarray:
        x = [
            float(r)
            for r in metadata_star.column_as_list(
                "particles", "_rlnCoordinateX"
            )
        ]
        y = [
            float(r)
            for r in metadata_star.column_as_list(
                "particles", "_rlnCoordinateY"
            )
        ]
        pos = np.stack([x, y], axis=1)
        return pos

    @classmethod
    def get_orientations_star(self, metadata_star) -> np.ndarray:
        euler_phi = metadata_star.column_as_list("particles", "_rlnAngleRot")
        euler_theta = metadata_star.column_as_list(
            "particles", "_rlnAngleTilt"
        )
        euler_psi = metadata_star.column_as_list("particles", "_rlnAnglePsi")

        num_particles = np.max(
            [len(euler_phi), len(euler_theta), len(euler_psi)]
        )
        if not euler_phi:  # if empty
            euler_phi = [np.nan] * num_particles
        if not euler_theta:
            euler_theta = [np.nan] * num_particles
        if not euler_psi:
            euler_psi = [np.nan] * num_particles
        euler = np.stack([euler_phi, euler_theta, euler_psi], axis=1)
        return euler

    @classmethod
    def get_class2D_star(self, metadata_star):
        class2d = metadata_star.column_as_list("particles", "_rlnClassNumber")
        if class2d:
            return np.array(class2d)
        else:
            return None

    # loading the config file
    @classmethod
    def load_config(self, config_path):
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        return config


class geom(object):
    # adapted from the pyem package by Daniel Asarnow.
    # Under the GNU General Public License v3.0
    @classmethod
    def expmap(self, e):
        """Convert axis-angle vector into 3D rotation matrix"""
        theta = np.linalg.norm(e)
        if theta < 1e-16:
            return np.identity(3, dtype=e.dtype)
        w = e / theta
        k = np.array(
            [[0, w[2], -w[1]], [-w[2], 0, w[0]], [w[1], -w[0], 0]],
            dtype=e.dtype,
        )
        r = (
            np.identity(3, dtype=e.dtype)
            + np.sin(theta) * k
            + (1 - np.cos(theta)) * np.dot(k, k)
        )
        return r

    @classmethod
    def rot2euler(self, r):
        """Decompose rotation matrix into Euler angles"""
        # assert(isrotation(r))
        # Shoemake rotation matrix decomposition
        # algorithm with same conventions as Relion.
        epsilon = np.finfo(np.double).eps
        abs_sb = np.sqrt(r[0, 2] ** 2 + r[1, 2] ** 2)
        if abs_sb > 16 * epsilon:
            gamma = np.arctan2(r[1, 2], -r[0, 2])
            alpha = np.arctan2(r[2, 1], r[2, 0])
            if np.abs(np.sin(gamma)) < epsilon:
                sign_sb = np.sign(-r[0, 2]) / np.cos(gamma)
            else:
                sign_sb = (
                    np.sign(r[1, 2])
                    if np.sin(gamma) > 0
                    else -np.sign(r[1, 2])
                )
            beta = np.arctan2(sign_sb * abs_sb, r[2, 2])
        else:
            if np.sign(r[2, 2]) > 0:
                alpha = 0
                beta = 0
                gamma = np.arctan2(-r[1, 0], r[0, 0])
            else:
                alpha = 0
                beta = np.pi
                gamma = np.arctan2(r[1, 0], -r[0, 0])
        return alpha, beta, gamma


class load_data(object):
    def __init__(
        self,
        meta_file: str | List[str],
        config_dir: str,
        particle_diameter: float,
        ugraph_shape: Tuple[int, int] = (4000, 4000),
        results_picking: dict | None = None,
        results_truth: dict | None = None,
        verbose: bool = True,
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
            "euler_phi": [],
            "euler_theta": [],
            "euler_psi": [],
            "ugraph_shape": [],
            "defocusU": [],
            "defocusV": [],
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
            "euler_phi": [],
            "euler_theta": [],
            "euler_psi": [],
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
        self.add_data()

    def add_data(
        self,
        meta_file: str | List[str] = "",
        config_dir: str = "",
        verbose: bool = True,
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
            if isinstance(self.meta_file, str):
                self.results_picking["metadata_filename"].extend(
                    [self.meta_file] * num_particles
                )  # add the metadata file to the picking results
            elif isinstance(self.meta_file, list):
                self.results_picking["metadata_filename"].extend(
                    [self.meta_file[0]] * num_particles
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
                total=len(ugraphs_to_load),
                desc="loading micrographs",
                disable=not self.verbose,
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
        self, meta_file: str | List[str] = "", verbose: bool = False
    ) -> Tuple[dict, str]:
        if isinstance(meta_file, str):
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
                    " {}. determined file type: {}".format(
                        meta_file, file_type
                    )
                )
        elif isinstance(meta_file, list):
            metadata = []
            file_types = []
            for file in meta_file:
                if file.endswith(".star"):
                    metadata.append(IO.load_star(file))
                    file_type = "star"
                    file_types.append(file_type)
                elif file.endswith(".cs"):
                    metadata.append(IO.load_cs(file))
                    file_type = "cs"
                    file_types.append(file_type)
                else:
                    raise ValueError(f"unknown metadata file type: {file}")
                if verbose:
                    print(
                        "loaded metadata from"
                        " {}. determined file type: {}".format(file, file_type)
                    )
            # check if the file types are the same
            if len(set(file_types)) > 1:
                raise ValueError(
                    "multiple metadata files were given to combine, \
                        but they are not all the same type"
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
        if isinstance(metadata, list) and file_type == "cs":
            print(f"loading {len(metadata)} files into the results")
            tmp_results = {"uid": [], "mask": []}
            for key in self.results_picking.keys():
                tmp_results[key] = []

            num_particles_loaded = 0
            for m in metadata:
                ugraph_filename = IO.get_ugraph_cs(m)
                num_particles = len(
                    ugraph_filename
                )  # total number of particles in the metadata file
                tmp_results["ugraph_filename"].extend(ugraph_filename)
                mask = self._check_if_ugraphs_exist(ugraph_filename)
                num_particles_loaded += np.sum(mask)
                tmp_results["mask"].extend(mask)

                uid = IO.get_uid_cs(m)
                if uid is not None:
                    tmp_results["uid"].extend(uid)
                else:
                    tmp_results["uid"].extend([np.nan] * num_particles)

                pos = IO.get_positions_cs(m)  # an array of all the x-
                # and y-positions in the metadata file
                if pos is not None:
                    tmp_results["position_x"].extend(
                        pos[:, 0]
                    )  # an array of all the x-positions in the metadata file
                    tmp_results["position_y"].extend(
                        pos[:, 1]
                    )  # an array of all the y-positions in the metadata file
                else:
                    tmp_results["position_x"].extend([np.nan] * num_particles)
                    tmp_results["position_y"].extend([np.nan] * num_particles)

                orientation = IO.get_orientations_cs(
                    m
                )  # an array of all the orientations in the metadata file
                if orientation is not None:
                    tmp_results["euler_phi"].extend(orientation[:, 0])
                    tmp_results["euler_theta"].extend(orientation[:, 1])
                    tmp_results["euler_psi"].extend(orientation[:, 2])
                else:
                    tmp_results["euler_phi"].extend([np.nan] * num_particles)
                    tmp_results["euler_theta"].extend([np.nan] * num_particles)
                    tmp_results["euler_psi"].extend([np.nan] * num_particles)

                ugraph_shape = IO.get_ugraph_shape_cs(
                    m
                )  # the shape of the micrograph
                if ugraph_shape is not None:
                    tmp_results["ugraph_shape"].extend(ugraph_shape)
                else:
                    tmp_results["ugraph_shape"].extend(
                        [self.ugraph_shape] * num_particles
                    )

                defocus = IO.get_ctf_cs(
                    m
                )  # an array of all the defocus values in the metadata file
                if defocus is not None:
                    tmp_results["defocusU"].extend(defocus[:, 0])
                    tmp_results["defocusV"].extend(defocus[:, 1])
                else:
                    tmp_results["defocusU"].extend([np.nan] * num_particles)
                    tmp_results["defocusV"].extend([np.nan] * num_particles)

                # an array of all the class labels in the metadata file
                # if present, otherwise None
                class2D = IO.get_class2D_cs(m)
                if class2D is not None:
                    tmp_results["class2D"].extend(class2D)
                else:
                    tmp_results["class2D"].extend([np.nan] * num_particles)

            if "metadata_filename" in tmp_results.keys():
                tmp_results.pop("metadata_filename")
            df_tmp = pd.DataFrame(tmp_results, columns=tmp_results.keys())
            mask_true = df_tmp["mask"]
            df_tmp = df_tmp.groupby("uid").agg(
                {
                    "ugraph_filename": lambda x: x[mask_true].iloc[0],
                    "mask": "sum",
                    "position_x": "first",
                    "position_y": "first",
                    "euler_phi": "first",
                    "euler_theta": "first",
                    "euler_psi": "first",
                    "ugraph_shape": "first",
                    "defocusU": "first",
                    "defocusV": "first",
                    "class2D": "first",
                }
            )

            # add the temporary results to the results
            self.results_picking["position_x"].extend(
                df_tmp["position_x"].values
            )
            self.results_picking["position_y"].extend(
                df_tmp["position_y"].values
            )
            self.results_picking["euler_phi"].extend(
                df_tmp["euler_phi"].values
            )
            self.results_picking["euler_theta"].extend(
                df_tmp["euler_theta"].values
            )
            self.results_picking["euler_psi"].extend(
                df_tmp["euler_psi"].values
            )
            self.results_picking["ugraph_shape"].extend(
                df_tmp["ugraph_shape"].values
            )
            self.results_picking["defocusU"].extend(df_tmp["defocusU"].values)
            self.results_picking["defocusV"].extend(df_tmp["defocusV"].values)
            self.results_picking["class2D"].extend(df_tmp["class2D"].values)
            self.results_picking["ugraph_filename"].extend(
                df_tmp["ugraph_filename"].values
            )

            # remove the temporary results and update the number of particles
            tmp_results = {}
            num_particles = num_particles_loaded
            print(f"found {num_particles} particles")

        elif isinstance(metadata, list) and file_type == "star":
            pass  # should not happen, but just in case we add it here

        elif file_type == "cs":
            ugraph_filename, mask = IO.get_ugraph_cs(
                metadata,
                self.config_dir,
            )  # a list of all microraphs in the metadata file
            print("checking if ugraphs exist...")
            num_particles = len(ugraph_filename)
            self.results_picking["ugraph_filename"].extend(ugraph_filename)

            pos = IO.get_positions_cs(
                metadata
            )  # an array of all the x- and y-positions in the metadata file
            if pos is not None:
                self.results_picking["position_x"].extend(
                    pos[:, 0][mask]
                )  # an array of all the x-positions in the metadata file
                self.results_picking["position_y"].extend(
                    pos[:, 1][mask]
                )  # an array of all the y-positions in the metadata file
            else:
                self.results_picking["position_x"].extend(
                    [np.nan] * num_particles
                )
                self.results_picking["position_y"].extend(
                    [np.nan] * num_particles
                )
            orientation = IO.get_orientations_cs(
                metadata
            )  # an array of all the orientations in the metadata file
            if orientation is not None:
                orientation = orientation[mask]
                self.results_picking["euler_phi"].extend(orientation[:, 0])
                self.results_picking["euler_theta"].extend(orientation[:, 1])
                self.results_picking["euler_psi"].extend(orientation[:, 2])
            else:
                self.results_picking["euler_phi"].extend(
                    [np.nan] * num_particles
                )
                self.results_picking["euler_theta"].extend(
                    [np.nan] * num_particles
                )
                self.results_picking["euler_psi"].extend(
                    [np.nan] * num_particles
                )

            ugraph_shape = IO.get_ugraph_shape_cs(
                metadata
            )  # the shape of the micrograph
            if ugraph_shape is not None:
                ugraph_shape = ugraph_shape[mask]
                self.results_picking["ugraph_shape"].extend(ugraph_shape)
            else:
                self.results_picking["ugraph_shape"].extend(
                    [self.ugraph_shape] * num_particles
                )

            defocus = IO.get_ctf_cs(
                metadata,
                mask,
            )  # an array of all the defocus values in the metadata file
            if defocus is not None:
                # defocus = defocus[mask]
                self.results_picking["defocusU"].extend(defocus[:, 0])
                self.results_picking["defocusV"].extend(defocus[:, 1])
            else:
                self.results_picking["defocusU"].extend(
                    [np.nan] * num_particles
                )
                self.results_picking["defocusV"].extend(
                    [np.nan] * num_particles
                )

            # an array of all the class labels in the metadata file if present
            # otherwise None
            class2D = IO.get_class2D_cs(metadata)
            if class2D is not None:
                self.results_picking["class2D"].extend(class2D[mask])
            else:
                self.results_picking["class2D"].extend(
                    [np.nan] * num_particles
                )

        elif file_type == "star":
            ugraph_filename = IO.get_ugraph_star(
                metadata
            )  # a list of all microraphs in the metadata file
            print("checking if ugraphs exist...")
            mask = self._check_if_ugraphs_exist(ugraph_filename)
            ugraph_filename = np.array(ugraph_filename)[mask]
            num_particles = len(ugraph_filename)
            self.results_picking["ugraph_filename"].extend(ugraph_filename)

            pos = IO.get_positions_star(
                metadata
            )  # an array of all the x- and y-positions in the metadata file
            if pos is not None:
                self.results_picking["position_x"].extend(
                    pos[:, 0][mask]
                )  # an array of all the x-positions in the metadata file
                self.results_picking["position_y"].extend(
                    pos[:, 1][mask]
                )  # an array of all the y-positions in the metadata file
            else:
                self.results_picking["position_x"].extend(
                    [np.nan] * num_particles
                )
                self.results_picking["position_y"].extend(
                    [np.nan] * num_particles
                )
            orientation = IO.get_orientations_star(
                metadata
            )  # an array of all the orientations in the metadata file
            if orientation is not None:
                orientation = orientation[mask]
                self.results_picking["euler_phi"].extend(orientation[:, 0])
                self.results_picking["euler_theta"].extend(orientation[:, 1])
                self.results_picking["euler_psi"].extend(orientation[:, 2])
            else:
                self.results_picking["euler_phi"].extend(
                    [np.nan] * num_particles
                )
                self.results_picking["euler_theta"].extend(
                    [np.nan] * num_particles
                )
                self.results_picking["euler_psi"].extend(
                    [np.nan] * num_particles
                )

            ugraph_shape = None  # not stored in .star files
            if ugraph_shape is not None:
                ugraph_shape = ugraph_shape[mask]
                self.results_picking["ugraph_shape"].extend(ugraph_shape)
            else:
                self.results_picking["ugraph_shape"].extend(
                    [self.ugraph_shape] * num_particles
                )

            defocus = IO.get_ctf_star(
                metadata
            )  # an array of all the defocus values in the metadata file
            if defocus is not None:
                defocus = defocus[mask]
                self.results_picking["defocusU"].extend(defocus[:, 0])
                self.results_picking["defocusV"].extend(defocus[:, 1])
            else:
                self.results_picking["defocusU"].extend(
                    [np.nan] * num_particles
                )
                self.results_picking["defocusV"].extend(
                    [np.nan] * num_particles
                )

            # an array of all the class labels in the metadata file if present
            # otherwise None
            class2D = IO.get_class2D_star(metadata)
            if class2D is not None:
                self.results_picking["class2D"].extend(class2D[mask])
            else:
                self.results_picking["class2D"].extend(
                    [np.nan] * num_particles
                )
        else:
            raise ValueError(f"unknown metadata file type: {file_type}")

        return num_particles

    def _check_if_ugraphs_exist(
        self, ugraph_filename: List[Any]
    ) -> np.ndarray:
        """
        Check if the micrograph exists in the config directory.
        If it does, return True, otherwise False.
        """
        mask = np.array(
            [
                os.path.exists(
                    os.path.join(
                        self.config_dir, ugraph.replace(".mrc", ".yaml")
                    )
                )
                for ugraph in ugraph_filename
            ]
        )
        return mask

    def _extract_from_config(
        self, config: dict[str, Any], verbose: bool = False
    ):
        defocus = config["microscope"]["lens"]["c_10"]
        ice_thickness = config["sample"]["box"][2]
        pixel_size = config["microscope"]["detector"]["pixel_size"]
        positions_list: List = []
        orientations_list: List = []
        filenames = []
        for molecules in config["sample"]["molecules"]["local"]:
            f = molecules["filename"]
            for instance in molecules["instances"]:
                position = instance["position"]
                orientation = instance["orientation"]  # rotation vector
                # convert to euler angles
                euler = geom.rot2euler(geom.expmap(np.array(orientation)))
                # euler = R.from_rotvec(orientation).as_euler("zyx")
                orientations_list.append(euler)
                positions_list.append(position)
                filenames.append(f)
        positions: np.ndarray = (
            np.array(positions_list) / pixel_size
        )  # convert to pixels
        orientations: np.ndarray = np.array(orientations_list)

        # add to results
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
        self.results_truth["euler_phi"] = np.append(
            self.results_truth["euler_phi"],
            orientations[:, 0],
        )
        self.results_truth["euler_theta"] = np.append(
            self.results_truth["euler_theta"],
            orientations[:, 1],
        )
        self.results_truth["euler_psi"] = np.append(
            self.results_truth["euler_psi"],
            orientations[:, 2],
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
                truth_particles_in_ugraph["defocus"].values[0]
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
