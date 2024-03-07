"""Sampling a molecular dynamics trajectory and saving the
conformations to PDB files.

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

import os
from typing import Tuple, List, Any
import argparse

import numpy as np
from matplotlib import pyplot as plt
import mdtraj as mdt
import glob2 as glob
from tqdm import tqdm


def add_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Set up arguments for parsing

    Args:
        parser (argparse.ArgumentParser): Waymarking argument parser

    Returns:
        argparse.ArgumentParser:  Waymarking argument parser with arguments
    """

    parser.add_argument(
        "--trajfiles_dir",
        help=(
            "Path to directory holding (dcd) trajectory files which make up"
            " the whole trajectory."
        ),
        type=str,
        default="",
    )

    parser.add_argument(
        "--topfile",
        help="The pdb holding the structure of molecule (no solvent)",
        type=str,
        default="",
    )

    parser.add_argument(
        "--verbose",
        help="Increase output verbosity",
        action="store_true",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--tqdm",
        help="Turn on progress bar",
        action="store_true",
    )

    # parser.add_argument(
    #     "--sampling_method",
    #     help=(
    #         "Choose whether to sample a trajectory uniformly in time"
    #         " (even_sampling) or by >rmsd threshold (waymark)"
    #     ),
    #     type=str,
    #     default="even_sampling",
    # )

    parser.add_argument(
        "--n_conformations",
        help=(
            "Number of conformations to make when sampling evenly"
            " in time from trajectory"
        ),
        type=int,
        default=1,
    )

    parser.add_argument(
        "--limit_n_traj_subfiles",
        help=(
            "Limit the sampling to the first N dcd files."
            " By default no limit is imposed."
        ),
        type=int,
        default=None,
    )

    parser.add_argument(
        "--traj_extension",
        help="File extension of the trajectory files. Default is .dcd",
        type=str,
        default=".dcd",
    )

    parser.add_argument(
        "--output_dir",
        help=(
            "Directory to save the sampled conformations to."
            " Default is local dir!"
        ),
        type=str,
        default=".",
    )

    parser.add_argument(
        "--random_startpoint_seed",
        help="Seed to use to select a random startpoint in the trajectory",
        type=int,
        default=500,
        required=False,
    )

    parser.add_argument(
        "--rnd_start",
        help="Whether to randomly select a startpoint in the trajectory",
        action="store_true",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--use_contiguous_conformations",
        help="Whether to ensure sampled conformations are contiguous or not",
        action="store_true",
        default=False,
        required=False,
    )

    parser.add_argument(
        "--investigate_trajectory_files",
        help="Whether to only investigate a trajectory dataset",
        action="store_true",
        default=False,
        required=False,
    )

    # parser.add_argument(
    #     "--rmsd",
    #     help="RMSD in nm to use for waymark sampling",
    #     type=float,
    #     default=0.3,
    #     required=False,
    # )

    parser.add_argument(
        "--digits",
        help=(
            "Number of digits (supplemented with leading zeros)"
            " in output file"
        ),
        type=int,
        default=6,
        required=False,
    )

    # parser.add_argument(
    #     "--waymarking_plots",
    #     help=("Directory to create and put waymarking plots into"),
    #     type=str,
    #     default="waymarking_plots",
    #     required=False,
    # )
    return parser


def get_name():
    return "conformations_sampling"


def get_trajfiles(
    trajfiles_dir: str, verbose: bool, traj_extension: str = ".dcd"
) -> List[str]:
    """Grab an ordered list of trajectory files

    Args:
        trajfiles_dir (str): Directory containing trajectory files
        verbose (bool): Increase output verbosity
        traj_extension (str, optional): File extension of trajectory files.
        Defaults to '.dcd'.

    Returns:
        list[str]: Alphanumerically ordered list of full trajectory file paths
    """
    trajfiles_path = os.path.join(trajfiles_dir, "*{}".format(traj_extension))
    if verbose:
        print("Traj files path to glob: {}".format(trajfiles_path))
    trajfiles = sorted(glob.glob(trajfiles_path))
    if verbose:
        for i, trajfile in enumerate(trajfiles):
            print("Trj file path {}: {}".format(i, trajfile))
    return trajfiles


def get_topfile(topfile: str, verbose: bool) -> str:
    """Grab the path to the topology file with option
     to output path

    Args:
        topfile (str): Path to topology file
        verbose (bool): Increase output verbosity

    Returns:
        str: Topology file path
    """
    if verbose:
        print("Top file path: {}".format(topfile))
    return topfile


def load_traj(trajfile: str, topfile: str, verbose: bool) -> mdt.Trajectory:
    """Load a single trajectory file into memory using
     mdtraj library

    Args:
        trajfile (str): Trajectory file path
        topfile (str): Topology file path
        verbose (bool): Increase output verbosity

    Returns:
        mdt.Trajectory: Trajectory object
    """
    t = mdt.load(trajfile, top=topfile)
    if verbose:
        print("mdt.load returns type: {}".format(type(t)))
    return t


"""
Waymarking is currently disabled. The function does not work
when only a single trajectory file is used.
The waymarking is also done incorrectly, with
new conformations only being checked against the
fist conformation, instead of all conformations
in the ensemble

We should redo waymarking entirely in the future
"""


def waymark(
    trajfiles: List[str],
    topfile: str,
    rmsd: float,
    verbose: bool,
    atom_indices=None,
) -> Tuple[
    List[List[int]],
    List[List[np.int64]],
    List[np.ndarray[Any, Any]],
    mdt.Trajectory,
    List[int],
]:
    """This function is designed to take a list of trajectory files making
    up a long trajectory) and return all non-redundant (according to rmsd
    metric) conformations which exist in a mdt.Trajectory (which is not
    contiguous or linear in simulation time!)

    Args:
        trajfiles (list[str]): List of trajectory filepaths
        topfile (str): Topology file path
        rmsd (float): Threshold for RMSD calculation (nm)
        verbose (bool): Increase output verbosity
        atom_indices (atom_indicesarray_like, optional): Subselection of
        atoms to use in RMSD calculation. Defaults to None.

    Returns:
        Tuple[
        list[list[int]],
        list[list[np.int64]],
        list[np.ndarray[np.int64,np.int64]],
        mdt.Trajectory,
        list[int]
        ]:
        List of lists (one per trajectory file) of indexes of
        waymark-kept conformations,
         List of lists (one per trajectory file) of cumulative
         count of waymark-kept conformations,
         List of arrays (one per trajectory file), each holding a
         [1, len(traj)] with rmsd values of reference frame compared to all
         frames in traj file,
         All non-redundant conformations found by waymarking,
         Number of non-redundant conformations (one entry per trajectory file)
    """
    if verbose:
        print(
            "Using {} trajectory files.\nThey are:{}".format(
                len(trajfiles), trajfiles
            )
        )

    waymarks = []  # list[list[int]]], with one entry per traj file (.dcd)
    counts = []  # list[list[np.int64]], with one entry per traj file
    assignments = (
        []
    )  # list[np.ndarray[np.int64, np.int64]] with one entry per traj file
    non_redundant_confs_counter_list = (
        []
    )  # counts number of conformations which are non-redundant at end of
    # each traj file processing

    # keep hold of all the conformations which are rmsd different by more than
    # rmsd thresh
    non_redundant_confs: mdt.Trajectory = trajfiles[0][0]
    non_redundant_confs_counter = 1
    frame_ind_list: list[int] = [0]

    # then for each trajectory (including the first one, calculate the
    # rmsd of the conformation from the original conformation and all
    # previously
    # identified conformations which exceed rmsd threshold)
    # loop over traj files
    for traj_file_n, trajfile in enumerate(trajfiles):
        if verbose:
            print("Loading trajfile {}: {}".format(traj_file_n, trajfile))
        traj: mdt.Trajectory = load_traj(trajfile, topfile, verbose)

        # we already kept the first conformation, so
        if traj_file_n == 0:
            continue
        else:
            frame_ind_list = []
            # if we're on the second or later trajfile, add the conformations
            # to the end of traj obj holding existin >rmsd thresh differing
            # traj'es
            traj = non_redundant_confs.join(traj)
            if verbose:
                print(non_redundant_confs)

        # get the rmsd of structure from the first conformation for each other
        # conformation
        rms_dists = mdt.rmsd(traj, traj, frame=0, atom_indices=atom_indices)
        # grab the indices of the conformations which were different by
        # >rmsd threshold
        frame_ind = int(np.argmax(rms_dists > rmsd))
        # convert array to at least 2d
        rms_dists_list = np.atleast_2d(rms_dists)

        # using j>0 while loop with argmax still works for multiple dcd file
        # method as reference trajectory still includes first conformation
        # from the first trajectory file
        while frame_ind > 0:
            # only append to the non_redundant list of conformations if it is
            # not already going to be in there! (this wastes some computation
            # but oh well for now)
            if frame_ind >= non_redundant_confs_counter:
                non_redundant_confs = non_redundant_confs.join(traj[frame_ind])
                non_redundant_confs_counter += 1
            frame_ind_list.append(frame_ind)
            rms_dists = mdt.rmsd(
                traj, traj, frame=frame_ind, atom_indices=atom_indices
            )
            rms_dists_list = np.append(
                rms_dists_list, np.atleast_2d(rms_dists), axis=0
            )
            frame_ind = int(np.argmax(rms_dists_list.min(axis=0) > rmsd))

        # these are kept for continuity in updating code but frankly will no
        # longer be useful to analyse things because
        # each iteration of for loop is not independent (as non-redundant_confs
        # is built upon each time!)
        waymarks.append(frame_ind_list)
        assignments.append(rms_dists_list.argmin(axis=0))
        counts.append(
            [
                (rms_dists_list.argmin(axis=0) == i).sum()
                for i in range(len(frame_ind_list))
            ]
        )
        non_redundant_confs_counter_list.append(non_redundant_confs_counter)

        if verbose:
            print(
                "By end of file {} there are {} non-redundant conformations"
                " using rmsd {}".format(
                    traj_file_n,
                    non_redundant_confs_counter,
                    rmsd,
                )
            )

    return (
        waymarks,
        counts,
        assignments,
        non_redundant_confs,
        non_redundant_confs_counter_list,
    )


def plot_waymarking(
    waymarking_plots: str,
    traj_steps: np.uint64,
    waymarks: List[int],
    suffix: str | None = None,
    xlabel: str = "n_frames",
):
    """Create and save plots of how the number of waymark-sampled trajectories
    increases as a function of the number of frames or number of trajectory
    files

    Args:
        waymarking_plots (str): Path to dir to put waymarking plots into
        traj_steps (np.uint64):
        waymarks (list[int]): Cumulative count of waymark-sampled trajectories
        suffix (str, optional): Descriptive suffix for plot names.
        Defaults to None.
        xlabel (str, optional): Label for plot x-axis. Defaults to n_frames
    """
    frames = []
    sum_frames = []
    nw = 0
    for nw, w in enumerate(waymarks):
        frames.append(w)
        sum_frames.append(nw)
        nw += 1
        frames.append(w)
        sum_frames.append(nw)
    frames.append(traj_steps - 1)
    sum_frames.append(sum_frames[-1])

    # make sure dir exists to put the plots
    if not os.path.exists(waymarking_plots):
        os.makedirs(waymarking_plots)

    plt.plot(frames, sum_frames)
    plt.xlabel(xlabel)
    plt.ylabel("n_waymarks")
    plt.tight_layout()
    plt.savefig(
        os.path.join(waymarking_plots, "waymark_{}.pdf".format(suffix))
    )
    plt.savefig(
        os.path.join(waymarking_plots, "waymark_{}.png".format(suffix))
    )
    plt.clf()
    return


def list_waymark_occupancy(
    traj_steps: np.uint64, counts: List[int], suffix: str | None = None
):
    """For each increment of the total number of waymark-sampled conformations,
     print verbose output of the index and the proportion of the total number
     of frames that were sampled

    Args:
        traj_steps (np.uint64): Total number of frames in trajectory
        counts (_type_): Cumulative count of waymark-sampled conformations
        for this trajectory
        suffix (str, optional): Descriptive label for trajectory.
        Defaults to None.
    """
    for frame in np.argsort(counts)[::-1]:
        print(
            "Waymark {} {:2d}: {:5.2f}%".format(
                suffix, frame, counts[frame] * 100 / traj_steps
            )
        )
    return


def write_non_redundant_confs(
    non_redundant_confs: mdt.core.trajectory.Trajectory,
    output_dir: str,
    verbose: bool = False,
    digits: int = 6,
):
    """Write waymark-sampled conformations to disk

    Args:
        non_redundant_confs (mdt.core.trajectory.Trajectory): Sampled
        conformations
        output_dir (str): Directory to write conformations into
        verbose (bool, optional): Increase output verbosity.
        Defaults to False.
        digits (int, optional): Digits to use in saved filenames -
        supplemented by leading zeros. Defaults to 6.
    """
    create_output_dir(output_dir, verbose=verbose)
    for i, traj in enumerate(non_redundant_confs):
        conf_file = os.path.join(
            output_dir, "waymark_{}.pdb".format(str(i).zfill(digits))
        )
        traj.save(conf_file)
    return


def get_traj_steps(
    trajfiles: list, topfile: str, verbose: bool
) -> Tuple[np.uint64, List[np.uint64]]:
    """Load all the trajectories and count the frames

    Args:
        trajfiles (list): List of trajectory files
        topfile (str): Topology file path
        verbose (bool): Increase output verbosity

    Returns:
        np.uint64: Number of frames in whole trajectory
        list[np.uint64]: Number of frames in each (dcd) file making up
        trajectory
    """
    traj_steps = 0
    steps_per_file = []
    for traj in trajfiles:
        t = load_traj(traj, topfile, verbose)
        len_t = len(t)
        traj_steps += len_t
        steps_per_file.append(np.uint64(len_t))

    if verbose:
        print("Total of {} steps in trajectory files".format(traj_steps))
    return np.uint64(traj_steps), steps_per_file


def create_output_dir(output_dir: str, verbose: bool) -> None:
    """Create dir if does not exist

    Args:
        output_dir (str): New directory file path
        verbose (bool): Increase output verbosity
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if verbose:
            print("{} already exists. Proceeding...".format(output_dir))
    return


def sample_dcd_evenly(
    trajfiles: List[str],
    topfile: str,
    traj_indices: List[List[int]],
    verbose: bool,
    output_dir: str,
    digits: int = 6,
    enable_progressbar: bool = False,
):
    """Reads a traj file (single dcd) and then selects which conformation(s)
    to save from sample_index list. Saves as pdb with indexes ranging over
    multiple dcd files (ie: a whole trajectory)

    Args:
        trajfiles (list[str]): List of filepaths for dcd files
        topfile (str): PDB for MD simulation
        traj_indices (list[list[int]]): List containing list of indices of
        conformation(s) in each dcd which need to be saved to file as pdb_
        verbose (bool): Increase output verbosity
        output_dir (str): Directory to save the sampled conformations to
        digits (int, optional): Number of digits to use in saved filenames
        enable_progressbar (bool, optional): Whether to show progress bar
    """
    create_output_dir(output_dir, verbose=verbose)

    conformation_counter = 0
    for traj_idx, traj_conf_idxs in enumerate(traj_indices):
        # check if need any conformations from this dcd before loading it
        if len(traj_conf_idxs) > 0:
            trajfile = trajfiles[traj_idx]
            if verbose:
                print(
                    "trajfile {} with pdbs being saved is {}".format(
                        traj_idx,
                        trajfile,
                    )
                )

            # load the trajectory file
            traj = load_traj(trajfile, topfile, verbose)

            # grab the conformations selected for saving as pdbs
            selected_conformations = traj[traj_conf_idxs]
            if verbose:
                print(
                    "number of selected conformations: {}".format(
                        len(selected_conformations)
                    )
                )
            progressbar = tqdm(
                total=len(selected_conformations),
                disable=not enable_progressbar,
            )
            # save the conformations selected from this trajectory file to disk
            for i_conf, conf in enumerate(selected_conformations):
                conf_file = os.path.join(
                    output_dir,
                    "conformation_{}.pdb".format(
                        str(conformation_counter).zfill(digits)
                    ),
                )
                conf.save(conf_file)
                conformation_counter += 1
                if verbose and not enable_progressbar:
                    print(
                        "Saved conformation_{}.pdb from traj file"
                        " {} ({}) to location {}".format(
                            str(conformation_counter).zfill(digits),
                            traj_idx,
                            trajfile,
                            conf_file,
                        )
                    )
                progressbar.update(1)
                progressbar.set_description(
                    f"conformation_ \
                        {str(conformation_counter).zfill(digits)}.pdb"
                )
            progressbar.close()

        else:
            # skip this traj file
            continue

    return


def get_traj_indices(
    traj_steps: np.uint64,
    steps_per_file: List[np.uint64],
    verbose: bool,
    n_conformations: np.uint64 = np.uint64(1),
    rnd_start: bool = False,
    seed: int = 1385737,
    contiguous: bool = False,
) -> List[List[int]]:
    """Determine the trajectory frame indices which you want to
    sample from full trajectory

    Args:
        traj_steps (np.uint64): Total number of frames in entire
        trajectory
        steps_per_file (list[np.uint64]): List of number of frames per
        trajectory file (1 entry per file)
        verbose (bool): Increase output verbosity
        n_conformations (np.uint64, optional): Number of conformations to
        sample. Defaults to 1.
        rnd_start (bool, optional): Whether to start sampling at a random
        frame. Defaults to False.
        seed (int, optional): Seed to use to generate random starting frame.
        Defaults to 1385737.
        contiguous (bool, optional): Whether to sample frames contiguously.
        Defaults to False.

    Returns:
        list[list[int]]: List of lists (one per trajectory file) filled with
        indices of frames to sample
    """

    # check you have enough conformations to satisfy the number requested
    if np.uint64(traj_steps) < np.uint64(n_conformations):
        print(
            "Number of conformations requested ({}) is larger than number in"
            " trajectory ({}). Exiting.".format(n_conformations, traj_steps)
        )
        exit(1)

    # check if user wants to randomify starting point of trajectory and if so,
    # adjust traj_steps and steps_per_file as if you started from there.
    # If there are then keep trying until
    # you can satisfy the number of conformations user requested
    if rnd_start:
        # grab rnd int in range of traj_steps
        start_int = traj_steps + 1
        np.random.seed(seed)
        # make sure enough conformations still exist if using this number,
        # else regen and try again
        while int(traj_steps - start_int) < int(n_conformations):
            start_int = np.random.randint(0, high=traj_steps)
    else:
        start_int = 0

    if verbose:
        print("Start integer is {}".format(start_int))

    # ensure all conformations are contiguous
    if contiguous:
        overall_indices = np.round(
            np.arange(start_int, start_int + n_conformations, 1)
        )
    else:
        # now select them as evenly in time as you can, with the assumption
        # that all time steps are equal
        # get the indexes equally spaced starting at 0 or random int
        overall_indices = np.round(
            np.linspace(start_int, traj_steps - 1, n_conformations)
        ).astype(int)

    # now assign each index to a traj file and pick out the index in
    # the respective traj file
    conformation_counter = 0
    traj_indices = []
    for traj_file_idx, n_traj_file_conformations in enumerate(steps_per_file):
        traj_file_conformations = []
        for conformation_idx in range(n_traj_file_conformations):
            # list of conformations indices from this traj file
            if conformation_counter in overall_indices:
                traj_file_conformations.append(conformation_idx)
            conformation_counter += 1
        traj_indices.append(traj_file_conformations)

    if verbose:
        for idx, conformations in enumerate(traj_indices):
            print(
                "traj file {} will produce output pdbs"
                " from conformation(s): {}".format(
                    idx,
                    conformations,
                )
            )

    return traj_indices


def main(args):
    trajfiles = get_trajfiles(
        args.trajfiles_dir, args.verbose, args.traj_extension
    )
    if args.limit_n_traj_subfiles:
        trajfiles = trajfiles[: args.limit_n_traj_subfiles]

    topfile = get_topfile(args.topfile, args.verbose)

    traj_steps, steps_per_file = get_traj_steps(
        trajfiles, topfile, args.verbose
    )
    if args.verbose:
        if len(np.unique(steps_per_file)) != 1:
            print(
                "There are different numbers of conformations in different"
                " files making up the trajectory.\nThe numbers of"
                " conformations are:"
            )
            for idx, confs in enumerate(steps_per_file):
                print("Traj file: {} has {} conformations".format(idx, confs))

    if args.investigate_trajectory_files:
        print(
            "Finished investigation of trajectory files. Cmd line arg"
            " --investigate_trajectory_files gives no output files"
        )
        return

    # only available sampling method at this time is even sampling.
    # waymarking has been disabled for now, as it is not correctly
    # implemented
    args.sampling_method = "even_sampling"

    if args.sampling_method == "even_sampling":
        # create function to get indices of N conformations to sample from
        # entire trajectory
        # output is list[list[int]]
        traj_indices = get_traj_indices(
            traj_steps,
            steps_per_file,
            args.verbose,
            n_conformations=args.n_conformations,
            rnd_start=args.rnd_start,
            seed=args.random_startpoint_seed,
            contiguous=args.use_contiguous_conformations,
        )

        # create functio to take list[list[int]] and save the pdbs to file
        sample_dcd_evenly(
            trajfiles,
            topfile,
            traj_indices,
            args.verbose,
            args.output_dir,
            args.digits,
            args.tqdm,
        )

        """
    elif args.sampling_method == "waymark":
        # we want to get: Tuple[list[int], list[np.int64],
        # np.ndarray[np.int64,np.int64]]
        # First is waymarks (frame_ind_list -> the trajectory frames which
        # constitute a new conformation)
        # Second is counts (the number of frames assigned to each conformation
        # Third is )
        (
            waymarks_list,
            counts_list,
            assignments_list,
            non_redundant_confs,
            non_redundant_confs_counter_list,
        ) = waymark(trajfiles, topfile, args.rmsd, args.verbose)
        # All we really care about here is getting the list of
        # non-redundant conformations,
        # which we should now have

        cumulative_steps_per_file = np.cumsum(steps_per_file).tolist()

        # Can not plot waymarking beyond the first traj file but lets
        # do this for first file (highest gradient)
        plot_waymarking(
            args.waymarking_plots,
            steps_per_file[0],
            waymarks_list[0],
            suffix="singlefile",
        )
        # Unless we only care about file-level granularity it the
        # plotplot_waymarking
        plot_waymarking(
            args.waymarking_plots,
            cumulative_steps_per_file[-1],
            waymarks_list[:][-1],
            suffix="allfiles",
            xlabel="n_traj_fles",
        )

        if args.verbose:
            # Can not list_waymark_occupancy plot beyond the first traj
            # filebut lets do this for first file
            # list_waymark_occupancy(traj_steps, counts)
            list_waymark_occupancy(
                steps_per_file[0], counts_list[0], suffix="singlefile"
            )
            # Unless we only care about file-level granularity in the plot
            list_waymark_occupancy(
                cumulative_steps_per_file[-1],
                counts_list[:][-1],
                suffix="allfiles",
            )

        # We now have a mdt.Trajectory which holds all the
        # non-redundant conformations.
        # Let's write them out
        write_non_redundant_confs(
            non_redundant_confs, args.output_dir, args.digits
        )
        """

    else:
        # method of sampling not recognised
        print(
            "Sampling method {} not recognised. Exiting.".format(
                args.sampling_method
            )
        )

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print("{}, {}".format(arg, getattr(args, arg)))
    main(args)
