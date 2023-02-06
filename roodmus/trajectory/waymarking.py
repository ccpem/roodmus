"""Sampling a molecular dynamics trajectory and saving the conformations to PDB files."""

import numpy as np
from matplotlib import pyplot as plt
import mdtraj as mdt
import glob2 as glob
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB import parse_pdb_header
from Bio.PDB.MMCIFParser import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.mmcifio import MMCIFIO
from gemmi import cif
import pandas as pd
from typing import Tuple
import argparse
import os

def add_arguments(parser):
    parser.add_argument(
        "--trajfiles_dir_path",
        help="Path to directory holding (dcd) trajectory files which make up the whole trajectory.",
        type=str,
        default="/mnt/parakeet_storage/trajectories/DESRES-Trajectory_sarscov2-13795965-no-water/sarscov2-13795965-no-water"
    )

    parser.add_argument(
        "--topfile_path",
        help="The pdb holding the structure of molecule (no solvent)",
        type=str,
        default="/mnt/parakeet_storage/trajectories/DESRES-Trajectory_sarscov2-13795965-no-water/sarscov2-13795965-no-water/DESRES-Trajectory_sarscov2-13795965-no-water.pdb"
    )

    parser.add_argument(
        "--debug",
        help="Whether to print debugging statements",
        type=bool,
        default=False
    )

    parser.add_argument(
        "--sampling_method",
        help="Choose whether to sample a trajectory uniformly in time (even_sampling) or by >rmsd threshold (waymark)",
        type=str,
        default='even_sampling'
    )

    parser.add_argument(
        "--n_conformations",
        help="Number of conformations to make when sampling evenly in time from trajectory",
        type=int,
        default=2
    )

    parser.add_argument(
        "--limit_n_traj_subfiles",
        help="Limit the sampling to the first N dcd files. By default no limit is imposed.",
        type=int,
        default=None
    )

    parser.add_argument(
        "--traj_extension",
        help="File extension of the trajectory files. Default is .dcd",
        type=str,
        default=".dcd"
    )

    parser.add_argument(
        "--output_dir",
        help="Directory to save the sampled conformations to. Default is local dir!",
        type=str,
        default='.'
    )
    return parser

def get_name():
    return "waymarking"

# TOPFILE = '/mnt/parakeet_storage/trajectories/DESRES-Trajectory_sarscov2-13795965-no-water/system.pdb'
TOPFILE = '/mnt/parakeet_storage/trajectories/DESRES-Trajectory_sarscov2-13795965-no-water/sarscov2-13795965-no-water/DESRES-Trajectory_sarscov2-13795965-no-water.pdb'
# TRAJFILES = '/mnt/parakeet_storage/trajectories/DESRES-Trajectory_sarscov2-13795965-no-water/sarscov2-13795965-no-water/*.dcd'
TRAJFILESDIR = '/mnt/parakeet_storage/trajectories/DESRES-Trajectory_sarscov2-13795965-no-water/sarscov2-13795965-no-water'
DEBUG = True
WAYMARK = False
EVEN_SAMPLING = True


def get_trajfiles(trajfiles_dir_path: str, debug:bool, traj_extension: str='.dcd') -> list[str]:

    trajfiles_path=os.path.join(trajfiles_dir_path, '*{}'.format(traj_extension))
    if debug:
        print('Traj files path to glob: {}'.format(trajfiles_path))
    trajfiles = sorted(glob.glob(trajfiles_path))
    if debug:
        for i,trajfile in enumerate(trajfiles):
            print('Trj file path {}: {}'.format(i, trajfile))
    return trajfiles

def get_topfile(topfile_path:str, debug:bool) -> str:

    if debug:
        print('Top file path: {}'.format(topfile_path))
    return topfile_path

def load_traj(trajfile: str, topfile: str, debug: bool):
    t = mdt.load(trajfile, top=topfile)
    if debug:
        print('mdt.load retruns type: {}'.format(type(t)))
    return t

def read_pdb_header(filename: str, debug: bool):
    # filename: str, structure_id: str, debug: bool
    """
    Wonderful quote from BioPython docs:
    You can extract the header and trailer (simple lists of strings) of the PDB file from the PDBParser object
    with the get header and get trailer methods. Note however that many PDB files contain headers with
    incomplete or erroneous information. Many of the errors have been fixed in the equivalent mmCIF files.
    Hence, if you are interested in the header information, it is a good idea to extract information from mmCIF
    files using the MMCIF2Dict tool described above, instead of parsing the PDB header
    But all I got is a PDB
    """
    with open(filename, "r") as handle:
        header_dict = parse_pdb_header(handle)
    if debug:
        print(header_dict)
        
def waymark(trajfiles: list[str], topfile: str, rmsd: float, debug: bool, atom_indices=None)\
 -> Tuple[list[list[int]], list[list[np.int64]], list[np.ndarray[np.int64,np.int64]], mdt.Trajectory, list[int]]:
    '''
    This function is designed to take a list of trajectory files making up a long trajectory) and return all
    non-redundant (according to rmsd metric) conformations which exist in a mdt.Trajectory (which is not contiguous or linear in simulation time!)

    atom_indices: atom_indicesarray_like

    Identify waymark structures
    '''
    
    print('Using {} trajectory files.\nThey are:{}'.format(len(trajfiles), trajfiles))

    waymarks = [] # list[list[int]]], with one entry per traj file (.dcd)
    counts = [] # list[list[np.int64]], with one entry per traj file
    assignments = [] # list[np.ndarray[np.int64, np.int64]] with one entry per traj file
    non_redundant_confs_counter_list = [] # counts number of confirmations which are non redundant at end of each traj file processing
    
    # keep hold of all the conformations which are rmsd different by more than rmsd thresh
    non_redundant_confs = None
    non_redundant_confs_counter = 0

    # then for each trajectory (including the first one, calculate the rmsd of the conformation from the original conformation and all previously
    # identified conformations which exceed rmsd threshold)
    # loop over traj files
    for traj_file_n, trajfile in enumerate(trajfiles):
        print('Loading trajfile {}: {}'.format(traj_file_n, trajfile))
        traj = load_traj(trajfile, topfile, debug)
        
        # we always keep the first conformation, so
        if traj_file_n==0:
            jlist = [0]
            non_redundant_confs = traj[0]
            non_redundant_confs_counter+=1
        else:
            jlist = []
            # if we're on the second or later trajfile, add the conformations to the end of traj obj holding existin >rmsd thresh differing traj'es
            traj = non_redundant_confs.join(traj)
            print(non_redundant_confs)

        # get the rmsd of structure from the first conformation for each other conformation
        r = mdt.rmsd(traj, traj, frame=0, atom_indices=atom_indices)
        # grab the indices of the conformations which were different by >rmsd threshold
        j = np.argmax(r > rmsd)
        # convert array to at least 2d (not sure why...)
        rlist = np.atleast_2d(r)

        # using j>0 while loop  with argmax still works for multiple dcd file method as reference trajectory still includes first conformation
        # from the first trajectory file 
        while j > 0:
            # only append to the non_redundant list of conformations if it is not already going to be in there! (this wastes some computation but oh well for now)
            if j>=non_redundant_confs_counter:
                non_redundant_confs = non_redundant_confs.join(traj[j])
                non_redundant_confs_counter+=1
            jlist.append(j)
            r = mdt.rmsd(traj, traj, frame=j, atom_indices=atom_indices)
            rlist = np.append(rlist, np.atleast_2d(r), axis=0)
            j = np.argmax(rlist.min(axis=0) > rmsd)

        # these are kept for continuity in updating code but frankly will no longer be useful to analyse things because
        # each iteration of for loop is not independent (as non-redundant_confs is built upon each time!)
        waymarks.append(jlist)
        assignments.append(rlist.argmin(axis=0))
        counts.append([(rlist.argmin(axis=0)==i).sum() for i in range(len(jlist))])
        non_redundant_confs_counter_list.append(non_redundant_confs_counter)

        # if debug:
        print('By end of file {} there are {} conformations using rmsd {}'.format(traj_file_n, non_redundant_confs_counter, rmsd))

    return waymarks, counts, assignments, non_redundant_confs, non_redundant_confs_counter

        

    """
    #--------------------------------------------------
    traj = load_traj(trajfiles[0], topfile, debug)

    # get the rmsd of structure from its original conformation, traj0 (which is 1d np array of rmsd of each traj from the frame'th conformation in the dcd file)
    r = mdt.rmsd(traj, traj, frame=0, atom_indices=atom_indices)
    # make a list of the indices of the maximum rmsd values found
    j = np.argmax(r > rmsd)
    # convert array to at least 2d (not sure why...)
    rlist = np.atleast_2d(r)
    # start a list and fill it with the initial conformation to begin with
    jlist = [0]
    # while loop which seems to be intended to fill:
    # jlist with the trajectory frames which constitute a new conformation
    # counts with the number of frames respectively assigned to each conformation
    # assignments with the indices of 
    # I think rlist in a 2d array where axis0 is scanned by different traj frames and axis1 is ...???
    while j > 0:
        jlist.append(j)
        r = mdt.rmsd(traj, traj, frame=j, atom_indices=atom_indices)
        rlist = np.append(rlist, np.atleast_2d(r), axis=0)
        j = np.argmax(rlist.min(axis=0) > rmsd)
    
    waymarks = jlist
    assignments = rlist.argmin(axis=0)
    counts = [(assignments==i).sum() for i in range(len(jlist))]
    if debug:
        print('assigments shape: {}'.format(assignments.shape))
        print('waymarks entries: {}\ncounts entries: {}\nassignments entries: {}'.format(len(waymarks), len(counts), len(assignments)))
        print('waymarks type: {}\t{}\ncounts type: {}\t{}\nassigments type: {}\t{}'.format(type(waymarks), type(waymarks[0]), type(counts), type(counts[0]), type(assignments), type(assignments[0])))
        print('waymarks: {}\ncounts: {}\nassignments: {}'.format(waymarks, counts, assignments))
    return waymarks, counts, assignments
    #--------------------------------
    """

def plot_waymarking(traj_steps: np.uint64, waymarks: list[int], suffix: str=None):
    snapshots = []
    n_ways = []
    nw = 0
    for w in waymarks:
        snapshots.append(w)
        n_ways.append(nw)
        nw += 1
        snapshots.append(w)
        n_ways.append(nw)
    snapshots.append(traj_steps-1)
    n_ways.append(n_ways[-1])

    plt.plot(snapshots, n_ways)
    plt.xlabel('n_frames')
    plt.ylabel('n_waymarks')
    plt.savefig('waymark_{}.pdf'.format(suffix))
    plt.savefig('waymark_{}.png'.format(suffix))
    plt.clf()
    return

def list_waymark_occupancy(traj_steps: np.uint64, counts, suffix: str=None):
    for i_way in np.argsort(counts)[::-1]:
        print('Waymark {} {:2d}: {:5.2f}%'.format( suffix, i_way, counts[i_way] * 100 / traj_steps))
    return

def write_waymark_pdbs(t: mdt.core.trajectory.Trajectory, waymarks):
    waytraj = t[waymarks]
    for i_way, traj in enumerate(waytraj):
        traj.save('waymark_{:06d}.pdb'.format(i_way))
    return

def write_non_redundant_confs(non_redundant_confs: mdt.core.trajectory.Trajectory, output_dir: str):
    
    create_output_dir(output_dir, debug=debug)
    for i, traj in enumerate(non_redundant_confs):
        conf_file = os.path.join(output_dir, 'waymark_{:06d}.pdb'.format(i))
        traj.save(conf_file)
    return

def get_traj_steps(trajfiles: list, topfile: str, debug: bool)->Tuple[np.uint64, list[np.uint64]]:
    """Load all the trajectories and sum the states

    Args:
        trajfiles (list): list of trajectory files

    Returns:
        np.uint64: number of states in whole trajectory
        list[np.uint64]: number of states in each (dcd) file making up trajectory
    """
    traj_steps = 0
    steps_per_file = []
    for traj in trajfiles:
        t = load_traj(traj, topfile, debug)
        len_t =len(t)
        traj_steps += len_t
        steps_per_file.append(len_t)

    if debug:
        print('Total of {} steps in trajectory files'.format(traj_steps))
    return traj_steps, steps_per_file

def create_output_dir(output_dir: str, debug: bool)->None:

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        if debug:
            print()
    return 

def sample_dcd_evenly(trajfiles: list[str], topfile: str, traj_indices: list[list[int]], debug: bool, output_dir: str):
    """Reads a traj file (single dcd) and then selects which conformation(s) to save from sample_index list. Saves
    as pdb with indexes ranging over multiple dcd files (ie: a whole trajectory)

    Args:
        trajfiles (list[str]): list of filepaths for dcd files
        topfile (str): pdb for MD simulation
        traj_indices (list[list[int]]): _list containing list of indices of conformation(s) in each dcd which need to be saved to file as pdb_
        debug (bool): _description_
    """
    create_output_dir(output_dir, debug=debug)

    conformation_counter = 0
    for traj_idx, traj_conf_idxs in enumerate(traj_indices):
        # check if need any conformations from this dcd before loading it
        if len(traj_conf_idxs)>0:
            trajfile = trajfiles[traj_idx]
            if debug:
                print('trajfile {} with pdbs being saved is {}'.format(traj_idx, trajfile))
            
            # load the trajectory file 
            traj = load_traj(trajfile, topfile, debug)
            
            # grab the conformations selected for saving as pdbs
            selected_conformations = traj[traj_conf_idxs]
            if debug:
                print('number of selected conformations: {}'.format(len(selected_conformations)))

            # save the conformations selected from this trajectory file to disk
            for i_conf, conf in enumerate(selected_conformations):
                conf_file = os.path.join(output_dir, 'conformation_{:06d}.pdb'.format(conformation_counter))
                conf.save(conf_file)
                conformation_counter += 1
                if debug:
                    print('Saved conformation_{:06d}.pdb from traj file {} ({}) to location {}'.format(conformation_counter, traj_idx, trajfile, conf_file))
        
        
        else:
            # skip this traj file
            continue

    return

def get_traj_indices(traj_steps:np.uint64, steps_per_file: list[np.uint64], debug: bool, n_conformations: np.uint64=1)->list[list[int]]:
    # check you have enough conformations to satisfy the number requested
    if np.uint64(traj_steps)<np.uint64(n_conformations):
        print('Number of conformations requested ({}) is larger than number in trajectory ({}). Exiting.'.format(n_conformations, traj_steps))
        exit(1)
    
    # now select them as evenly in time as you can, with the assumption that all time steps are equal
    # get the indexes equally spaced starting at 0
    overall_indices = np.round(np.linspace(0, traj_steps-1, n_conformations)).astype(int)
    
    # now assign each index to a traj file and pick out the index in the respective traj file
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
    
    if debug:
        for idx, conformations in enumerate(traj_indices):
            print('traj file {} will produce output pdbs from conformation(s): {}'.format(idx, conformations))

    return traj_indices

def main(args):

    trajfiles = get_trajfiles(args.trajfiles_dir_path, args.debug, args.traj_extension)
    if args.limit_n_traj_subfiles:
        trajfiles = trajfiles[:args.limit_n_traj_subfiles]
    
    topfile = get_topfile(args.topfile_path, args.debug)

    read_pdb_header(topfile, args.debug)
    traj_steps, steps_per_file = get_traj_steps(trajfiles, topfile, args.debug)
    if args.debug:
        if len(np.unique(steps_per_file))!=1:
            print('There are different numbers of conformations in different files making up the trajectory.\nThe numbers of conformations are:')
            for idx, confs in enumerate(steps_per_file):
                print('Traj file: {} has {} conformations'.format(idx,confs))

    if args.sampling_method=='even_sampling':
        # create function to get indices of N conformations to sample from entire trajectory
        # output is list[list[int]]
        traj_indices = get_traj_indices(traj_steps, steps_per_file, args.debug, n_conformations=args.n_conformations)

        # create functio to take list[list[int]] and save the pdbs to file
        sample_dcd_evenly(trajfiles, topfile, traj_indices, args.debug, args.output_dir)

    if args.sampling_method=='waymark':
        # currently wayfinding code is only set up for a single dcd file
        # load initial trajectory
        t = load_traj(trajfiles[0], topfile, args.debug)
        # setting rmsd cut of 0.3nm
        r_cut = 0.3
        # we want to get: Tuple[list[int], list[np.int64], np.ndarray[np.int64,np.int64]]
        # First is waymarks (jlist -> the trajectory frames which constitute a new conformation)
        # Second is counts (the number of frames assigned to each conformation
        # Third is )

        waymarks_list, counts_list, assignments_list, non_redundant_confs, non_redundant_confs_counter_list = waymark(trajfiles, topfile, r_cut, args.debug) # still only set up for 1 dcd
        
        # All we really care about here is getting the list of non-redundant conformations,
        # which we should now have

        cumulative_steps_per_file = np.cumsum(steps_per_file).tolist()

        # can no longer plot waymarking beyond the first traj file
        # plot_waymarking(traj_steps, waymarks)
        plot_waymarking(steps_per_file[0], waymarks_list[0], suffix='singlefile')
        # Unless we only care about file-level granularity it the plotplot_waymarking
        plot_waymarking(cumulative_steps_per_file[-1], waymarks_list[:][-1], suffix='allfiles')
        
        # can no longer do list_waymark_occupancy plot beyond the first traj file
        # list_waymark_occupancy(traj_steps, counts)
        list_waymark_occupancy(steps_per_file[0], counts_list[0], suffix='singlefile')
        # Unless we only care about file-level granularity it the plot
        list_waymark_occupancy(cumulative_steps_per_file[-1], counts_list[:][-1], suffix='allfiles')
        
        # no longer need to write out pdbs using write_waymark_pdbs as we have a mdt.Trajectory
        # which holds all the non-redundant conformations. So we write out all of those instead.
        # write_waymark_pdbs(t, waymarks)
        write_non_redundant_confs(non_redundant_confs, args.output_dir)
    
    return


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    if args.debug:
        for arg in vars(args):
            print('{}, {}'.format(arg, getattr(args, arg)))
    main(args)
    
