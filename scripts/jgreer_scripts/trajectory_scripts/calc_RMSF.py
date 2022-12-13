import argparse

import numpy as np
import matplotlib.pyplot as plt

import MDAnalysis as mda
from MDAnalysis.topology.PDBParser import PDBParser
from MDAnalysis.analysis import rms

from waymarking import get_trajfiles, get_topfile

from pprint import pprint

# grab the atoms and the rmsf for these atoms and arrange a dict for ez access
chainID_atoms_label = "universe"
chainID_rmsf_label = "rmsf"

universe_bfactors_label = "bfactors"
universe_residues_label = "residues"

def resname_exceptions():
    exceptions = ['CYM', 'HID']
    return exceptions

def rmsf_to_bfactor(rmsf):
    """Convert from rmsf in angstroms to b-factor in angstroms

    Args:
        rmsf (_type_): _description_

    Returns:
        _type_: _description_
    """
    bfactor = (8./3.) * np.power((rmsf*np.pi),2)
    return bfactor

def print_unique_counts(var: np.array, multiple_only: bool=False):
    uniqs, counts = np.unique(var, return_counts=True)
    for uniq, count in zip(uniqs,counts):
        if multiple_only:
            if count>1:
                print('Value {} occurs {} times'.format(uniq, count))
        else:
            print('Value {} occurs {} times'.format(uniq, count))
    return

def print_universe_chain_summary(universe, chainID) -> None:
    print('There are {} resids'.format(len(universe[chainID][chainID_atoms_label].resids)))
    print('There are {} atoms'.format(len(universe[chainID][chainID_atoms_label].atoms)))
    print('There are {} bfactors'.format(len(universe[chainID][chainID_atoms_label].atoms.bfactors)))
    return

def check_MD_and_PDB_chains_match(args, moldyn_universe, refined_pdb) -> None:
    for chainID in args.chain_id:
        print('Checking chain {} of provided universes against each other!')
        
        variable = moldyn_universe[chainID][chainID_atoms_label].resnames
        investigate_object(args, variable, print_vars=False)

        variable2 = refined_pdb[chainID][chainID_atoms_label].resnames
        
        print('resnames refined_PDB : {}'.format(len(variable)))
        print('resnames chainIDs : {}'.format(len(variable2)))
        
        minlen = np.minimum(len(variable), len(variable2))
        i=0
        for (var1, var2) in zip(variable[:minlen], variable2[:minlen]):
            if var1 != var2:
                print('{} - {} : {}'.format(i, var1, var2))
            i+=1

        print('For moldyn universe:')
        print_universe_chain_summary(moldyn_universe, chainID)

        print('For refined pdb universe:')
        print_universe_chain_summary(refined_pdb, chainID)

        print('\n\n\n')
    return

def investigate_object(args, investigate_me, print_vars=False):
    if args.debug:
        print('Investigating: {} of type {}'.format(investigate_me, type(investigate_me)))
        try:
            print('Vars:')
            pprint(vars(investigate_me))
        except:
            print('No vars available')
        print('Dir:')
        pprint(dir(investigate_me))
        try:
            print('len of var: {}'.format(len(investigate_me)))
        except:
            print('No len of object can be retrieved')
    return

def plot_bfactor_comparison(args, universe1, universe2) -> None:
    # plot the residue number against b factor for each chain
    # with calculated trajectory b-factors in blue and RCSB refined b-factors in red
    for i, chainID in enumerate(args.chain_id):
        fig = plt.gcf()
        fig.set_size_inches(100, 10)
        # plot traj calculated bfactors
        plt.plot(universe1[chainID][chainID_atoms_label].resids, rmsf_to_bfactor(universe1[chainID][chainID_rmsf_label].rmsf), color='blue')
        # plot refined from rcsb pdb with reocnstructed density map bfactors
        plt.plot(universe2[chainID][universe_residues_label], universe2[chainID][universe_bfactors_label], color='red')
        plt.xlabel('Residue number (chain {})'.format(chainID))
        plt.ylabel('b-factor ($\AA ^{2}$)')
        plt.grid(which='both')
        plt.xticks(np.arange(min(universe1[chainID][chainID_atoms_label].resids), max(universe1[chainID][chainID_atoms_label].resids)+1, 20))
        plt.tight_layout()

        plt.savefig('bfactor-comparison-chain{}.png'.format(chainID), dpi=200)
        plt.savefig('bfactor-comparison-chain{}.pdf'.format(chainID))
        plt.clf()
    return

def run_main(args):
    
    # grab the paths for the trajectory files
    trajfiles = get_trajfiles(args.trajfiles_dir_path, args.debug, args.traj_extension)
    if args.limit_n_traj_subfiles:
        trajfiles = trajfiles[:args.limit_n_traj_subfiles]

    # get the path for the top file
    topfile = get_topfile(args.topfile_path, args.debug)

    # mdtraf rmsf utility reportedly returns a single value for each conformation
    # so instead am using the MDAnalysis library
    
    # load the traj into `universal' class
    traj = mda.Universe(topfile, trajfiles, topology_format='PDB', permissive=True)
    # select out protein and CA only
    c_alphas = traj.select_atoms('protein and name CA')

    # select out each ChainID
    md_traj = {}
    
    for chainID in args.chain_id:
        md_traj[chainID] = {}
        md_traj[chainID][chainID_atoms_label] = c_alphas.select_atoms('chainID {}'.format(chainID))
        # for each chainID calculate the RMSF
        # work out the rmsf on a per-residue basis
        # (cite https://userguide.mdanalysis.org/stable/references.html#id28)
        md_traj[chainID][chainID_rmsf_label] = rms.RMSF(md_traj[chainID][chainID_atoms_label]).run()
    

    # load in the RCSB published PDB (post-refmac refinement)
    # and grab each chain
    refined_pdb = mda.Universe(args.refined_pdb_path)
    # select only the residues and protein
    refined_pdb = refined_pdb.select_atoms('protein and name CA')

    refined_PDB = {}

    for chainID in args.chain_id:
        refined_PDB[chainID] = {}
        refined_PDB[chainID][chainID_atoms_label] = refined_pdb.select_atoms('chainID {}'.format(chainID))
        refined_PDB[chainID][universe_bfactors_label] = refined_PDB[chainID][chainID_atoms_label].atoms.bfactors
        refined_PDB[chainID][universe_residues_label] = refined_PDB[chainID][chainID_atoms_label].resids

    # match up the residues from the refined PDB to the 
    # moldyn data residues
    for chainID in args.chain_id:
        print('Chain {}'.format(chainID))
        print('There are {} (refined_PDB) resids compared to {} (md_traj)'.format(len(refined_PDB[chainID][universe_residues_label]), len(md_traj[chainID][chainID_atoms_label].resids)))
        print('There are {} (refined_PDB) bfactors compared to {} (md_traj)'.format(len(refined_PDB[chainID][universe_bfactors_label]), len(md_traj[chainID][chainID_rmsf_label].rmsf)))
        res_match = refined_PDB[chainID][universe_residues_label] == md_traj[chainID][chainID_atoms_label].resids
        print('Do the residues all match?: {}'.format(res_match))
        print('')
        if not np.all(res_match == True):
            print('The unique residues of ChainIDs are: {}'.format(np.unique(md_traj[chainID][chainID_atoms_label].resids)))
            print('The unique residues of refined_PDB are: {}'.format(np.unique(refined_PDB[chainID][universe_residues_label])))
            print('The counted unique residues of ChainIDs are: {}')
            print_unique_counts(md_traj[chainID][chainID_atoms_label].resids, multiple_only=True)
            print('\n')
            print('The counted unique residues of refined_PDB are: {}')
            print_unique_counts(refined_PDB[chainID][universe_residues_label],  multiple_only=True)
            print('Residues not all the same! Exiting')
            exit(1)
    
    if args.debug:
        check_MD_and_PDB_chains_match(args, refined_PDB, md_traj)

    plot_bfactor_comparison(args, md_traj, refined_PDB)

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

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
        default="/mnt/parakeet_storage/trajectories/DESRES-Trajectory_sarscov2-13795965-no-water/sarscov2-13795965-no-water/DESRES-Trajectory_sarscov2-13795965-no-water_manuallyremovedplusminuschargenotation.pdb"
    )

    parser.add_argument(
        "--refined_pdb_path",
        help="The refined pdb to compare b-factors to",
        type=str,
        default= "/mnt/parakeet_storage/trajectories/trajic/waymarking/refined_ccpem_job049_bfactorrestraintsremoved.pdb"
        # "/mnt/parakeet_storage/trajectories/trajic/waymarking/refined_ccpem_job038.pdb"
    )

    parser.add_argument(
        "--debug",
        help="Whether to print debugging statements",
        type=bool,
        default=False
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

    parser.add_argument(
        "-cid",
        "--chain_id",
        help="ChainId(s) to make b-factor plots for",
        type=str,
        default=[],
        action='append'
    )

    args = parser.parse_args()
    if args.debug:
        for arg in vars(args):
            print('{}, {}'.format(arg, getattr(args, arg)))
    run_main(args)
