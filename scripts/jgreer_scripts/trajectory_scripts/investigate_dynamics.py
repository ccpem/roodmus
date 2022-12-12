"""
This script is designed to:
take a directory full of pdb files,
read their filenames into a list,
copy files to new directories based on a sampling granularity level
Note that it will not explicitly be checked whether filename indexes
are contiguous before sorting them  
"""

import numpy as np
import glob2 as glob
import os
import argparse
import shutil

def run_main(args):
    # read in and sort the list of conformations
    conformations_path=os.path.join(args.conformations_dir, args.conformations_searchstring)
    conformations = sorted(glob.glob(conformations_path))
    if args.debug:
        print("Sorted conformations read from file are:")
        for conf in conformations:
            print(conf)
        print("\n")

    # get rid of any unwanted conformations
    conformations = conformations[:args.limit_contiguous_conformations]
    if args.debug and args.limit_contiguous_conformations:
        print("After dropping requested conformations the list is:")
        for conf in conformations:
            print(conf)
        print("\n")

    # sample with required granularity
    conformations = np.array(conformations, dtype=str)[::args.sampling_granularity].tolist()
    if args.debug:
        print("After sampling conformations with requested granularity of {} the list is:".format(args.sampling_granularity))
        for conf in conformations:
            print(conf)
        print("\n")

    # create new dir to save sampled conformations to
    if not os.path.exists(args.sampled_conformations_dir):
        os.makedirs(args.sampled_conformations_dir)
        if args.debug:
            print("Created new directory:\n{}".format(args.sampled_conformations_dir))
            print('')
    
    # save/copy sampled conformations (with same filenames as original)
    for conf in conformations:
        shutil.copyfile(conf, os.path.join(args.sampled_conformations_dir, os.path.basename(conf)))
        if args.debug:
            print('Copied {} to {} directory'.format(conf, args.sampled_conformations_dir))
    return

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--conformations_dir",
        help="Directory path to sampled conformations which will be sorted and copied",
        type=str,
        default=""
    )

    parser.add_argument(
        "--limit_contiguous_conformations",
        help="Number of contiguous conformations (pdb files) to grab granular samples from",
        type=int,
        default=None,
        required=False
    )

    parser.add_argument(
        "--sampling_granularity",
        help="Granularity with which to sample list of conformations. 1 is use all",
        type=int,
        default=1,
        required=False
    )

    parser.add_argument(
        "--sampled_conformations_dir",
        help="Directory to create and put sampled conformations into",
        type=str,
        default="Sampled_Conformations"
    )

    parser.add_argument(
        "--conformations_searchstring",
        help="String (including bash wildcards) to search in conformations_dir for conformation files with",
        type=str,
        default="*.pdb",
        required=False
    )

    parser.add_argument(
        "--debug",
        help="Debug flag to print additional info during processing",
        type=bool,
        default=False,
        required=False
    )

    args = parser.parse_args()
    if args.debug:
        for arg in vars(args):
            print('{}, {}'.format(arg, getattr(args, arg)))
    run_main(args)