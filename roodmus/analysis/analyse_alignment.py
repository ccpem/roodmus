"""script to compare estimated 3D alignments from RELION or CryoSPARC to the ground-truth orientation values used in Parakeet data generation"""

### arguments
def add_arguments(parser):
    parser.add_argument("--mrc-dir", help="directory with .mrc files and .yaml config files", type=str)
    parser.add_argument("--meta-file", help="particle metadata file. Can be .star (RELION) or .cs (CryoSPARC)", type=str)
    parser.add_argument("--plot-dir", help="output file name", type=str, default="alignment.png")
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    return parser

def get_name():
    return "analyse_alignment"

### imports
# general
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
# roodmus
from roodmus.analysis.utils import IO

### functions

### plotting functions

### main
def main(args):
    pass

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print('{}, {}'.format(arg, getattr(args, arg)))
    main(args)
