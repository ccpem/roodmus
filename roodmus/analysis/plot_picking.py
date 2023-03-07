"""script to plot statistics from picking analyses and example overlays of picked and truth particles on micrographs"""

### arguments
def add_arguments(parser):
    parser.add_argument("--config-dir", help="directory with .mrc files and .yaml config files", type=str)
    parser.add_argument("--mr-dir", help="directory with .mrc files. The same as 'config-dir' by default", type=str, default=None)
    parser.add_argument("--meta-file", help="particle metadata file. Can be .star (RELION) or .cs (CryoSPARC)", type=str)
    parser.add_argument("-N", "--num-ugraphs", help="number of micrographs to consider in analyses. Default 'all'", type=int, default=None)
    parser.add_argument("--box-width", help="Full width of overlay boxes on images", type=float, default=50., required=False)
    parser.add_argument("--box-height", help="Full height of overlay boxes on images", type=float, default=50., required=False)
    parser.add_argument("--particle-diameter", help="Expected maximum particle diameter. Used to limit search radius for matching picked particles to truth particles", type=float, default=250., required=False)
    # parser.add_argument("--pixel-bin-width", help="Number of image pixels to bin together for histograms", type=int, default=100, required=False)
    # parser.add_argument("--plot-truth-centres", help="Plot the ground-truth particle centres", action="store_true")
    # parser.add_argument("--plot-per-ugraph", help="Plot the results per micrograph", action="store_true")
    # parser.add_argument("--plot-collective-boundary", help="Plot the collective boundary investigation", action="store_true")
    # parser.add_argument("--plot-collective-depth", help="Plot the collective depth investigation", action="store_true")
    # parser.add_argument("--plot-collective-overlap", help="Plot the collective overlap investigation", action="store_true")
    parser.add_argument("--plot-dir", help="output file name", type=str)
    parser.add_argument("--plot-types", help="types of analysis results to plot", type=str, nargs="+")
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    return parser

def get_name():
    return "analyse_picking"

### imports
# general
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np
import mrcfile
from typing import Tuple
# roodmus
from roodmus.analysis.analysis import particle_picking

### plotting functions
def _twoD_image_bboxs(particles_x: np.array, particles_y: np.array, box_width: float, box_height: float, verbose: bool=False)->list[list[float]]:

    box_half_width = box_width/2.
    box_half_height = box_height/2.
    
    if verbose:
        print('Using box half width: {} and half height: {}'.format(box_half_width, box_half_height))

    # now fill a list with x,y point positions of the particles
    twod_pos = []
    for (x,y) in zip(particles_x, particles_y):
        twod_pos.append([float(x),float(y)])

    # use this list to fill a list of boxes, each corresponding to a particle
    boxes = []
    for i in range(0, len(twod_pos)):
        temp_box = [twod_pos[i][0]-box_half_width, twod_pos[i][1]-box_half_height, twod_pos[i][0]+box_half_width, twod_pos[i][1]+box_half_height]
        boxes.append(temp_box)

    return boxes

def label_micrograph_truth(particles: pd.DataFrame, ugraph_index: int=0, mrc_dir: str=None,
                           box_width: int=50, box_height: int=50,
                           verbose: bool=False)->Tuple[plt.Figure, plt.Axes]:
    # get the micrograph name
    ugraph_filename = np.unique(particles["ugraph_filename"])[ugraph_index]
    print(f"plotted index {ugraph_index}; micrograph: {ugraph_filename}")
    ugraph_path = os.path.join(mrc_dir, ugraph_filename)
    particles_ugraph = particles.groupby("ugraph_filename").get_group(ugraph_filename)
        
    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(ugraph_path) as mrc:
        data = mrc.data

        fig, ax = plt.subplots(figsize=[16, 16])
        ax.imshow(data[0], cmap='gray')
        fig.tight_layout()

        # Now that you've plotted the true central points of each particle, also plot the boxes
        boxes = _twoD_image_bboxs(particles_ugraph["position_x"], particles_ugraph["position_y"], box_width, box_height, verbose)
        if verbose:
            print(f"number of boxes: {len(boxes)}")
        
        for bbox in boxes:
            corner = [bbox[0], bbox[1]] 
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[0, 1, 0], facecolor='none'
            )
            ax.add_patch(rect)
        green_patch = patches.Patch(color='green', label='Truth particles')
        ax.legend(handles=[green_patch])
    return fig, ax

def label_micrograph_picked(particles: pd.DataFrame, ugraph_index: int=0, mrc_dir: str=None,
                           box_width: int=50, box_height: int=50,
                           verbose: bool=False)->Tuple[plt.Figure, plt.Axes]:
    # get the micrograph name
    ugraph_filename = np.unique(particles["ugraph_filename"])[ugraph_index]
    print(f"plotted index {ugraph_index}; micrograph: {ugraph_filename}")
    ugraph_path = os.path.join(mrc_dir, ugraph_filename)
    particles_ugraph = particles.groupby("ugraph_filename").get_group(ugraph_filename)
        
    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(ugraph_path) as mrc:
        data = mrc.data

        fig, ax = plt.subplots(figsize=[16, 16])
        ax.imshow(data[0], cmap='gray')
        fig.tight_layout()

        # Now that you've plotted the true central points of each particle, also plot the boxes
        boxes = _twoD_image_bboxs(particles_ugraph["position_x"], particles_ugraph["position_y"], box_width, box_height, verbose)
        if verbose:
            print(f"number of boxes: {len(boxes)}")
        
        for bbox in boxes:
            corner = [bbox[0], bbox[1]] 
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[1, 0, 0], facecolor='none'
            )
            ax.add_patch(rect)
        red_patch = patches.Patch(color='red', label='Picked particles')
        ax.legend(handles=[red_patch])
    return fig, ax

def label_micrograph_truth_and_picked(picked_particles: pd.DataFrame, truth_particles: pd.DataFrame, ugraph_index: int=0, mrc_dir: str=None,
                           box_width: int=50, box_height: int=50,
                           verbose: bool=False)->Tuple[plt.Figure, plt.Axes]:
    # get the micrograph name
    ugraph_filename = np.unique(truth_particles["ugraph_filename"])[ugraph_index]
    print(f"plotted index {ugraph_index}; micrograph: {ugraph_filename}")
    ugraph_path = os.path.join(mrc_dir, ugraph_filename)
    truth_particles_ugraph = truth_particles.groupby("ugraph_filename").get_group(ugraph_filename)
    picked_particles_ugraph = picked_particles.groupby("ugraph_filename").get_group(ugraph_filename)
        
    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(ugraph_path) as mrc:
        data = mrc.data

        fig, ax = plt.subplots(figsize=[16, 16])
        ax.imshow(data[0], cmap='gray')
        fig.tight_layout()

        # Now that you've plotted the true central points of each particle, also plot the boxes
        boxes = _twoD_image_bboxs(picked_particles_ugraph["position_x"], picked_particles_ugraph["position_y"], box_width, box_height, verbose)
        if verbose:
            print(f"number of boxes: {len(boxes)}")
        
        for bbox in boxes:
            corner = [bbox[0], bbox[1]] 
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[1, 0, 0], facecolor='none'
            )
            ax.add_patch(rect)
        red_patch = patches.Patch(color='red', label='Picked particles')
        
        boxes = _twoD_image_bboxs(truth_particles_ugraph["position_x"], truth_particles_ugraph["position_y"], box_width, box_height, verbose)
        if verbose:
            print(f"number of boxes: {len(boxes)}")
        
        for bbox in boxes:
            corner = [bbox[0], bbox[1]] 
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[0, 1, 0], facecolor='none'
            )
            ax.add_patch(rect)
        green_patch = patches.Patch(color='green', label='Truth particles')
        ax.legend(handles=[red_patch, green_patch])
    return fig, ax

def plot_precision(picked_particles: pd.DataFrame, truth_particles: pd.DataFrame):
    ## precision is calculated as follows:
    ## precision = TP / (TP + FP)
    ## where TP is the number of true positives, which is stored in the picked_particles dataframe
    ## and FP is the number of false positives, which can be extracted from the truth_particles dataframe by looking at the number of truth particles
    ## that have 0 multiplicity

    # get the number of true positives

    return



### main
def main(args):
    ## this analysis tool makes plots of the picked and ground-truth particles in a number of micrographs. It then 
    ## makes quantitative comparisons between the two.

    analysis = particle_picking(args.meta_file, args.config_dir, args.particle_diameter, verbose=args.verbose)
    df_picked = pd.DataFrame(analysis.results_picking) # data frame containing the picked particles
    df_truth = pd.DataFrame(analysis.results_truth) # data frame containing the ground-truth particles

    for plot_type in args.plot_types:
        
        if plot_type == "label_truth": # plot the ground-truth particles
            for ugraph_index, ugraph_filename in enumerate(np.unique(df_truth["ugraph_filename"])[:args.num_ugraphs]):
                print(f"plotting micrograph {ugraph_filename}")
                print("plotting ground-truth particles...")
                fig, ax = label_micrograph_truth(df_truth, ugraph_index, args.mrc_dir, box_width=args.box_width, box_height=args.box_height, verbose=args.verbose)
            
        if plot_type == "label_picked": # plot the picked particles
            for ugraph_index, ugraph_filename in enumerate(np.unique(df_picked["ugraph_filename"])[:args.num_ugraphs]):
                print(f"plotting micrograph {ugraph_filename}")
                print("plotting picked particles...")
                fig, ax = label_micrograph_picked(df_picked, ugraph_index, args.mrc_dir, box_width=args.box_width, box_height=args.box_height, verbose=args.verbose)
                
        if plot_type == "label_truth_and_picked":
            for ugraph_index, ugraph_filename in enumerate(np.unique(df_picked["ugraph_filename"])[:args.num_ugraphs]):
                print(f"plotting micrograph {ugraph_filename}")
                print("plotting picked particles...")
                fig, ax = label_micrograph_truth_and_picked(df_picked, df_truth, ugraph_index, args.mrc_dir, box_width=args.box_width, box_height=args.box_height, verbose=args.verbose)


    # ## now we can make the plots
    # # plots per micrograph
    # if args.plot_per_ugraph:
    #     print("plotting per micrograph...")
    #     for ugraph_filename in list(gt_particles.keys()):
    #         print(f"plotting micrograph {ugraph_filename}")
    #         print("plotting ground-truth particles...")
    #         ugraph_path = os.path.join(args.mrc_dir, ugraph_filename)
    #         label_micrograph_truth(ugraph_path, args.plot_dir, gt_particles[ugraph_filename], args.box_width, args.box_height, args.plot_truth_centres, args.verbose)

    #         print("plotting picked particles...")
    #         label_micrograph_picked(ugraph_path, args.plot_dir, particles[ugraph_filename], args.box_width, args.box_height, args.verbose)


    #         print("plotting ground-truth particles and picked particles...")
    #         label_micrograph_truth_and_picked(ugraph_path, args.plot_dir, particles[ugraph_filename], gt_particles[ugraph_filename], args.box_width, args.box_height, args.verbose)

    #         print("plotting ground-truth particles and picked particles (unpicked truth only)...")
    #         label_micrograph_unpicked_truth_and_picked(ugraph_path, args.plot_dir, particles[ugraph_filename], gt_particles[ugraph_filename], args.box_width, args.box_height, args.verbose)
                 
    # # plots with collective statistics
    # if args.plot_collective_boundary:
    #     print("plotting collective statistics (boundary investigation)...")
    #     boundary_investigation(gt_particles, particles, args.plot_dir, args.pixel_bin_width, args.verbose)    

    # if args.plot_collective_overlap:
    #     print("plotting collective statistics (overlap investigation)...")
    #     overlap_investigation(gt_particles, particles, args.plot_dir, args.verbose)

    # if args.plot_collective_depth:
    #     print("plotting collective statistics (depth investigation)...")
    #     depth_investigation(gt_particles, args.plot_dir, args.pixel_bin_width, args.verbose)

    # # calculate picking efficiency per micrograph and insert into truth and picked particle dicts
    # # this is a super simple alg which does not check 
    # # for double-counted matches. These are mitigated through
    # # the use of args.particle_diameter, which is the expected max particle diameter
    # # particles, picked_particles = calculate_picking_efficiency(args, particles, picked_particles)

    # # create a truth particle summary yaml from truth particles dict for future analysis/plotting
    # # if args.truth_particles_yaml_filename:
    # #     save_truth_particles(args, particles)
    # # # create a picked particle summary yaml from picked particles dict for future analysis/plotting
    # # if args.picked_particles_yaml_filename:
    # #     save_picked_particles(args, picked_particles)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print('{}, {}'.format(arg, getattr(args, arg)))
    main(args)