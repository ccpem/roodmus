"""script to analyse the result of particle picking, compared to the ground-truth particle positions."""

import os
from typing import Tuple, Optional

import yaml
from tqdm import tqdm
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from matplotlib import patches
from scipy.spatial import cKDTree as ckdtree

from roodmus.analysis.utils import IO


### arguments
def add_arguments(parser):
    parser.add_argument("--mrc-dir", help="directory with .mrc files and .yaml config files", type=str)
    parser.add_argument("--meta-file", help="particle metadata file. Can be .star (RELION) or .cs (CryoSPARC)", type=str)
    parser.add_argument("-N", "--num-ugraphs", help="number of micrographs to consider in analyses. Default -1 (all)", type=int, default=-1)
    parser.add_argument("--box-width", help="Full width of overlay boxes on images", type=float, default=50., required=False)
    parser.add_argument("--box-height", help="Full height of overlay boxes on images", type=float, default=50., required=False)
    parser.add_argument("--particle-diameter", help="Expected maximum particle diameter. Used to limit search radius for matching picked particles to truth particles", type=float, default=250., required=False)
    parser.add_argument("--pixel-bin-width", help="Number of image pixels to bin together for histograms", type=int, default=100, required=False)
    parser.add_argument("--plot-truth-centres", help="Plot the ground-truth particle centres", action="store_true")
    parser.add_argument("--plot-per-ugraph", help="Plot the results per micrograph", action="store_true")
    parser.add_argument("--plot-collective-boundary", help="Plot the collective boundary investigation", action="store_true")
    parser.add_argument("--plot-collective-depth", help="Plot the collective depth investigation", action="store_true")
    parser.add_argument("--plot-collective-overlap", help="Plot the collective overlap investigation", action="store_true")
    parser.add_argument("--plot-dir", help="output file name", type=str)
    parser.add_argument("--verbose", help="increase output verbosity", action="store_true")
    return parser

def get_name():
    return "analyse_picking"

def load_metadata(meta_file: str, verbose: bool=False)->dict:
    if meta_file.endswith(".star"):
        metadata = IO.load_star(meta_file)
        file_type = "star"
    elif meta_file.endswith(".cs"):
        metadata = IO.load_cs(meta_file)
        file_type = "cs"
    else:
        raise ValueError(f"unknown metadata file type: {meta_file}")
    
    if verbose:
        print(f"loaded metadata from {meta_file}. determined file type: {file_type}")
    return metadata, file_type

def extract_from_metadata(metadata: object, file_type: str, verbose: bool=False)->dict:
    if file_type == "cs":
        ugraph_paths = IO.get_ugraph_cs(metadata) # a list of all microraps in the metadata file
        positions = IO.get_positions_cs(metadata) # an array of all the defocus values in the metadata file
    elif file_type == "star":
        ugraph_paths = IO.get_ugraph_star(metadata) # a list of all microraps in the metadata file
        positions = IO.get_positions_star(metadata) # an array of all the defocus values in the metadata file
    else:
        raise ValueError(f"unknown metadata file type: {file_type}")
    
    if verbose:
        print(f"extracted ugraph paths and positions from metadata file")
        print(positions.shape)
        print(len(ugraph_paths))
        
    particles = {}
    for key in np.unique(ugraph_paths):
        particles[key] = {}
        particles[key]["ugraph_filename"] = key
        particles[key]["pos_x"] = [r for r, val in zip(positions[:, 0], ugraph_paths) if val == key]
        particles[key]["pos_y"] = [r for r, val in zip(positions[:, 1], ugraph_paths) if val == key]
        
    return particles

def extract_from_config(config: object, verbose: bool=False)->dict: 
    pixel_size = config["microscope"]["detector"]["pixel_size"]
    image_x_pixels = config["microscope"]["detector"]["nx"]
    image_y_pixels = config["microscope"]["detector"]["ny"]
    image_z_pixels = config["sample"]["box"][-1] / pixel_size # not used here
    positions = []
    for molecules in config["sample"]["molecules"]["local"]:
        f = molecules["filename"] # not used here
        for instance in molecules["instances"]:
            orientation = instance["orientation"] # not used here 
            position = instance["position"]
            positions.append(position)

    positions = np.array(positions) / pixel_size # convert to pixels
    if verbose:
        print(f"extracted positions from config file. shape: {positions.shape}")
        
    gt_particles = {}
    gt_particles["pos_x"] = positions[:, 0]
    gt_particles["pos_y"] = positions[:, 1]
    gt_particles["pos_z"] = positions[:, 2]
    gt_particles["pixel_size"] = [pixel_size]*len(positions)
    gt_particles["image_x_pixels"] = [image_x_pixels]*len(positions)
    gt_particles["image_y_pixels"] = [image_y_pixels]*len(positions)
    gt_particles["image_z_pixels"] = [image_z_pixels]*len(positions)
        
    return gt_particles

def twoD_image_bboxs(particles_x: np.array, particles_y: np.array, box_width: float, box_height: float, verbose: bool=False)->list[list[float]]:

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

def associate_truth_and_picked_particles(truth_particles: dict, picked_particles: dict, particle_diameter: float, verbose: bool=False)->Tuple[dict, dict]:
    # set a single large radius to get sparse distance matrix using
    image_x_pixels = truth_particles["image_x_pixels"][0] # assuming all the same
    image_y_pixels = truth_particles["image_y_pixels"][0]
    r = np.sqrt(np.power(float(image_x_pixels),2)+np.power(float(image_y_pixels),2))
    # r = args.particle_diameter

    # create a list to hold masks to apply to select unpicked particles for each ugraph
    # loop over the micrographs and for each find the unmatched truth particles
    # create a np array to ckdtree for number of neighbours as function of distance
    truth_centres = grab_xy_array(truth_particles, "pos_x", "pos_y")

    # do the same for the picked particles
    picked_centres = grab_xy_array(picked_particles, "pos_x", "pos_y")

    # get the sparse_distance_matrix
    sdm = ckdtree(picked_centres).sparse_distance_matrix(ckdtree(truth_centres), r).toarray()
    sdm[sdm<np.finfo(float).eps] = np.nan
    if verbose: 
        print('Shape of sdm: {}'.format(sdm.shape))

    # find the minimum value along axis 0 (1 per picked particle)
    # and keep track of the index of the truth particle
    closest_truth_index = []
    picked_particles["truth_match_index"] = []
    truth_particles["picked_match_index"] = [np.nan] * len(truth_particles["pos_x"])
    for j, picked_particle in enumerate(sdm):
        truth_particle_index = int(np.nanargmin(picked_particle))
        # check if closest truth particle is within particle diameter of picked particle
        if picked_particle[truth_particle_index]>particle_diameter:
            closest_truth_index.append(np.nan)
            picked_particles["truth_match_index"].append(np.nan)
        # if it is, consider picking successful and allow the picked and truth particle to be associated with each other
        else:
            closest_truth_index.append(truth_particle_index)
        
            # grab the truth and picked particle indexes which are associated and add to particle dicts so you don't have to recalc this
            picked_particles["truth_match_index"].append(truth_particle_index)
        
            # any truth particles without assn picked particle have np.nan entry instead by default
            # filter them out using mask below
            truth_particles["picked_match_index"][truth_particle_index] = int(j)

    # check whether any truth particles had multiple picked particles mapped to them
    non_unique_count = len(np.unique(np.array(closest_truth_index, dtype=float)[~np.isnan(closest_truth_index)])[np.unique(np.array(closest_truth_index, dtype=float)[~np.isnan(closest_truth_index)], return_counts=True)[1]!=1])
    print('There are {} non-unique picked particles!'.format(non_unique_count))
    if non_unique_count>0:
        print('This may cause problems with overwritten assns in truth particles dict!')

    # select the truth particles which were not closest to a picked particle and create a mask for easy selection
    mask = np.ones(len(truth_centres), dtype=bool)
    # get rid of the nans (where a picked particle didn't match a truth particle)
    mask[np.array(closest_truth_index, dtype=float)[~np.isnan(closest_truth_index)].astype(int)] = False
    # assign unpicked particles mask to the truth particles based on ugraph index
    truth_particles["unpicked_mask"] = mask.tolist()

    return truth_particles, picked_particles

def grab_xy_array(particles: dict, xlabel: str, ylabel: str)->np.ndarray:
    # create a np array to ckdtree for number of neighbours as function of distance
    particle_centres = np.empty((int(len(particles[xlabel])) + int(len(particles[ylabel]))), dtype=float)
    particle_centres[0::2] = particles[xlabel]
    particle_centres[1::2] = particles[ylabel]
    particle_centres = particle_centres.reshape(int(len(particle_centres)/2), 2)
    return particle_centres

### plotting functions
def label_micrograph_truth(ugraph_path: str, plot_dir: str, particles: dict,  
                           box_width: int=50, box_height: int=50,
                           plot_truth_centres: bool=False, verbose: bool=False)->None:

    ugraph_index = os.path.basename(ugraph_path).split(".")[0]
        
    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(ugraph_path) as mrc:
        data = mrc.data

        if plot_truth_centres:
            # grab and plot the lists of x and y coords of the truth particles
            plt.figure(1, figsize=[16,16])
            plt.scatter(particles["pos_x"], particles["pos_y"])
            plt.imshow(data[0], cmap='gray')
            plt.tight_layout()
            plt.savefig(os.path.join(plot_dir, f'{ugraph_index}_plot_truth_centres.png'), dpi=300)
            plt.savefig(os.path.join(plot_dir, f'{ugraph_index}_plot_truth_centres.pdf'))
            plt.clf()
            
            if verbose:
                print(f"number of particles plotted : {len(particles['pos_x'])}")

        plt.figure(1, figsize=[16,16])
        plt.imshow(data[0], cmap='gray')
        plt.tight_layout()

        # Now that you've plotted the true central points of each particle, also plot the boxes
        boxes = twoD_image_bboxs(particles["pos_x"], particles["pos_y"], box_width, box_height, verbose)
        if verbose:
            print(f"number of boxes: {len(boxes)}")
        
        ax = plt.gca()
        for bbox in boxes:
            corner = [bbox[0], bbox[1]] 
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[0, 1, 0], facecolor='none'
            )
            ax.add_patch(rect)
        green_patch = patches.Patch(color='green', label='Truth  particles')
        plt.legend(handles=[green_patch])
        plt.savefig(os.path.join(plot_dir, f'{ugraph_index}_plot_truth_boxes.png'))
        plt.savefig(os.path.join(plot_dir, f'{ugraph_index}_plot_truth_boxes.pdf'), dpi=300)
        plt.clf()
        return

def label_micrograph_picked(ugraph_path: str, plot_dir: str, particles: dict,  
                           box_width: int=50, box_height: int=50,
                           verbose: bool=False)->None:

    ugraph_index = os.path.basename(ugraph_path).split(".")[0]
        
    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(ugraph_path) as mrc:
        data = mrc.data

        plt.figure(1, figsize=[16,16])
        plt.imshow(data[0], cmap='gray')
        plt.tight_layout()

        # Now that you've plotted the true central points of each particle, also plot the boxes
        boxes = twoD_image_bboxs(particles["pos_x"], particles["pos_y"], box_width, box_height, verbose)
        if verbose:
            print(f"number of boxes: {len(boxes)}")
        
        ax = plt.gca()
        for bbox in boxes:
            corner = [bbox[0], bbox[1]] 
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[1, 0, 0], facecolor='none'
            )
            ax.add_patch(rect)
        red_patch = patches.Patch(color='red', label='Truth  particles')
        plt.legend(handles=[red_patch])
        plt.savefig(os.path.join(plot_dir, f'{ugraph_index}_plot_picked_boxes.png'))
        plt.savefig(os.path.join(plot_dir, f'{ugraph_index}_plot_picked_boxes.pdf'), dpi=300)
        plt.clf()
        return

def label_micrograph_truth_and_picked(ugraph_path: str, plot_dir: str, picked_particles: dict, truth_particles: dict,  
                           box_width: int=50, box_height: int=50,
                           verbose: bool=False)->None:

    ugraph_index = os.path.basename(ugraph_path).split(".")[0]
        
    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(ugraph_path) as mrc:
        data = mrc.data

        plt.figure(1, figsize=[16,16])
        plt.imshow(data[0], cmap='gray')
        plt.tight_layout()

        # Now that you've plotted the true central points of each particle, also plot the boxes
        picked_boxes = twoD_image_bboxs(picked_particles["pos_x"], picked_particles["pos_y"], box_width, box_height, verbose)
        if verbose:
            print(f"number of boxes: {len(picked_boxes)}")
        
        ax = plt.gca()
        for bbox in picked_boxes:
            corner = [bbox[0], bbox[1]] 
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[1, 0, 0], facecolor='none'
            )
            ax.add_patch(rect)
        red_patch = patches.Patch(color='red', label='Truth  particles')
        
        truth_boxes = twoD_image_bboxs(truth_particles["pos_x"], truth_particles["pos_y"], box_width, box_height, verbose)
        if verbose:
            print(f"number of boxes: {len(truth_boxes)}")
        
        ax = plt.gca()
        for bbox in truth_boxes:
            corner = [bbox[0], bbox[1]] 
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[0, 1, 0], facecolor='none'
            )
            ax.add_patch(rect)
        green_patch = patches.Patch(color='green', label='Truth  particles')
        
        
        plt.legend(handles=[green_patch, red_patch])
        plt.savefig(os.path.join(plot_dir, f'{ugraph_index}_plot_truth_and_picked_boxes.png'))
        plt.savefig(os.path.join(plot_dir, f'{ugraph_index}_plot_truth_and_picked_boxes.pdf'), dpi=300)
        plt.clf()
        return

def label_micrograph_unpicked_truth_and_picked(ugraph_path: str, plot_dir: str, picked_particles: dict, truth_particles: dict,  
                           box_width: int=50, box_height: int=50,
                           verbose: bool=False)->None:
    ugraph_index = os.path.basename(ugraph_path).split(".")[0]
        
    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(ugraph_path) as mrc:
        data = mrc.data

        # Get the truth boxes        
        boxes = twoD_image_bboxs(np.array(truth_particles["pos_x"], dtype=float)[truth_particles["unpicked_mask"]],
                                 np.array(truth_particles["pos_y"], dtype=float)[truth_particles["unpicked_mask"]], box_width, box_height, verbose)
        
        # Get the picked boxes
        picked_boxes = twoD_image_bboxs(picked_particles["pos_x"], picked_particles["pos_y"], box_width, box_height, verbose)

        plt.figure(1, figsize=[16,16])
        plt.imshow(data[0], cmap='gray')
        
        ax = plt.gca()
        for bbox in boxes:
            corner = [bbox[0], bbox[1]] 
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[0, 0, 1], facecolor='none'
            )
            ax.add_patch(rect)
        for bbox in picked_boxes:
            corner = [bbox[0], bbox[1]] 
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[1, 0, 0], facecolor='none'
            )
            ax.add_patch(rect)
        blue_patch = patches.Patch(color='blue', label='Unpicked truth particles')
        red_patch = patches.Patch(color='red', label='Picked particles')
        plt.legend(handles=[blue_patch, red_patch])
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, f'{ugraph_index}_plot_unpicked_truth_and_picked_boxes.png'))
        plt.savefig(os.path.join(plot_dir, f'{ugraph_index}_plot_unpicked_truth_and_picked_boxes.pdf'), dpi=300)
        plt.clf()
    return

def boundary_investigation(truth_particles: dict, picked_particles:dict, plot_dir: str, pixel_bin_width: int=4, verbose: bool=False)->None:
    """Make plots of numbers of particles as a function of x, y and z.
    Also plot the expected number with 3 std devs

    Args:
        args: Parsed command line arguments
        particles (dict): Truth particles dictionary
        picked_particles (dict): Picked particles dictionary
    """
    particles_per_ugraph = len(truth_particles[list(truth_particles.keys())[0]]["pos_x"])
    image_x_pixels = truth_particles[list(truth_particles.keys())[0]]["image_x_pixels"][0] # assuming all the same
    image_y_pixels = truth_particles[list(truth_particles.keys())[0]]["image_y_pixels"][0]
    image_z_pixels = truth_particles[list(truth_particles.keys())[0]]["image_z_pixels"][0]
    hist_bins_x = np.arange(0., float(image_x_pixels)+pixel_bin_width, pixel_bin_width)
    hist_bins_y = np.arange(0., float(image_y_pixels)+pixel_bin_width, pixel_bin_width)
    hist_bins_z = np.arange(0., float(image_z_pixels)+pixel_bin_width, pixel_bin_width)

    if verbose:
        print('Hist x bins: {}'.format(hist_bins_x))
        print('Hist y bins: {}'.format(hist_bins_y))
        print('Hist z bins: {}'.format(hist_bins_z))

    expected_per_bin_x = (pixel_bin_width/float(image_x_pixels))*particles_per_ugraph*len(truth_particles.keys())
    expected_per_bin_y = (pixel_bin_width/float(image_y_pixels))*particles_per_ugraph*len(truth_particles.keys())
    expected_per_bin_z = (pixel_bin_width/float(image_z_pixels))*particles_per_ugraph*len(truth_particles.keys())
    
    if verbose:
        print(f"Expected number of particles per bin in x: {expected_per_bin_x}")
        print(f"Expected number of particles per bin in y: {expected_per_bin_y}")
        print(f"Expected number of particles per bin in z: {expected_per_bin_z}")
    
    # truth particles
    # grab all the x centres in one array
    truth_x_centres = []
    # grab all the y centres in one array
    truth_y_centres = []
    # grab all the z centres in one array
    truth_z_centres = []
    # picked particles
    # grab all the x centres in one array
    picked_x_centres = []
    # grab all the y centres in one array
    picked_y_centres = []
    
    for ugraph_filename in truth_particles.keys():
        truth_x_centres.extend(truth_particles[ugraph_filename]["pos_x"])
        truth_y_centres.extend(truth_particles[ugraph_filename]["pos_y"])
        truth_z_centres.extend(truth_particles[ugraph_filename]["pos_z"])
        picked_x_centres.extend(picked_particles[ugraph_filename]["pos_x"])
        picked_y_centres.extend(picked_particles[ugraph_filename]["pos_y"])
        
    if verbose:
        print(f"Number of truth particles: {len(truth_x_centres)}")
        print(f"Number of picked particles: {len(picked_x_centres)}")

    # plot particles as function of x for both
    # blue is truth
    # red is picked
    plt.figure(1, figsize=[16,16])
    # fig, ax = plt.subplots(figsize=[16,16])
    # plot the truth
    plt.hist(truth_x_centres, bins=hist_bins_x, histtype='step', label='truth', color='blue')
    # plot the picked
    plt.hist(picked_x_centres, bins=hist_bins_x, histtype='step', label='picked', color='red')
    # plt.xticks(hist_bins_x)
    # plot the expected
    plt.hlines([expected_per_bin_x], 0., image_x_pixels, colors=['black'], linestyles=['dashed'], label='expected')

    plt.grid(which='both')
    plt.xlabel('x coordinate (angstroms)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'particles_x.png'), dpi=300)
    plt.savefig(os.path.join(plot_dir, 'particles_x.pdf'))
    plt.clf()

    # plot particles as function of y for both
    # blue is truth
    # red is picked
    plt.figure(1, figsize=[16,16])
    # plot the truth
    plt.hist(truth_y_centres, bins=hist_bins_y, histtype='step', label='truth', color='blue')
    # plot the picked
    plt.hist(picked_y_centres, bins=hist_bins_y, histtype='step', label='picked', color='red')
    # plot the expected
    plt.hlines([expected_per_bin_y], 0., image_y_pixels, colors=['black'], linestyles=['dashed'], label='expected')

    plt.grid(which='both')
    plt.xlabel('y coordinate (angstroms)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'particles_y.png'), dpi=300)
    plt.savefig(os.path.join(plot_dir, 'particles_y.pdf'))
    plt.clf()

    # plot particles as function of z for truth only
    # blue is truth
    plt.figure(1, figsize=[16,16])
    # plot the truth
    plt.hist(truth_z_centres, bins=hist_bins_z, histtype='step', label='truth', color='blue')
    # plot the expected
    plt.hlines([expected_per_bin_z], 0., image_z_pixels, colors=['black'], linestyles=['dashed'], label='expected')

    plt.grid(which='both')
    plt.xlabel('z coordinate (angstroms)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'particles_z.png'), dpi=300)
    plt.savefig(os.path.join(plot_dir, 'particles_z.pdf'))
    plt.clf()
    return 

def overlap_investigation(truth_particles: dict, picked_particles:dict, plot_dir: str, verbose: bool=False)->None:
    """Load in all the truth and picked particles, then work out the number of particles
    which 'overlap' for a range of particle diameters. Particles whose centre+radius 
    reach the centre of another particle count as overlapped.

    Plot:
    1) using only truth particles, count the number of particle overlaps 
    as a function of particle diameter 
        - should show proportion of particles which overlap for given avg particle diameter
        - grab particle diameter info and check what proportion 'should' overlap and therefore
        might expect to be lost
    2) using truth and picked particles, count the number of particle overlaps 
    as a function of particle diameter 
        - "" ""

    Args:
        args (_type_): Parsed command line arguments
        particles (dict): Truth particles dictionary
        picked_particles (dict): Picked particles dictionary
    """
    
    # create list of particle diameters to check
    diameters_to_check = np.arange(10., 401., 10.)

    # check closeness of truth particles
    # create list of proportion with no overlap, with 1 (list) entry per particle radius
    # so will be a list[np.array(one entry per ugraph)]
    # ie: all_ugraph_neighbours[a,b] gives access to diameter a, and micrograph b
    all_ugraph_neighbours = [[]]*len(diameters_to_check)
    all_ugraph_picked_neighbours = [[]]*len(diameters_to_check)
    # init the micrograph lists to correct size with zeros
    for i in range(len(diameters_to_check)):
        all_ugraph_neighbours[i] = np.zeros(len(truth_particles), dtype=int)
        all_ugraph_picked_neighbours[i] = np.zeros(len(picked_particles), dtype=int)

    # determine for each diameter, the number of truth particles which overlap in a given ugraph
    # loop over each micrograph
    for i, ugraph_index in enumerate(list(truth_particles.keys())):
        # create a np array to ckdtree for number of neighbours as function of distance
        truth_centres = grab_xy_array(truth_particles[ugraph_index], "pos_x", "pos_y")

        # do the same for the picked particles
        picked_centres = grab_xy_array(picked_particles[ugraph_index], "pos_x", "pos_y")

        # get a vector of the number of neighbours, one entry per diameter
        neighbours = (ckdtree(truth_centres).count_neighbors(ckdtree(truth_centres), r=np.array(diameters_to_check)/2.))
        # account for the fact each particle matches itself, which would be 200 matches by default
        neighbours = neighbours - len(truth_particles[ugraph_index]["pos_x"])
        # account for fact each particle is matched 2 times! Once from particle A->B and once from B->A
        neighbours = neighbours/2

        # do the same but this time between the picked particles and the truth particles
        picked_neighbours = (ckdtree(picked_centres).count_neighbors(ckdtree(picked_centres), r=np.array(diameters_to_check)/2.))
        # account for fact each particle is matched 2 times! Once from particle A->B and once from B->A
        picked_neighbours = picked_neighbours/2
        
        # insert the number of neighbours found for each diameter in this ugraph to the list
        for j, (val, picked_val) in enumerate(zip(neighbours, picked_neighbours)):
            all_ugraph_neighbours[j][i] = val
            all_ugraph_picked_neighbours[j][i] = picked_val

            if verbose:
                print(f"ugraph {ugraph_index} has {val} neighbours for diameter {diameters_to_check[j]}")
                print(f"ugraph {ugraph_index} has {picked_val} picked neighbours for diameter {diameters_to_check[j]}")

    # avg over all the ugraph entries, to get an avg per ugraph only for a given particle diameter
    all_ugraph_neighbours = [np.average(ugraph_neighbours) for ugraph_neighbours in all_ugraph_neighbours]
    all_ugraph_picked_neighbours = [np.average(ugraph_neighbours) for ugraph_neighbours in all_ugraph_picked_neighbours]

    # plt.plot number of particle overlaps as a function of particle diameter
    plt.figure(1, figsize=[16,16])
    # plot the truth
    plt.plot(diameters_to_check, all_ugraph_neighbours, label='# truth particle overlaps', color='blue', marker='o')
    # plt.plot number of particle overlaps as a function of particle diameter
    # between the picked particles and the truth particles on same plot
    plt.plot(diameters_to_check, all_ugraph_picked_neighbours, label='# picked overlaps with truth', color='red', marker='x')

    plt.grid(which='both')
    plt.xlabel('Particle diameter')
    plt.ylabel('# Overlaps with a truth particle')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'particle_overlaps.png'), dpi=300)
    plt.savefig(os.path.join(plot_dir, 'particle_overlaps.pdf'))
    plt.clf()

    return

def depth_investigation(truth_particles: dict, plot_dir: str, pixel_bin_width: int=4, verbose: bool=False)->None:
    """ Take matched picked particles and turth particles and make:
    Histogram of the depth (z) of non-matched truth particles on same plot as 
    depth (z) of all truth particles, both normalized to be a pdf

    Args:
        args (_type_): Parsed command line arguments
        particles (dict): Truth particles dictionary
        picked_particles (dict): Picked particles dictionary
    """

    image_z_pixels = truth_particles[list(truth_particles.keys())[0]]["image_z_pixels"][0]

    # grab a single list of all the truth z position and those for unmatched particles
    all_z = []
    all_unpicked_z = []
    for ugraph_index in truth_particles.keys():
        all_z.extend(truth_particles[ugraph_index]["pos_z"])
        all_unpicked_z.extend(np.array(truth_particles[ugraph_index]["pos_z"], dtype=float)[truth_particles[ugraph_index]["unpicked_mask"]].tolist())

    if verbose:
        print(f"number of truth particles: {len(all_z)}")
        print(f"number of unpicked truth particles: {len(all_unpicked_z)}")

    # want to plot z distribution of unpicked particles against that of all truth particles
    hist_bins_z = np.arange(0., float(image_z_pixels)+pixel_bin_width, pixel_bin_width)
    plt.figure(1, figsize=[16,16])
    # plot all truth particles
    plt.hist(all_z, bins=hist_bins_z, histtype='step', label='truth', color='blue', density=True)
    # plot the unpicked truth particles
    plt.hist(all_unpicked_z, bins=hist_bins_z, histtype='step', label='unpicked truth', color='red', density=True)
    plt.grid(which='both')
    plt.xlabel('Particle z (angstroms)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'unpicked_z.png'), dpi=300)
    plt.savefig(os.path.join(plot_dir, 'unpicked_z.pdf'))
    plt.clf()

    # could extend to also plot:
    # - x, y distribution of unpicked particles against all particles (both normalised)
    # - 2d x,y distribution of unpicked particles
    return

### main
def main(args):
    ## this analysis tool makes plots of the picked and ground-truth particles in a number of micrographs. It then 
    ## makes quantitative comparisons between the two.
    
    # first get the picked particle positions
    metadata, file_type = load_metadata(args.meta_file, args.verbose)
    print("metadata loaded. extracting particle positions...")
    particles = extract_from_metadata(metadata, file_type, args.verbose)
    print(f"extracted postions from {len(particles.keys())} micrographs.")
    num_particles = 0
    for key in particles.keys():
        num_particles += len(particles[key]["pos_x"])
    print(f"total number of particles: {num_particles}")
    
    # next, get the ground-truth particle positions
    progressbar = tqdm(total=args.num_ugraphs, desc="loading ground-truth particle positions")
    gt_particles = {} # stores all particles
    for ugraph_filename in list(particles.keys())[:args.num_ugraphs]:
        config = IO.load_config(os.path.join(args.mrc_dir, str(ugraph_filename).replace(".mrc", ".yaml")))
        gt_particles_ugraph = extract_from_config(config, args.verbose)
        gt_particles[ugraph_filename] = gt_particles_ugraph
        gt_particles[ugraph_filename]["ugraph_filename"] = gt_particles_ugraph
        _ = progressbar.update(1)
    progressbar.close()
    print(f"extracted ground-truth postions from {len(gt_particles.keys())} micrographs.")
    gt_num_particles = 0
    for key in gt_particles.keys():
        gt_num_particles += len(gt_particles[key]["pos_x"])
    print(f"total number of ground-truth particles: {gt_num_particles}")

    # associate the picked particles with the ground-truth particles    
    for ugraph_filename in list(gt_particles.keys()):
        print("associating picked particles with ground-truth particles...")
        gt_particles_ugraph, particles_ugraph = associate_truth_and_picked_particles(gt_particles[ugraph_filename], particles[ugraph_filename], args.particle_diameter, args.verbose)
        gt_particles[ugraph_filename] = gt_particles_ugraph
        particles[ugraph_filename] = particles_ugraph

    ## now we can make the plots
    # plots per micrograph
    if args.plot_per_ugraph:
        print("plotting per micrograph...")
        for ugraph_filename in list(gt_particles.keys()):
            print(f"plotting micrograph {ugraph_filename}")
            print("plotting ground-truth particles...")
            ugraph_path = os.path.join(args.mrc_dir, ugraph_filename)
            label_micrograph_truth(ugraph_path, args.plot_dir, gt_particles[ugraph_filename], args.box_width, args.box_height, args.plot_truth_centres, args.verbose)

            print("plotting picked particles...")
            label_micrograph_picked(ugraph_path, args.plot_dir, particles[ugraph_filename], args.box_width, args.box_height, args.verbose)


            print("plotting ground-truth particles and picked particles...")
            label_micrograph_truth_and_picked(ugraph_path, args.plot_dir, particles[ugraph_filename], gt_particles[ugraph_filename], args.box_width, args.box_height, args.verbose)

            print("plotting ground-truth particles and picked particles (unpicked truth only)...")
            label_micrograph_unpicked_truth_and_picked(ugraph_path, args.plot_dir, particles[ugraph_filename], gt_particles[ugraph_filename], args.box_width, args.box_height, args.verbose)
                 
    # plots with collective statistics
    if args.plot_collective_boundary:
        print("plotting collective statistics (boundary investigation)...")
        boundary_investigation(gt_particles, particles, args.plot_dir, args.pixel_bin_width, args.verbose)    

    if args.plot_collective_overlap:
        print("plotting collective statistics (overlap investigation)...")
        overlap_investigation(gt_particles, particles, args.plot_dir, args.verbose)

    if args.plot_collective_depth:
        print("plotting collective statistics (depth investigation)...")
        depth_investigation(gt_particles, args.plot_dir, args.pixel_bin_width, args.verbose)

    # calculate picking efficiency per micrograph and insert into truth and picked particle dicts
    # this is a super simple alg which does not check 
    # for double-counted matches. These are mitigated through
    # the use of args.particle_diameter, which is the expected max particle diameter
    # particles, picked_particles = calculate_picking_efficiency(args, particles, picked_particles)

    # create a truth particle summary yaml from truth particles dict for future analysis/plotting
    # if args.truth_particles_yaml_filename:
    #     save_truth_particles(args, particles)
    # # create a picked particle summary yaml from picked particles dict for future analysis/plotting
    # if args.picked_particles_yaml_filename:
    #     save_picked_particles(args, picked_particles)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser = add_arguments(parser)
    args = parser.parse_args()
    if args.verbose:
        for arg in vars(args):
            print('{}, {}'.format(arg, getattr(args, arg)))
    main(args)