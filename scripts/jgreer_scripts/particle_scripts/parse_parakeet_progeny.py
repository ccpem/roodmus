"""
Adapted and extended to map picked particles to truth particles
by Joel Greer (STFC) from a .ipynb written by 
Beatriz Costa Gomes (Alan Turing Institute) - thanks Bea!
"""

import argparse
import os

from typing import Tuple, Optional
import pprint

import yaml
import pandas as pd
import numpy as np
import mrcfile
import matplotlib.pyplot as plt
from matplotlib import patches
import glob2 as glob
from gemmi import cif
from scipy.spatial import cKDTree as ckdtree

def read_from_log(args)->Tuple[dict, int]:

    # Open and read the file which includes the parakeet log info
    f = open(args.log_filepath, "r")
    lines = f.readlines()

    # Set the default parser values and init a dataframe
    flag_positions = 0
    flag_orientations = 0
    flag_end = 0

    # parse the file and fill a dict with all the info you want
    particles = {} # indexed by micrograph number (starts at args.index_start_value)
    # structure is dict containing dict[list] accessed via e.g. particles["000000"]["x"] for example]
    
    mol = 0 # using molecule index to set dict entry
    micrograph_ind = args.index_start_value - 1 # make so minimum value is 0, because incremented before used

    for line in lines:
        parse_it = line.strip().split()
        
        # increment the micrograph number
        if line.strip()=='Placing molecules:':
            micrograph_ind+=1
            particles[str(micrograph_ind).zfill(6)] = {} # fill with empty dict of particle property lists for now
            particles[str(micrograph_ind).zfill(6)]["molecule"] = []
            particles[str(micrograph_ind).zfill(6)]["x"] = []
            particles[str(micrograph_ind).zfill(6)]["y"] = []
            particles[str(micrograph_ind).zfill(6)]["z"] = []
            particles[str(micrograph_ind).zfill(6)]["phi"] = []
            particles[str(micrograph_ind).zfill(6)]["theta"] = []
            particles[str(micrograph_ind).zfill(6)]["psi"] = []
        # check if this is one of the lines to parse and if so, grab the info from it
        """
        Example outut from line.split():
        0: placing 
        1: /mnt/parakeet_storage3/ConformationSampling/DESRES-Trajectory_sarscov2-13795965-no-water/even_sampling/even_sampling_10000/conformation_000020.pdb
        2: at 
        3: (591.37,
        4: 767.31,
        5: 281.39) 
        6: with 
        7: orientation 
        8: (-0.11, 
        9: -1.12, 
        10: 1.03)
        """
        if len(parse_it)>0:
            if parse_it[0]=='placing':
                # Fill the particle dict with:
                # molecule pdb
                # position
                # orientation 
                particles[str(micrograph_ind).zfill(6)]["molecule"] = str(parse_it[1])

                particles[str(micrograph_ind).zfill(6)]["x"].append(float(parse_it[3][1:-1]))
                particles[str(micrograph_ind).zfill(6)]["y"].append(float(parse_it[4][:-1]))
                particles[str(micrograph_ind).zfill(6)]["z"].append(float(parse_it[5][:-1]))

                particles[str(micrograph_ind).zfill(6)]["phi"].append(float(parse_it[8][1:-1]))
                particles[str(micrograph_ind).zfill(6)]["theta"].append(float(parse_it[9][:-1]))
                particles[str(micrograph_ind).zfill(6)]["psi"].append(float(parse_it[10][:-1]))

    return particles, micrograph_ind-args.index_start_value


def grab_filepaths(path:str)->list[str]:
    filepaths = sorted(glob.glob(path))
    return filepaths


def grab_ugraph_index(args, filepath: str)->str:
    index = os.path.basename(filepath)[len(args.basename_prefix):-len(args.basename_suffix)]
    if args.debug:
        print('Grabbed index: {}'.format(index))
    return index


def load_picked_particles_from_starfile(args, starfile: str, block_name: Optional[str]="particles", loop_name_prefix: Optional[str]="_rln")->dict:

    if os.path.exists(starfile):
        # load the starfile via gemmi 
        gemmi_block = cif.read_file(starfile)[block_name]
        """ 
        Example loop headings
        loop_ 
        _rlnCoordinateX #1 
        _rlnCoordinateY #2 
        _rlnAutopickFigureOfMerit #3 
        _rlnClassNumber #4 
        _rlnAnglePsi #5 
        _rlnImageName #6 
        _rlnMicrographName #7 
        _rlnOpticsGroup #8 
        _rlnCtfMaxResolution #9 
        _rlnCtfFigureOfMerit #10 
        _rlnDefocusU #11 
        _rlnDefocusV #12 
        _rlnDefocusAngle #13 
        _rlnCtfBfactor #14 
        _rlnCtfScalefactor #15 
        _rlnPhaseShift #16 
        """
        # create a dict to read these picked particles into
        # make this indexed by micrograph index
        picked_particles = {}

        CoordinateX = gemmi_block.find_values(loop_name_prefix+"CoordinateX")
        CoordinateY = gemmi_block.find_values(loop_name_prefix+"CoordinateY")
        AutopickFigureOfMerit = gemmi_block.find_values(loop_name_prefix+"AutopickFigureOfMerit")
        ClassNumber = gemmi_block.find_values(loop_name_prefix+"ClassNumber")
        AnglePsi = gemmi_block.find_values(loop_name_prefix+"AnglePsi")
        ImageName = gemmi_block.find_values(loop_name_prefix+"ImageName")
        MicrographName = gemmi_block.find_values(loop_name_prefix+"MicrographName")
        OpticsGroup = gemmi_block.find_values(loop_name_prefix+"OpticsGroup")
        CtfMaxResolution = gemmi_block.find_values(loop_name_prefix+"CtfMaxResolution")
        CtfFigureOfMerit = gemmi_block.find_values(loop_name_prefix+"CtfFigureOfMerit")
        DefocusU = gemmi_block.find_values(loop_name_prefix+"DefocusU")
        DefocusV = gemmi_block.find_values(loop_name_prefix+"DefocusV")
        DefocusAngle = gemmi_block.find_values(loop_name_prefix+"DefocusAngle")
        CtfBfactor = gemmi_block.find_values(loop_name_prefix+"CtfBfactor")
        CtfScalefactor = gemmi_block.find_values(loop_name_prefix+"CtfScalefactor")
        PhaseShift = gemmi_block.find_values(loop_name_prefix+"PhaseShift")
        
        # find the unique micrographs and create a dict with the
        # micrograph indices as keys
        # convert all entries to string instead of raw_string also
        ugraph_names = []
        for i in range(len(MicrographName)):
            ugraph_names.append(MicrographName.str(i))
        ugraph_names = np.array(ugraph_names, dtype=str)
        # get the unique names and sort them
        unique_ugraphs = sorted(np.unique(ugraph_names).tolist())
        # take out only the basename and grab the index only (as string!)
        for i in range(len(unique_ugraphs)):
            unique_ugraphs[i]=os.path.basename(unique_ugraphs[i])[len(args.basename_prefix):-len(args.basename_suffix)]
            # create a dict to hold particle property lists
            picked_particles[unique_ugraphs[i]] = {}
            picked_particles[unique_ugraphs[i]]["CoordinateX"] = []
            picked_particles[unique_ugraphs[i]]["CoordinateY"] = []
            picked_particles[unique_ugraphs[i]]["AutopickFigureOfMerit"] = []
            picked_particles[unique_ugraphs[i]]["ClassNumber"] = []
            picked_particles[unique_ugraphs[i]]["AnglePsi"] = []
            picked_particles[unique_ugraphs[i]]["ImageName"] = []
            picked_particles[unique_ugraphs[i]]["MicrographName"] = []
            picked_particles[unique_ugraphs[i]]["OpticsGroup"] = []
            picked_particles[unique_ugraphs[i]]["CtfMaxResolution"] = []
            picked_particles[unique_ugraphs[i]]["CtfFigureOfMerit"] = []
            picked_particles[unique_ugraphs[i]]["DefocusU"] = []
            picked_particles[unique_ugraphs[i]]["DefocusV"] = []
            picked_particles[unique_ugraphs[i]]["DefocusAngle"] = []
            picked_particles[unique_ugraphs[i]]["CtfBfactor"] = []
            picked_particles[unique_ugraphs[i]]["CtfScalefactor"] = []
            picked_particles[unique_ugraphs[i]]["PhaseShift"] = []

        # fill the dict lists with particle dicts
        for i in range(len(CoordinateX)):
            # grab the micrograph name and decipher which dict to add this particle to
            ugraph_index = os.path.basename(MicrographName.str(i))[len(args.basename_prefix):-len(args.basename_suffix)]

            picked_particles[ugraph_index]["CoordinateX"].append(CoordinateX.str(i))
            picked_particles[ugraph_index]["CoordinateY"].append(CoordinateY.str(i))
            picked_particles[ugraph_index]["AutopickFigureOfMerit"].append(AutopickFigureOfMerit.str(i))
            picked_particles[ugraph_index]["ClassNumber"].append(ClassNumber.str(i))
            picked_particles[ugraph_index]["AnglePsi"].append(AnglePsi.str(i))
            picked_particles[ugraph_index]["ImageName"].append(ImageName.str(i))
            picked_particles[ugraph_index]["MicrographName"].append(MicrographName.str(i))
            picked_particles[ugraph_index]["OpticsGroup"].append(OpticsGroup.str(i))
            picked_particles[ugraph_index]["CtfMaxResolution"].append(CtfMaxResolution.str(i))
            picked_particles[ugraph_index]["CtfFigureOfMerit"].append(CtfFigureOfMerit.str(i))
            picked_particles[ugraph_index]["DefocusU"].append(DefocusU.str(i))
            picked_particles[ugraph_index]["DefocusV"].append(DefocusV.str(i))
            picked_particles[ugraph_index]["DefocusAngle"].append(DefocusAngle.str(i))
            picked_particles[ugraph_index]["CtfBfactor"].append(CtfBfactor.str(i))
            picked_particles[ugraph_index]["CtfScalefactor"].append(CtfScalefactor.str(i))
            picked_particles[ugraph_index]["PhaseShift"].append(PhaseShift.str(i))

        for ugraph_index in picked_particles.keys():
            picked_particles[ugraph_index]["CoordinateX"] = np.array(picked_particles[ugraph_index]["CoordinateX"], dtype=str).astype(float).tolist()
            picked_particles[ugraph_index]["CoordinateY"] =  np.array(picked_particles[ugraph_index]["CoordinateY"], dtype=str).astype(float).tolist()
            picked_particles[ugraph_index]["AutopickFigureOfMerit"] = np.array(picked_particles[ugraph_index]["AutopickFigureOfMerit"], dtype=str).astype(float).tolist()
            picked_particles[ugraph_index]["ClassNumber"] = np.array(picked_particles[ugraph_index]["ClassNumber"], dtype=str).astype(int).tolist()
            picked_particles[ugraph_index]["AnglePsi"] = np.array(picked_particles[ugraph_index]["AnglePsi"], dtype=str).astype(float).tolist()
            picked_particles[ugraph_index]["ImageName"] = np.array(picked_particles[ugraph_index]["ImageName"], dtype=str).astype(str).tolist()
            picked_particles[ugraph_index]["MicrographName"] = np.array(picked_particles[ugraph_index]["MicrographName"], dtype=str).astype(str).tolist()
            picked_particles[ugraph_index]["OpticsGroup"] = np.array(picked_particles[ugraph_index]["OpticsGroup"], dtype=str).astype(int).tolist()
            picked_particles[ugraph_index]["CtfMaxResolution"] = np.array(picked_particles[ugraph_index]["CtfMaxResolution"], dtype=str).astype(float).tolist()
            picked_particles[ugraph_index]["CtfFigureOfMerit"] = np.array(picked_particles[ugraph_index]["CtfFigureOfMerit"], dtype=str).astype(float).tolist()
            picked_particles[ugraph_index]["DefocusU"] = np.array(picked_particles[ugraph_index]["DefocusU"], dtype=str).astype(float).tolist()
            picked_particles[ugraph_index]["DefocusV"] = np.array(picked_particles[ugraph_index]["DefocusV"], dtype=str).astype(float).tolist()
            picked_particles[ugraph_index]["DefocusAngle"] = np.array(picked_particles[ugraph_index]["DefocusAngle"], dtype=str).astype(float).tolist()
            picked_particles[ugraph_index]["CtfBfactor"] = np.array(picked_particles[ugraph_index]["CtfBfactor"], dtype=str).astype(float).tolist()
            picked_particles[ugraph_index]["CtfScalefactor"] = np.array(picked_particles[ugraph_index]["CtfScalefactor"], dtype=str).astype(float).tolist()
            picked_particles[ugraph_index]["PhaseShift"] = np.array(picked_particles[ugraph_index]["PhaseShift"], dtype=str).astype(float).tolist()

    else:
        print('File:\n{}Does not exist! Exiting!'.format(starfile))
        exit(1)

    return picked_particles


def twoD_image_bboxs(args, particles_x: list, particles_y: list, box_width: float, box_height: float)->list[list[float]]:

    box_half_width = box_width/2.
    box_half_height = box_height/2.
    
    if args.debug:
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


def save_image(args, micrograph_file: str)->None:
    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(micrograph_file) as mrc:
        data = mrc.data

        # work out the index of the image so you can grab the truth particles
        ugraph_index = grab_ugraph_index(args, micrograph_file)

        plt.figure(1, figsize=[16,16])
        plt.imshow(data[0], cmap='gray')

        plt.tight_layout()
        plt.savefig('{}_image.png'.format(ugraph_index))
        plt.savefig('{}_image.pdf'.format(ugraph_index), dpi=300)
        plt.clf()

    return




def label_micrograph_truth(args, micrograph_file: str, particles: dict, plot_truth_centres: bool=False)->None:

    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(micrograph_file) as mrc:
        data = mrc.data

        # work out the index of the image so you can grab the truth particles
        ugraph_index = grab_ugraph_index(args, micrograph_file)

        if plot_truth_centres:
            # grab and plot the lists of x and y coords of the truth particles
            plt.figure(1, figsize=[16,16])
            plt.scatter(particles[ugraph_index]["x"], particles[ugraph_index]["y"])
            plt.imshow(data[0], cmap='gray')
            plt.tight_layout()
            plt.savefig('{}_plot_truth_centres.png'.format(ugraph_index), dpi=300)
            plt.savefig('{}_plot_truth_centres.pdf'.format(ugraph_index))
            plt.clf()

        # Now that you've plotted the true central points of each particle, also plot the boxes
        boxes = twoD_image_bboxs(args, particles[ugraph_index]["x"], particles[ugraph_index]["y"], args.box_width, args.box_height)
        plt.figure(1, figsize=[16,16])
        plt.imshow(data[0], cmap='gray')
        
        ax = plt.gca()
        for bbox in boxes:
            corner = [bbox[0], bbox[1]] 
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[0, 1, 0], facecolor='none'
            )
            ax.add_patch(rect)
        green_patch = patches.Patch(color='green', label='Truth particles')
        plt.legend(handles=[green_patch])
        plt.tight_layout()
        plt.savefig('{}_plot_truth_boxes.png'.format(ugraph_index))
        plt.savefig('{}_plot_truth_boxes.pdf'.format(ugraph_index), dpi=300)
        plt.clf()
    return

def label_micrograph_picked(args, micrograph_file: str, picked_particles: dict)->None:
    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(micrograph_file) as mrc:
        data = mrc.data

        # work out the index of the image so you can grab the truth particles
        ugraph_index = grab_ugraph_index(args, micrograph_file)

        # Get the picked boxes
        picked_boxes = twoD_image_bboxs(args, picked_particles[ugraph_index]["CoordinateX"], picked_particles[ugraph_index]["CoordinateY"], args.box_width, args.box_height)

        plt.figure(1, figsize=[16,16])
        plt.imshow(data[0], cmap='gray')
        
        ax = plt.gca()
        for bbox in picked_boxes:
            corner = [bbox[0], bbox[1]] 
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[1, 0, 0], facecolor='none'
            )
            ax.add_patch(rect)
        red_patch = patches.Patch(color='red', label='Picked particles')
        plt.legend(handles=[red_patch])
        plt.tight_layout()
        plt.savefig('{}_plot_picked_boxes.png'.format(ugraph_index))
        plt.savefig('{}_plot_picked_boxes.pdf'.format(ugraph_index), dpi=300)
        plt.clf()
    return

def label_micrograph_truth_and_picked(args, micrograph_file: str, particles: dict, picked_particles: dict)->None:
    """Label and save a micrograph image with truth and picked particles

    Args:
        args (_type_): _description_
        micrograph_file (str): _description_
        particles (dict): _description_
        picked_particles (dict): _description_
    """

    # Open up a mrc file to overlay the boxes with
    with mrcfile.open(micrograph_file) as mrc:
        data = mrc.data

        # work out the index of the image so you can grab the truth particles
        ugraph_index = grab_ugraph_index(args, micrograph_file)

        # Get the truth boxes
        boxes = twoD_image_bboxs(args, particles[ugraph_index]["x"], particles[ugraph_index]["y"], args.box_width, args.box_height)
        
        # Get the picked boxes
        picked_boxes = twoD_image_bboxs(args, picked_particles[ugraph_index]["CoordinateX"], picked_particles[ugraph_index]["CoordinateY"], args.box_width, args.box_height)

        plt.figure(1, figsize=[16,16])
        plt.imshow(data[0], cmap='gray')
        
        ax = plt.gca()
        for bbox in boxes:
            corner = [bbox[0], bbox[1]] 
            height = bbox[3] - bbox[1]
            width = bbox[2] - bbox[0]
            rect = patches.Rectangle(
                corner, width, height, linewidth=1, edgecolor=[0, 1, 0], facecolor='none'
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
        green_patch = patches.Patch(color='green', label='Truth particles')
        red_patch = patches.Patch(color='red', label='Picked particles')
        plt.legend(handles=[green_patch, red_patch])
        plt.tight_layout()
        plt.savefig('{}_plot_truth_and_picked_boxes.png'.format(ugraph_index))
        plt.savefig('{}_plot_truth_and_picked_boxes.pdf'.format(ugraph_index), dpi=300)
        plt.clf()
        
    return


def boundary_investigation(args, particles: dict, picked_particles:dict)->None:
    """Make plots of numbers of particles as a function of x, y and z.
    Also plot the expected number with 3 std devs

    Args:
        args: Parsed command line arguments
        particles (dict): Truth particles dictionary
        picked_particles (dict): Picked particles dictionary
    """
    hist_bins_x = np.arange(0., float(args.image_x_pixels)+args.pixel_bin_width, args.pixel_bin_width)
    hist_bins_y = np.arange(0., float(args.image_y_pixels)+args.pixel_bin_width, args.pixel_bin_width)
    hist_bins_z = np.arange(0., float(args.image_z_pixels)+args.pixel_bin_width, args.pixel_bin_width)

    if args.debug:
        print('Hist x bins: {}'.format(hist_bins_x))
        print('Hist y bins: {}'.format(hist_bins_y))
        print('Hist z bins: {}'.format(hist_bins_z))

    expected_per_bin_x = (args.pixel_bin_width/float(args.image_x_pixels))*args.particles_per_ugraph*len(particles)
    expected_per_bin_y = (args.pixel_bin_width/float(args.image_y_pixels))*args.particles_per_ugraph*len(particles)
    expected_per_bin_z = (args.pixel_bin_width/float(args.image_z_pixels))*args.particles_per_ugraph*len(particles)

    # truth particles
    # grab all the x centres in one array
    truth_x_centres = []
    # grab all the y centres in one array
    truth_y_centres = []
    # grab all the z centres in one array
    truth_z_centres = []

    for ugraph_key in list(particles.keys()):
        truth_x_centres.extend(particles[ugraph_key]["x"]) 
        truth_y_centres.extend(particles[ugraph_key]["y"])
        truth_z_centres.extend(particles[ugraph_key]["z"])

    # picked particles
    # grab all the x centres in one array
    picked_x_centres = []
    # grab all the y centres in one array
    picked_y_centres = []

    for ugraph_key in list(picked_particles.keys()):
        picked_x_centres.extend(picked_particles[ugraph_key]["CoordinateX"])
        picked_y_centres.extend(picked_particles[ugraph_key]["CoordinateY"])

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
    plt.hlines([expected_per_bin_x], 0., args.image_x_pixels, colors=['black'], linestyles=['dashed'], label='expected')

    plt.grid(which='both')
    plt.xlabel('x coordinate (angstroms)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('particles_x.png', dpi=300)
    plt.savefig('particles_x.pdf')
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
    plt.hlines([expected_per_bin_y], 0., args.image_y_pixels, colors=['black'], linestyles=['dashed'], label='expected')

    plt.grid(which='both')
    plt.xlabel('y coordinate (angstroms)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('particles_y.png', dpi=300)
    plt.savefig('particles_y.pdf')
    plt.clf()

    # plot particles as function of z for truth only
    # blue is truth
    plt.figure(1, figsize=[16,16])
    # plot the truth
    plt.hist(truth_z_centres, bins=hist_bins_z, histtype='step', label='truth', color='blue')
    # plot the expected
    plt.hlines([expected_per_bin_z], 0., args.image_z_pixels, colors=['black'], linestyles=['dashed'], label='expected')

    plt.grid(which='both')
    plt.xlabel('z coordinate (angstroms)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('particles_z.png', dpi=300)
    plt.savefig('particles_z.pdf')
    plt.clf()
    return 


def grab_xy_array(ugraph_particles: dict, xlabel: str, ylabel: str)->np.ndarray:
    # create a np array to ckdtree for number of neighbours as function of distance
    particle_centres = np.empty((int(len(ugraph_particles[xlabel])) + int(len(ugraph_particles[ylabel]))), dtype=float)
    particle_centres[0::2] = ugraph_particles[xlabel]
    particle_centres[1::2] = ugraph_particles[ylabel]
    particle_centres = particle_centres.reshape(int(len(particle_centres)/2), 2)
    return particle_centres


def overlap_investigation(args, particles: dict, picked_particles:dict)->None:
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
        all_ugraph_neighbours[i] = np.zeros(len(particles), dtype=int)
        all_ugraph_picked_neighbours[i] = np.zeros(len(picked_particles), dtype=int)

    # determine for each diameter, the number of truth particles which overlap in a given ugraph
    # loop over each micrograph
    for i, ugraph_index in enumerate(list(particles.keys())):
        # create a np array to ckdtree for number of neighbours as function of distance
        truth_centres = grab_xy_array(particles[ugraph_index], "x", "y")

        # do the same for the picked particles
        picked_centres = grab_xy_array(picked_particles[ugraph_index], "CoordinateX", "CoordinateY")

        # get a vector of the number of neighbours, one entry per diameter
        neighbours = (ckdtree(truth_centres).count_neighbors(ckdtree(truth_centres), r=np.array(diameters_to_check)/2.))
        # account for the fact each particle matches itself, which would be 200 matches by default
        neighbours = neighbours - len(particles[ugraph_index]["x"])
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
    plt.savefig('particle_overlaps.png', dpi=300)
    plt.savefig('particle_overlaps.pdf')
    plt.clf()

    return

def invert_mask(mask: np.ndarray)->np.ndarray:
    return ~mask

def associate_truth_and_picked_particles(args, particles: dict, picked_particles: dict)->Tuple[dict, dict]:
     # set a single large radius to get sparse distance matrix using
    r = np.sqrt(np.power(float(args.image_x_pixels),2)+np.power(float(args.image_y_pixels),2))
    # r = args.particle_diameter

    # create a list to hold masks to apply to select unpicked particles for each ugraph
    # loop over the micrographs and for each find the unmatched truth particles
    for i, ugraph_index in enumerate(list(particles.keys())):
        # create a np array to ckdtree for number of neighbours as function of distance
        truth_centres = grab_xy_array(particles[ugraph_index], "x", "y")

        # do the same for the picked particles
        picked_centres = grab_xy_array(picked_particles[ugraph_index], "CoordinateX", "CoordinateY")

        # get the sparse_distance_matrix
        sdm = ckdtree(picked_centres).sparse_distance_matrix(ckdtree(truth_centres), r).toarray()
        sdm[sdm<np.finfo(float).eps] = np.nan
        if args.debug: 
            print('Shape of sdm: {}'.format(sdm.shape))

        # find the minimum value along axis 0 (1 per picked particle)
        # and keep track of the index of the truth particle
        closest_truth_index = []
        picked_particles[ugraph_index]["truth_match_index"] = []
        particles[ugraph_index]["picked_match_index"] = [np.nan] * len(particles[ugraph_index]["x"])
        for j, picked_particle in enumerate(sdm):
            truth_particle_index = int(np.nanargmin(picked_particle))
            # check if closest truth particle is within particle diameter of picked particle
            if picked_particle[truth_particle_index]>args.particle_diameter:
                closest_truth_index.append(np.nan)
                picked_particles[ugraph_index]["truth_match_index"].append(np.nan)
            # if it is, consider picking successful and allow the picked and truth particle to be associated with each other
            else:
                closest_truth_index.append(truth_particle_index)
            
                # grab the truth and picked particle indexes which are associated and add to particle dicts so you don't have to recalc this
                picked_particles[ugraph_index]["truth_match_index"].append(truth_particle_index)
            
                # any truth particles without assn picked particle have np.nan entry instead by default
                # filter them out using mask below
                particles[ugraph_index]["picked_match_index"][truth_particle_index] = int(j)

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
        particles[ugraph_index]["unpicked_mask"] = mask.tolist()

    return particles, picked_particles

def depth_investigation(args, particles: dict, picked_particles: dict, micrograph_paths: list[str]=None)->None:
    """ Take matched picked particles and turth particles and make:
    Histogram of the depth (z) of non-matched truth particles on same plot as 
    depth (z) of all truth particles, both normalized to be a pdf

    Args:
        args (_type_): Parsed command line arguments
        particles (dict): Truth particles dictionary
        picked_particles (dict): Picked particles dictionary
    """

    # grab a single list of all the truth z position and those for unmatched particles
    all_z = []
    all_unpicked_z = []
    for ugraph_index in particles.keys():
        all_z.extend(particles[ugraph_index]["z"])
        all_unpicked_z.extend(np.array(particles[ugraph_index]["z"], dtype=float)[particles[ugraph_index]["unpicked_mask"]].tolist())

    # want to plot z distribution of unpicked particles against that of all truth particles
    hist_bins_z = np.arange(0., float(args.image_z_pixels)+args.pixel_bin_width, args.pixel_bin_width)
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
    plt.savefig('unpicked_z.png', dpi=300)
    plt.savefig('unpicked_z.pdf')
    plt.clf()

    # could extend to also plot:
    # - x, y distribution of unpicked particles against all particles (both normalised)
    # - 2d x,y distribution of unpicked particles
    return


def save_truth_particles(args, particles: dict):
    output_yaml(args, args.truth_particles_yaml_filename, particles)
    return


def save_picked_particles(args, picked_particles: dict):
    output_yaml(args, args.picked_particles_yaml_filename, picked_particles)
    return


def output_yaml(args, outfile: str, my_dict: dict):
    with open(outfile, 'w') as writefile:
        yaml.dump(my_dict, writefile)
    if args.debug:
        print('Wrote output yaml: {}'.format(outfile))
    return


def input_yaml(args, infile: str=None)->dict:
    with open(infile, 'r') as read_file:
        my_dict = yaml.safe_load(read_file)
        if args.debug:
            print(type(my_dict))
            print(my_dict)
    return config

def label_micrograph_unpicked_truth_and_picked(args, micrograph_paths: list[str], particles: dict, picked_particles: dict)->None:
    # plot all the unpicked particles on the micrographs
    if micrograph_paths is not None:
        for micrograph_file in micrograph_paths:
            # Open up a mrc file to overlay the boxes with
            with mrcfile.open(micrograph_file) as mrc:
                data = mrc.data

                # work out the index of the image so you can grab the truth particles
                ugraph_index = grab_ugraph_index(args, micrograph_file)

                # Get the truth boxes
                boxes = twoD_image_bboxs(args, np.array(particles[ugraph_index]["x"], dtype=float)[particles[ugraph_index]["unpicked_mask"]], np.array(particles[ugraph_index]["y"], dtype=float)[particles[ugraph_index]["unpicked_mask"]], args.box_width, args.box_height)
                
                # Get the picked boxes
                picked_boxes = twoD_image_bboxs(args, picked_particles[ugraph_index]["CoordinateX"], picked_particles[ugraph_index]["CoordinateY"], args.box_width, args.box_height)

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
                plt.savefig('{}_plot_unpicked_truth_and_picked_boxes.png'.format(ugraph_index))
                plt.savefig('{}_plot_unpicked_truth_and_picked_boxes.pdf'.format(ugraph_index), dpi=300)
                plt.clf()
    return


def calculate_picking_efficiency(args, particles: dict, picked_particles: dict)->Tuple[dict,dict]:
    # add a key holding a float (which is the efficiency of particle picking) to the ugraph dict,
    # for both the truth and picked particles dicts

    # loop over ugraphs
    for ugraph_index in particles.keys():
        # work out the efficiency
        n_truths = float(len(particles[ugraph_index]["x"]))
        n_picked = n_truths - float(len(np.array(particles[ugraph_index]["unpicked_mask"],dtype=bool)[np.array(particles[ugraph_index]["unpicked_mask"], dtype=bool)==True]))
        picking_eff = n_picked / n_truths
    
        # add the efficiency to the ugraph info for truth and picked particle dicts
        particles[ugraph_index]["picking_efficiency"] = picking_eff
        picked_particles[ugraph_index]["picking_efficiency"] = picking_eff

        # print ot screen
        if args.debug:
            print('Micrograph {} has picking efficiency of {}'.format(ugraph_index, picking_eff))

    return particles, picked_particles

def run_main(args):

    # Let's parse the parakeet log file to get an outline (empty)
    # dataframe, dictionaries of pos and orientations with one entry per image
    # and the total number of molecules
    particles, n_micrographs = read_from_log(args)
    
    # if a path to one or more mrc file are given,
    # load them and assign labels one by one
    # before saving the labelled images
    if args.image_globpath is not None:
        # get the list of filepaths to micrograph files given by the user
        micrograph_paths = grab_filepaths(args.image_globpath)

        # now for each micrograph in the list, load them up, label them and save the labelled image
        for micrograph_file in micrograph_paths:
            save_image(args, micrograph_file)
            label_micrograph_truth(args, micrograph_file, particles)

    # if any investigations into picked particles, load in
    # the file with the picked particles
    if args.picked_particle_file is not None:
        picked_particles = load_picked_particles_from_starfile(args, args.picked_particle_file)

        for micrograph_file in micrograph_paths:
            label_micrograph_picked(args, micrograph_file, picked_particles)

        # find the assns between picked and truth particles and add info to dicts
        particles, picked_particles = associate_truth_and_picked_particles(args, particles, picked_particles)

        if args.image_globpath is not None:
            for micrograph_file in micrograph_paths:
                # overlay all truth and picked particles
                label_micrograph_truth_and_picked(args, micrograph_file, particles, picked_particles)
            # overlay unpicked truth particles and picked particles only    
            label_micrograph_unpicked_truth_and_picked(args, micrograph_paths, particles, picked_particles)

        if args.boundary_investigation:
            boundary_investigation(args, particles, picked_particles)

        if args.overlap_investigation:
            overlap_investigation(args, particles, picked_particles)

        if args.depth_investigation:
            depth_investigation(args, particles, picked_particles, micrograph_paths=micrograph_paths)

        # calculate picking efficiency per micrograph and insert into truth and picked particle dicts
        # this is a super simple alg which does not check 
        # for double-counted matches. These are mitigated through
        # the use of args.particle_diameter, which is the expected max particle diameter
        particles, picked_particles = calculate_picking_efficiency(args, particles, picked_particles)

    # create a truth particle summary yaml from truth particles dict for future analysis/plotting
    if args.truth_particles_yaml_filename:
        save_truth_particles(args, particles)
    # create a picked particle summary yaml from picked particles dict for future analysis/plotting
    if args.picked_particles_yaml_filename:
        save_picked_particles(args, picked_particles)
    
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--debug",
        help="Whether to print debugging statements",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "--log_filepath",
        "-p",
        help="Path to the 'log' file generated by Parakeet (which may also include other printed info).",
        type=str,
        default="",
    )

    parser.add_argument(
        "--image_globpath",
        "-i",
        help="Path to look for list of images in to plot boxes over. Include wildcards for multiple images!",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--basename_prefix",
        help="Part of Parakeet-generated files before the image index",
        type=str,
        default="image",
        required=False,
    )

    parser.add_argument(
        "--basename_suffix",
        help="Part of Parakeet-generated files after image index",
        type=str,
        default=".mrc",
        required=False,
    )

    parser.add_argument(
        "--index_start_value",
        help="Index of first image. No leading zeros required.",
        type=int,
        default=0,
        required=False,
    )

    parser.add_argument(
        "--picked_particle_file",
        help="File with picked particles to load in and perform studies on (.star)",
        type=str,
        default=None,
        required=False,
    )

    parser.add_argument(
        "--boundary_investigation",
        help="Whether to perform investigation into n particles picked as function of closeness to boundary",
        type=bool,
        default=False,
        required=False,
    )

    parser.add_argument(
        "--overlap_investigation",
        help="Whether to perform investigation into n overlapping particles as function of particle diameter",
        type=bool,
        default=False,
        required=False,
    )

    parser.add_argument(
        "--depth_investigation",
        help="Whether to perform investigation into n particles picked as function of depth into volume",
        type=bool,
        default=False,
        required=False,
    )

    parser.add_argument(
        "--box_width",
        help="Full width of overlay boxes on images",
        type=float,
        default=50.,
        required=False,
    )

    parser.add_argument(
        "--box_height",
        help="Full height of overlay boxes on images",
        type=float,
        default=50.,
        required=False,
    )

    parser.add_argument(
        "--pixel_bin_width",
        help="Number of image pixels to bin together for histograms",
        type=float,
        default=100.,
        required=False,
    )

    parser.add_argument(
        "--image_x_pixels",
        help="Number of image pixels along x axis",
        type=int,
        default=4000,
        required=False,
    )

    parser.add_argument(
        "--image_y_pixels",
        help="Number of image pixels along y axis",
        type=int,
        default=4000,
        required=False,
    )

    parser.add_argument(
        "--image_z_pixels",
        help="Number of image pixels along z axis",
        type=int,
        default=500,
        required=False,
    )

    parser.add_argument(
        "--particles_per_ugraph",
        help="Number of particles added to each ugraph",
        type=int,
        default=200,
        required=True,
    )

    parser.add_argument(
        "--truth_particles_yaml_filename",
        help="File name for yaml file which truth particle dict is saved into",
        type=str,
        default="truth_particles.yaml",
        required=False
    )

    parser.add_argument(
        "--picked_particles_yaml_filename",
        help="File name for yaml file which picked particle dict is saved into",
        type=str,
        default="picked_particles.yaml",
        required=False
    )    

    parser.add_argument(
        "--particle_diameter",
        help="Expected maximum particle diameter. Used to limit search radius for matching picked particles to truth particles",
        type=float,
        default=250.,
        required=False
    )

    args = parser.parse_args()
    if args.debug:
        for arg in vars(args):
            print('{}, {}'.format(arg, getattr(args, arg)))
    run_main(args)