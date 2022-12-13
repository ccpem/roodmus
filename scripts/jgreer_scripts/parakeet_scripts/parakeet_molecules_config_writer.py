from this import d
import yaml
import numpy as np
import os
import sys
import argparse

from parakeet_config_writer import output_yaml

# The intention of this code is to create a dict of lists of dictionaries (1 outermost dict per image where the inner dicts 
# each contain a filename (ie: each conformation used) and the number of instances of it to put
# in the image). 
# This .yaml file will then be read by parakeet_config_writer.py which will insert the list of dicts in
# the config file dictionary entry: config["sample"]["molecules"]["local"]

def summarise_defoci(args, defocus_values: dict):
    micrograph_indices = defocus_values.keys()
    defoci = defocus_values.values()
    with open(args.defocus_logfilename, 'w') as f:
        print('micrograph_index, defocus', file=f)
        for micrograph_index, defocus in zip(micrograph_indices, defoci):
            print('{}, {}'.format(str(micrograph_index).zfill(6), defocus), file=f)
    return

def generate_defocus_value(args):
    assert args.spread_defocus_type=="gaussian" or args.spread_defocus_type=="uniform"
    if args.spread_defocus_type=="gaussian":
        defocus_val = np.random.normal(loc=args.mean_defocus, scale=args.spread_defocus, size=1)
    else:
        # ie; args.spread_defocus_type=="uniform"
        defocus_val = np.random.randint(args.mean_defocus-args.spread_defocus, args.mean_defocus+args.spread_defocus, size=1)
    return defocus_val

def summarise_confs(args, count_confs: np.array)->None:

    uniq, count = np.unique(count_confs, return_counts=True)
    # Order these in ascending conf order
    sorting_arr = np.argsort(uniq)
    
    uniq = uniq[sorting_arr]
    count = count[sorting_arr]
    # Print these stats to console/file
    with open(args.logfilename, 'w') as f:
        print('conformation, n_occurences', file=f)
        for (conf, occs) in zip(uniq, count):
            print('{}, {}'.format(str(conf).zfill(args.image_no_sigfig), occs), file=f)
    return

def run_main(args):
    # the argparser gives both of the following:
    # grab the number of images required (from bash script)
    # grab the number of conformations which were generated

    # uniformly sample from these conformations (don't ensure that equal number of each conf is used, just rely on stats)
    # - however, do print out a log (yaml) file with the list of dicts and another with the number of each conf used (maybe in same yaml?)
    # The parakeet_config_writer.py will make use of this dict of lists of dicts yaml file
    
    # Make a dict to hold the local mols list with one entry per image for config["sample"]["molecules"]["local"]
    local_mols = {}
    count_confs = np.array((), dtype=int)

    defocus_values = {}

    for image_n in range(args.n_images):
        # get list (np array) of confs in this image
        confs_in_img = np.random.randint(0, args.n_confs, size=args.n_particles_per_img)
        count_confs = np.append(count_confs, confs_in_img, axis=0)

        # convert this to key value dict with no of occurences of each
        uniq, count = np.unique(confs_in_img, return_counts=True)

        img_mols = []
        for (conf, n_particles) in zip(uniq, count):
            # now we make 1 dict (filename and instances) for each conf this img uses
            mol_dict = {}
            mol_dict["filename"] = args.conf_path+args.sampling_type+"_"+str(args.n_confs)+"/"+args.conformations_file+str(conf).zfill(args.image_no_sigfig)+args.conformations_file_ext
            mol_dict["instances"] = int(n_particles)
            img_mols.append(mol_dict)

        # so now we have a list of dicts for a given image
        # let's make this into a dict of lists of dicts (indexed by 6-digit image number)
        # ie: dict->list->dict (which holds filename+instances fields)
        local_mols[str(image_n).zfill(6)] = img_mols 

        # sample a defocus value for this image
        # make a dict of c_10 values indexed by image index
        defocus_val = float(generate_defocus_value(args))
        defocus_values[str(image_n).zfill(6)] = defocus_val

    # now write out the list to a yaml file (I guess as part of a dict)
    output_yaml(args.filename, local_mols)
    # also write out the yaml with the defocus values indexed by image index
    output_yaml(args.filename_defocus, defocus_values)

    # Now count the number of each conf which occur in the entire dataset
    summarise_confs(args, count_confs)
    # Print the defocus value for each micrograph to a file
    summarise_defoci(args, defocus_values)

    return


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--n_images", help="Number of images to generate with Parakeet", type=int, default=1)
    parser.add_argument("-c", "--n_confs", help="Number of conformations to generate images using Parakeet", type=int, default=1)
    parser.add_argument("-np", "--n_particles_per_img", help="Number of particles per image", type=int, default=1)
    parser.add_argument("-cp", "--conf_path", help="Path to head conformation sampling directory", type=str, default="")
    parser.add_argument("-st", "--sampling_type", help="Name of sampling type of for conformation generation. Is the subdir name.", type=str, default="")
    parser.add_argument("-cf", "--conformations_file", help="Start of the conformation filename (the bit before the number)", type=str, default="")
    parser.add_argument("-cfe", "--conformations_file_ext", help="Conformations filename extension", type=str, default="")
    parser.add_argument("-isf", "--image_no_sigfig", help="Number of figures including leading zeros in image name", type=int, default=6)
    parser.add_argument("-f", "--filename", help="Name of yaml file to write dictionary holding list of local molecule dicts", type=str, default="local_mols.yaml")
    parser.add_argument("-l", "--logfilename", help="Name of log file to write summary table of confs for dataset", type=str, default="local_mols.log")
    parser.add_argument("-md", "--mean_defocus", help="Mean defocus value", type=float, default=-15000)
    parser.add_argument("-sd", "--spread_defocus", help="Std deviation or uniform range half-width of defocus values", type=float, default=2000)
    parser.add_argument("-sdt", "--spread_defocus_type", help="Choose gaussian or uniform spread for defocus values", type=str, default="gaussian")
    parser.add_argument("-fd", "--filename_defocus", help="filename_defocus", type=str, default="defoci.yaml")
    parser.add_argument("-d", "--defocus_logfilename", help="Name of logfile to write defocus of each micrograph to", type=str, default="defoci.log")
    args=parser.parse_args()
    run_main(args)
