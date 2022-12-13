

import yaml
import os
import gemmi
import numpy as np

num_images = 3
num_molecules_per_image = 375
num_molecules_per_pdb = 1
defocus_range = [-12000]
exposure = 400

def sample_defocus(defocus):
    # defocus = np.random.normal(defocus, 200)# value for sigma chosen arbitrarily
    return defocus

def split_pdb(pdb_gemmi, pdb_file, outdir="."):
    """split the pdb file into multiple pdb files"""
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outfilenames = []
    for midx, mdl in enumerate(pdb_gemmi):
        outfilename = os.path.join(outdir, os.path.basename(pdb_file.replace(".pdb", f"_{midx}.pdb")))
        outfilenames.append(outfilename)
        strct = gemmi.Structure()
        strct.add_model(mdl)
        strct.write_pdb(outfilename)
    return outfilenames

for n in range(num_images):

    #################### configure parakeet
    yaml_file = open('config.yaml', 'r')
    config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    # set the defocus
    defocus = sample_defocus(defocus_range[n%len(defocus_range)])
    config["microscope"]["lens"]["c_10"] = defocus

    # set the exposure
    config["microscope"]["beam"]["electrons_per_angstrom"] = exposure

    # load current pdb file
    pdb_file = f"../output/haptoglobin_haemoglobin/reference/pdb/traj_random_sample_{n*num_molecules_per_image}.pdb"
    pdb_gemmi = gemmi.read_structure(pdb_file)
    filenames = split_pdb(pdb_gemmi, pdb_file)

    config["sample"]["molecules"]["local"] = []
    for f, filename in enumerate(filenames):
        config["sample"]["molecules"]["local"].append({"filename" : os.path.basename(filename), "instances": []})

        # place N molecules into the micrograph
        for i in range(num_molecules_per_pdb):
            phi = np.random.uniform(0, 2*np.pi)
            config["sample"]["molecules"]["local"][f]["instances"].append({
                "position": None,
                "orientation": [0, 0, phi]
            })

        # save config
        with open('config.yaml', 'w') as outfile:
            yaml.dump(config, outfile, default_flow_style=False)
        yaml_file.close()

    #################### run parakeet

    os.system("parakeet.sample.new -c config.yaml")
    os.system("parakeet.sample.add_molecules -c config.yaml >> log.txt")
    os.system("parakeet.simulate.exit_wave -c config.yaml")
    os.system("parakeet.simulate.optics -c config.yaml")
    os.system("parakeet.simulate.image -c config.yaml")
    os.system(f"parakeet.export image.h5 -o {os.path.basename(pdb_file).replace('pdb', 'mrc')}")

    #################### clean up

    os.system("rm -rf *.h5")
    for filename in filenames:
        os.remove(filename)

