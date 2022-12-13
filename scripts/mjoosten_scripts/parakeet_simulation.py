
import argparse
parser = argparse.ArgumentParser(description='Parakeet simulation script')
cfg = parser.add_argument_group('configuration')
cfg.add_argument("--exposure", help="exposure [e/A^2]", type=float, default=40)
cfg.add_argument("--voltage", help="voltage [kV]", type=float, default=300)
cfg.add_argument("--box_xy", help="box size in xy direction", type=float, default=4000)
cfg.add_argument("--box_z", help="box size in z direction", type=float, default=200)
cfg.add_argument("--pixel_size", help="pixel size", type=float, default=1)
cfg.add_argument("--defocus", help="defocus", type=float, default=-5000, nargs="+")
cfg.add_argument("--pdb_files", help="pdb files of the proteins", type=str, default="6nbc.pdb", nargs="+")
cfg.add_argument("--num_molecules", help="number of molecules", type=int, default=1, nargs="+")
cfg.add_argument("--mode", help="scan mode", type=str, default="still")
cfg.add_argument("--num_images", help="number of images", type=int, default=1)
cfg.add_argument("--ice", help="ice", action="store_true")
cfg.add_argument("--radiation", help="radiation", action="store_true")
parser.add_argument("--NoSplit", help="if there are multiple models in a .pdb file, do not split them", action="store_true")
args = parser.parse_args()

import parakeet
import os
import gemmi
import numpy as np
from tqdm import tqdm

# debug
# args.pdb_files = [
#     "traj_random_sample_0.pdb",
#     "traj_random_sample_1125.pdb",
#     "traj_random_sample_1500.pdb",
#     "traj_random_sample_375.pdb",
#     "traj_random_sample_750.pdb",
# ]

def set_config(Config, exposure=100,  voltage=300, box_xy=4000, box_z=200, pixel_size=1, defocus=-5000,
  pdb_files=[], num_molecules=1, molecule_poses=None, molecule_positions=None, mode="still", num_images=1, ice=False, radiation=True):
    """configure Parakeet simulation"""

    Config.microscope.beam.electrons_per_angstrom = exposure
    Config.microscope.beam.energy = voltage
    # Config.microscope.beam.defocus_drift
    Config.microscope.detector.nx = box_xy
    Config.microscope.detector.ny = box_xy
    Config.microscope.detector.pixel_size = pixel_size
    Config.microscope.lens.c_10 = defocus
    Config.sample.box = [box_xy*pixel_size, box_xy*pixel_size, box_z]
    Config.sample.centre = [box_xy*pixel_size/2, box_xy*pixel_size/2, box_z/2]
    Config.sample.molecules = parakeet.config.Molecules()

    ## create molecule poses and positions
    if molecule_poses is not None and molecule_positions is not None:
        pose_config = []
        for molecule_pose, molecule_position in zip(molecule_poses, molecule_positions):
            molpos_list = []
            for m_pose, m_position in zip(molecule_pose, molecule_position):
                molpose = parakeet.config.MoleculePose()
                molpose.orientation = m_pose
                molpose.position = m_position
                molpos_list.append(molpose)    
            pose_config.append(molpos_list) 
    ## create only poses
    elif molecule_poses is not None:
        pose_config = []
        for molecule_pose in molecule_poses:
            molpos_list = []
            for m_pose in molecule_pose:
                molpose = parakeet.config.MoleculePose()
                molpose.orientation = m_pose
                molpose.position = None
                molpos_list.append(molpose)    
            pose_config.append(molpos_list)

    ## add molecules
    if molecule_poses is None:
        Config.sample.molecules.local = [parakeet.config.LocalMolecule(filename=r, instances=n) for r, n in zip(pdb_files, num_molecules)]
    else:
        Config.sample.molecules.local = [parakeet.config.LocalMolecule(filename=r, instances=n) for r,n in zip(pdb_files, pose_config)]

    Config.sample.shape.cuboid.length_x = box_xy*pixel_size
    Config.sample.shape.cuboid.length_y = box_xy*pixel_size
    Config.sample.shape.cuboid.length_z = box_z
    Config.sample.shape.type = "cuboid"
    Config.scan.mode = mode
    Config.scan.num_images = num_images

    # extra options
    Config.simulation.ice = ice
    Config.simulation.radiation_damage_model = radiation

def generate_molecule_poses(num_molecules, box_xy, box_z, pixel_size):
    molecule_poses = []
    for j in range(num_molecules):
        # for the experiment where the poses of all molecules are top-down
        phi = 0
        theta = 0
        # random in-plane rotation
        psi = np.random.uniform(0, 360)/180*np.pi
        molecule_poses.append((phi, theta, psi))
    return molecule_poses

def generate_molecule_positions(num_molecules, box_xy, box_z, pixel_size):
    molecule_positions = []
    x = np.linspace(800, box_xy*pixel_size-800, box_xy//400)
    y = np.linspace(800, box_xy*pixel_size-800, box_xy//400)
    for i in range(num_molecules):
        x_idx = i % len(x)
        y_idx = i // len(x)
        molecule_positions.append([x[x_idx], y[y_idx], box_z/2])
    return molecule_positions

def sample_defocus(defocus):
    # defocus = np.random.normal(defocus, 200)# value for sigma chosen arbitrarily
    return defocus

def get_total_num_atoms(num_images, num_molecules, defocus):
    total_num_atoms = num_images * sum(num_molecules) * len(defocus)
    return int(total_num_atoms)


def run_simulation(Config, outfilename, exposure, voltage, box_xy, box_z, pixel_size, defocus, pdb_files, num_molecules, mode, num_images, ice, radiation, gen_poses=False, gen_positions=False):

    if gen_poses:
        molecule_poses = []
        for n in num_molecules:
            molecule_poses.append(generate_molecule_poses(n, box_xy, box_z, pixel_size))
    else:
        molecule_poses = None

    if gen_positions:
        molecule_positions = []
        for n in num_molecules:
            molecule_positions.append(generate_molecule_positions(n, box_xy, box_z, pixel_size))
    else:
        molecule_positions = None

    # set the configuration
    set_config(Config, exposure=exposure, voltage=voltage, box_xy=box_xy, box_z=box_z, pixel_size=pixel_size, defocus=defocus,
        pdb_files=pdb_files, molecule_poses=molecule_poses, molecule_positions=molecule_positions, num_molecules=num_molecules,
         mode=mode, num_images=num_images, ice=ice, radiation=radiation)

    # run the simulation
    Sample = parakeet.sample.new(Config, filename="sample.h5")
    Sample = parakeet.sample.add_molecules(Config, sample_file="sample.h5")
    parakeet.simulate.exit_wave(Config, Sample, exit_wave_file="exit_wave.h5")
    parakeet.simulate.optics(Config, exit_wave_file="exit_wave.h5", optics_file="optics.h5")
    parakeet.simulate.image(Config, optics_file="optics.h5", image_file="image.h5")
    # parakeet.config.save(Config, "config.yaml")
    # extract the image to .mrc file
    os.system("parakeet.export image.h5 -o " + outfilename)

    # clean up the files
    os.system("rm sample.h5 exit_wave.h5 optics.h5 image.h5 -f")

    return Sample

def split_pdb(pdb_gemmi, pdb_file, outdir="."):
    """split the pdb file into multiple pdb files"""
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    outfilenames = []
    for midx, mdl in enumerate(pdb_gemmi):
        outfilename = os.path.join(outdir, pdb_file.replace(".pdb", f"_{midx}.pdb"))
        outfilenames.append(outfilename)
        strct = gemmi.Structure()
        strct.add_model(mdl)
        strct.write_pdb(outfilename)
    return outfilenames

def main(args):
    # initialise config
    Config = parakeet.config.new(filename="config.yaml", full=True)

    # if the supplied .pdb file(s) contain multiple molecules, then we need to split them into multiple .pdb files
    if not args.NoSplit:
        # determine total number of structures
        num_structures = 0
        for pdb_file in args.pdb_files:
            pdb_gemmi = gemmi.read_structure(pdb_file)
            num_structures += len(pdb_gemmi)
        
        split_pdbs = []
        to_be_removed = []
        split_num_molecules = []
        idx = 0
        for pidx, pdb_file in enumerate(args.pdb_files):
            pdb_gemmi = gemmi.read_structure(pdb_file)
            if len(pdb_gemmi) > 1:
                # split the pdb file
                split_filename = split_pdb(pdb_gemmi, pdb_file)
                split_pdbs += split_filename
                to_be_removed += split_filename

                # determine the number of molecules for each split pdb file
                if len(args.num_molecules) == 1:
                    # if only a single value is supplied, then use that value for all split pdb files
                    split_num_molecules += [args.num_molecules[0] for _ in range(len(split_filename))]
                elif len(args.num_molecules) == len(args.pdb_files):
                    # if a value is supplied for each pdb file, then use that value for each split pdb file
                    split_num_molecules += [args.num_molecules[pidx] for _ in range(len(split_filename))]
                elif len(args.num_molecules) == num_structures:
                    # if a value is supplied for each structure, then use that value for each split pdb file
                    split_num_molecules.append(args.num_molecules[idx])
                    idx += 1
                else:
                    # unknown number of values
                    raise ValueError("Unknown number of values for num_molecules")                

            else:
                split_pdbs.append(pdb_file)
                split_num_molecules.append(args.num_molecules[pidx] if len(args.num_molecules) > 1 else args.num_molecules[0])
        args.pdb_files = split_pdbs
        args.num_molecules = split_num_molecules

    # generate the images
    metadata = np.recarray(shape=get_total_num_atoms(args.num_images, args.num_molecules, args.defocus), dtype=[
        ("name", object),
        ("defocus", float),
        ("frame", int),
        ("nImage", int),
        ("voltage", float),
        ("exposure", float),
        ("pixel_size", float),
        ("position", list),
        ("orientation", list),
    ])

    progressbar = tqdm(total=int(args.num_images*len(args.defocus)), desc="Simulating Parakeet data")
    idx = 0
    for i in range(args.num_images):
        for defocus in args.defocus:
            # sample defocus
            defocus_sample = sample_defocus(defocus)
            # set filename
            outfilename = f"image_{i}_{defocus}.mrc"
            # run the simulation
            Sample = run_simulation(Config, outfilename=outfilename, exposure=args.exposure, voltage=args.voltage,
             box_xy=args.box_xy, box_z=args.box_z, pixel_size=args.pixel_size, defocus=defocus_sample, pdb_files=args.pdb_files,
              num_molecules=args.num_molecules, mode=args.mode, num_images=args.num_images, ice=args.ice, radiation=args.radiation,
              gen_poses=True, gen_positions=False)

            # extract the molecule coordinates and positions
            for molecule in Sample.molecules:
                _, positions, orientations = Sample.get_molecule(molecule)
                for position, orientation in zip(positions, orientations):
                    metadata[idx].name = molecule
                    metadata[idx].defocus = defocus_sample
                    metadata[idx].frame = 0
                    metadata[idx].nImage = i
                    metadata[idx].voltage = args.voltage
                    metadata[idx].exposure = args.exposure
                    metadata[idx].pixel_size = args.pixel_size
                    metadata[idx].position = position
                    metadata[idx].orientation = orientation
                    idx += 1

            progressbar.update(1)
    progressbar.close() 

    # save the metadata
    np.save("metadata.npy", metadata)

    # remove the temporary files
    if not args.NoSplit:
        for filename in to_be_removed:
            os.remove(filename)

if __name__ == "__main__":
    if type(args.pdb_files) == str:
        args.pdb_files = [args.pdb_files]
    if type(args.defocus) == int:
        args.defocus = [args.defocus]
    if type(args.num_molecules) == int:
        args.num_molecules = [args.num_molecules]

    main(args)
    print("Done!")
    exit(0)


