import os
import gemmi
import argparse
import glob

# Written by Tom Burnley with edits from Joel Greer
# Probably included under this coppyright:
#
#     Copyright (C) 2021 CCP-EM
#
#     This code is distributed under the terms and conditions of the
#     CCP-EM Program Suite Licence Agreement as a CCP-EM Application.
#     A copy of the CCP-EM licence can be obtained by writing to the
#     CCP-EM Secretary, RAL Laboratory, Harwell, OX11 0FA, UK.
#

# Gemmi read the docs:
# https://gemmi.readthedocs.io/en/latest/index.html

# Gemmi Python API:


def create_map(structure, map_out, resolution=3.0):
    # Move coordinates to centre of P1 cubic box with margin of 10 Angstroms
    structure.spacegroup_hm = "P1"

    box = structure.calculate_box(margin=10.0)
    com = structure[0].calculate_center_of_mass()
    dims = (
        box.maximum[0] - box.minimum[0],
        box.maximum[1] - box.minimum[1],
        box.maximum[2] - box.minimum[2],
    )

    offset = 0.5 * max(dims)
    trans = (offset - com[0], offset - com[1], offset - com[2])
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    atom.pos = gemmi.Position(
                        atom.pos.x + trans[0],
                        atom.pos.y + trans[1],
                        atom.pos.z + trans[2],
                    )
    structure.cell.set(max(dims), max(dims), max(dims), 90, 90, 90)

    # Calculate Density w/ Electron form factors
    # See Density for FFT:
    #   https://gemmi.readthedocs.io/en/latest/hkl.html

    dencalc = gemmi.DensityCalculatorE()
    # Set resolution in Angstrom
    dencalc.d_min = resolution
    # Typical sampling rate
    dencalc.rate = 1.5
    dencalc.set_grid_cell_and_spacegroup(structure)
    dencalc.put_model_density_on_grid(structure[0])
    denmap = gemmi.Ccp4Map()
    denmap.grid = dencalc.grid

    # Set to map to mode 2 (32-bit signed real) for details see:
    #   https://www.ccpem.ac.uk/mrc_format/mrc2014.php
    denmap.update_ccp4_header(2, True)

    print("Write map: {}".format(map_out))
    denmap.write_ccp4_map(map_out)


def create_ensemble_from_structures(
    structures_dir, 
    file_extension, 
    ensemble_filename, 
    output_dir,
    simulate_density=False,
    ):
    # Get list of structures from directory
    print("Read structures from directory: {}".format(structures_dir))
    structure = None

    # read files, then sort them before making ensemble structure
    for file in sorted(glob.glob(structures_dir + "/*.{}".format(file_extension))):
        print("    {}".format(file))
        if structure is None:
            structure = gemmi.read_structure(file)
        else:
            model = gemmi.read_structure(file)[0]
            structure.add_model(model)
    # Rename models to start with 1 for Coot compatability
    #   Coot will not accept "Model 0" as a valid PDB
    for n, model in enumerate(structure, start=1):
        model.name = str(n)

    if simulate_density:
        # Call before saving PDB as simulate density my change unit cell dims
        map_out = os.path.join(
            output_dir,
            "{}.mrc".format(ensemble_filename)
        )
        create_map(structure=structure, map_out=map_out, resolution=3.0)

    # Save modified PDB
    ens = os.path.join(
        output_dir,
        "{}.{}".format(ensemble_filename, file_extension)
    )
    print("Save ensemble PDB: {}".format(ens))
    structure.write_pdb(ens)


def main():
    """
    Load multiple coordinate files in PDB format and make single ensemble PDB
    """
    print("\nEnsemble Tools\n")
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--structures_dir", type=str, default=os.getcwd())
    parser.add_argument("-s", "--simulate_density", type=bool, default=False)
    parser.add_argument("-e", "--extension", type=str, default='pdb')
    parser.add_argument("-o", "--ensemble_filename", type=str, default='gemmi_ens')
    parser.add_argument("-od", "--output_dir", type=str, default='')
    args = parser.parse_args()

    create_ensemble_from_structures(
        structures_dir=args.structures_dir,
        file_extension=args.extension,
        ensemble_filename=args.ensemble_filename,
        output_dir=args.output_dir,
        simulate_density=args.simulate_density,
    )


if __name__ == "__main__":
    main()
