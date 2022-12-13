

gen_pdb_splits = False
gen_ensemble = True

# script to generate a data set of images from the hemoglobin MD trajecory
if gen_pdb_splits:
    from analysis.functions import convert_traj_to_pdb
    import pytraj
    import gemmi
    import random
    import numpy as np


    trajfile = "output/haptoglobin_haemoglobin/reference/trajectory_t_75_k_5000.nc"
    topology = "data/haptoglobin_haemoglobin/5hu6_MD.pdb"
    traj = pytraj.load(trajfile, top=topology)
    print("number of frames:", traj.n_frames)

    idx = np.arange(traj.n_frames)
    random.shuffle(idx)

    num_molecules_per_image = 375

    # convert the trajectory to a pdb file
    for i in range(0, traj.n_frames, num_molecules_per_image):
        pseudo_traj = traj[idx[i:i+num_molecules_per_image]]
        ensemble = convert_traj_to_pdb(pseudo_traj, gemmi.read_structure(topology), num_molecules_per_image, f"output/haptoglobin_haemoglobin/reference/traj_random_sample_{i}.pdb")
        # remove chain C and D from ensemble
        #for model in ensemble:
        #    model.remove_chain("C")
        #    model.remove_chain("D")
        #ensemble.write_pdb(f"output/hemoglobin/reference/traj_random_sample_{i}_noCD.pdb")

if gen_ensemble:
    # create ensemble from trajectory and simulate ensemble map
    from emmer.pdb.convert.convert_pdb_to_map import convert_pdb_to_map
    from emmer.ndimage.filter.low_pass_filter import low_pass_filter
    from analysis.functions import convert_traj_to_pdb
    import mrcfile
    import gemmi
    import pytraj
    from tqdm import tqdm
    import numpy as np

    trajfile = "output/haptoglobin_haemoglobin/reference/trajectory_t_75_k_5000.nc"
    topology = "data/haptoglobin_haemoglobin/5hu6_MD.pdb"
    traj = pytraj.load(trajfile, top=topology)
    print("number of frames:", traj.n_frames)

    # example_map_file = "data/hemoglobin/cryosparc_P51_J177_map.mrc"
    # example_map = gemmi.read_ccp4_map(example_map_file)
    # unitcell = example_map.grid.unit_cell
    # vsize = example_map.grid.spacing[0]
    # grid_size = example_map.grid.shape
    starting_model_gemmi = gemmi.read_structure(topology)
    unitcell = starting_model_gemmi.cell
    vsize = 1.0
    grid_size = (320, 320, 320)
    resolution = 3

    ensemble = convert_traj_to_pdb(traj, gemmi.read_structure(topology), 500, f"output/hemoglobin/reference/traj_random_sample_ensemble.pdb")
    ensemble_map = np.zeros(grid_size)
    progressbar = tqdm(total=len(ensemble))
    for single_model in ensemble:
        S = gemmi.Structure()
        S.add_model(single_model)
        map_from_model = convert_pdb_to_map(
            input_pdb=S,
            unitcell=unitcell,
            apix=vsize,
            size=grid_size,
        )
        map_from_model = low_pass_filter(map_from_model, vsize, resolution)
        ensemble_map += map_from_model
        progressbar.update(1)
    progressbar.close()


    with mrcfile.new("output/hemoglobin/reference/traj_random_sample_ensemble.mrc", overwrite=True) as mrc:
        mrc.set_data(np.float32(ensemble_map))
        mrc.voxel_size = vsize




