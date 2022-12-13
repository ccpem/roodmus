
# generate a parakeet data set from template pdb files

import os
import numpy as np

num_images = 1
defocus_list = [-1500, -2000, -2500, -3000]
exposure = 40
voltage = 200
box_xy = 4000
box_z = 160
pixel_size = 1
mode = "still"
ice = False
radiation = False

datadir = "output/haptoglobin_haemoglobin/reference"
pdbdir = "output/haptoglobin_haemoglobin/reference/pdb"
mrcdir = "output/haptoglobin_haemoglobin/reference/mrc"

if not os.path.exists(pdbdir):
    os.mkdir(pdbdir)
if not os.path.exists(mrcdir):
    os.mkdir(mrcdir)

idx = 0
for f in os.listdir(datadir):
    if f.endswith(".pdb") and "traj_random_sample" in f:
        # move current pdb file to working directory
        os.system("mv " + os.path.join(datadir, f) + " " + f)
        # break

        command_line = "python parakeet/parakeet_simulation.py --pdb_files {} --num_molecules {} --num_images {} --defocus {} --exposure {} --voltage {} --box_z {} --pixel_size {} --mode {} ".format(
            f,
            1,
            num_images,
            defocus_list[idx],
            exposure,
            voltage,
            box_z,
            pixel_size,
            mode,
        )
        if ice:
            command_line += "--ice"
        if radiation:
            command_line += "--radiation"
        os.system(command=command_line)

        # move generated mrc file to mrc directory
        image_name = f"image_0_*.mrc"
        image_new_name = os.path.basename(f).replace(".pdb", f"_{defocus_list[idx]}.mrc")
        os.system("mv " + image_name + " " + os.path.join(mrcdir, image_new_name))

        # move pdb file to pdb directory
        os.system("mv " + f + " " + pdbdir)

        # rename metadata file
        metadata_name = f.replace("traj_ramdom_sample", "metadata").replace(".pdb", ".npy")
        os.system("mv metadata.npy " + metadata_name)

        idx = (idx + 1) % len(defocus_list)

        # break

# combine metadata files into one
idx = 0
for f in os.listdir():
    if f.endswith(".npy") and "traj_random_sample" in f:
        metadata = np.load(f, allow_pickle=True)
        if idx == 0:
            metadata_all = metadata
        else:
            metadata_all = np.concatenate((metadata_all, metadata))
        idx += 1
np.save("metadata_all.npy", metadata_all)

# remove individual metadata files
for f in os.listdir():
    if f.endswith(".npy") and "traj_random_sample" in f:
        os.remove(f)

        
