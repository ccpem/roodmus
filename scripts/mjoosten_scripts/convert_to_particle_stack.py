
# imports
import os
import mrcfile
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

## load each stack of micrographs and convert to a particle stack
micrograph_all = []
micrograph_filename = []
metadata_all = []
metadata_filename = []
for filename in os.listdir():
    if filename.endswith(".mrc"): # load the micrographs
        micrograph = mrcfile.open(filename, permissive=True)
        micrograph_all.append(micrograph.data)
        micrograph_filename.append(filename)

    if filename.endswith(".npy"): # load the metadata
        metadata = np.load(filename, allow_pickle=True)
        metadata_all.append(metadata)
        metadata_filename.append(filename)

# convert the micrographs to a particle stack
L = 128 # size of the particle stack
particles_all = []
particles_all_aligned = []
particles_aligned_average = np.zeros((2*L, 2*L))
rotation_angles = []
for midx, micrograph in enumerate(micrograph_all):
    # show the micrograph
    # plt.imshow(micrograph[0,:,:])
    # plt.show()
    # plt.savefig(micrograph_filename[midx].replace(".mrc", ".png"))

    # get the metadata
    filename = micrograph_filename[midx]
    n = "_"+filename.split("_")[2].split(".")[0]+".npy"
    defocus = filename.split("_")[1]
    if defocus != "12000": continue
    for mm, m in enumerate(metadata_filename):
        if defocus in m and n in m:
            metadata = metadata_all[mm]
            break

    print(f"found metadata: {metadata_filename[mm]}")
    print(f"extracting particles from {micrograph_filename[midx]}")
    # get the particle coordinates
    for pidx, particle_metadata in enumerate(metadata):
        position = [int(r) for r in particle_metadata.position]
        orientation = particle_metadata.orientation

        # extract the particle
        particle = micrograph[0,position[1]-L:position[1]+L, position[0]-L:position[0]+L]
        
        # discard the particle if the size is not correct
        if particle.shape != (2*L, 2*L):
            continue

        # add the particle to the stack
        particles_all.append(particle)

        # rotate the particle
        angle = orientation[2]/np.pi*180 # convert to degrees
        rotation_angles.append(angle)
        particle_rotated = rotate(particle, angle, reshape=False)
        particles_all_aligned.append(particle_rotated)
        particles_aligned_average += particle_rotated

# turn particle stack to numpy array
particles_stack = np.array(particles_all)
print(f"particles_stack.shape = {particles_stack.shape}")
# save the particle stack as tif
tifffile.imwrite("particles_12000.tif", particles_stack)

# turn aligned particle stack to numpy array
particles_stack_aligned = np.array(particles_all_aligned)
print(f"particles_stack_aligned.shape = {particles_stack_aligned.shape}")
# save the aligned particle stack as tif
tifffile.imwrite("particles_12000_aligned.tif", particles_stack_aligned)

# save average of aligned particles
particles_aligned_average /= len(particles_all_aligned)
tifffile.imwrite("particles_12000_aligned_average.tif", particles_aligned_average)

# save the rotation angles
np.save("rotation_angles_12000.npy", rotation_angles)


