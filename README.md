# roodmus

## Utility of the program
The aim of roodmus is to turn an MD trajectory into a set of micrographs that can serve as ground-truth for testing cryo-EM (heterogeneous) reconstruction methods. The core of the micrograph simulation is done using parakeet. The program consists of three modules: 

### 1. Sample pdb/mmcif models from trajectory dataset
Given a directory of trajectory files (in for example .nc or .dcd format) and a topology file (for example a .pdb file), this module saves a selection of frames from the trajectory as .pdb files.

### 2. Generate SPA dataset using Parakeet python API:
Given a directory containing (any) .pdb files, a desired number of images to simulate and a number of molecules per image, this module generates a configuration file to run parakeet and then executes the parakeet simulation. Each micrograph subsamples the .pdb files to the number of molecules to generate, if not enough .pdb files are available, multiple instances of the same file are used. The config file is saved as a .yaml file with the same name as the image it corresponds to.

### 3. Tools to visualise and investigate truth particles 
This molecule can be used to compare the ground truth simulated particles to the output of common reconstruction steps in a cryo-EM processing pipeline. These steps include the particle picking, the CTF estimation and the 3D alignment. At the moment relion .star file and cryosparc .cs file inputs are accepted.

## scope of the project
### minimal
showcase simulating parakeet data sets using MD simulations
- between 1 and 5 examples of MD trajectories turned to parakeet data

- homogeneous reconstruction
    - long RNA transcription complex
    - homogeneous reconstruction
    - 3D classification
    - CryoDRGN?

- data set with some form of different discrete states
    - DEshaw spike trajectories
    - 3D classification to show different conformations
    - fit trajectory frames to 3D classes
    - pca on trajectory to get different discrete states, compare to 3D classes

- steered MD spike protein simulation
    - reconstruction
    - cryoDRGN

- course-grained MD simulation
    - possibly for later paper

- morphing structure (molecular motion database)
    - no trajectory known
    - large conformational change, unweighted
    - 3D classification
    - homogeneous reconstruction

- show can simulate prefered orientation
- compare reconstruction steps to ground truth (analysis module)
    - particle picking
    - CTF estimation
    - 3D alignment

### extended
investigations into how heterogeneous reconstruction tools handle the simulated data.

## Plan of Action (updated 25/01/203)
- wrapper to drive the program (Joel)
- analysis module: porting the picked particle comparison script from Joel's code
- analysis module: new script to compare 3D alignments to ground-truth particle orientations

# Licensing
Need to figure out what kind of license we want and when/how we need to get it for the repo. May have to use the same licence as parakeet.

# flow chart of current structure of Roodmus
![flowchart](flowchart.png)


