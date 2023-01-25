# roodmus

## Utility of the program

Envisioning separate utilities of toolset to be:
1. Sample pdb/mmcif models from trajectory dataset
2. Generate SPA dataset using Parakeet python API:
   1. Parse arguments for SPA generation utility and use to 
   2. Set/randomly generate Parakeet configuration parameter lists (ie: defoci, lists of conformations to generate for each micrograph)
   3. Save Parakeet configuration to file (along with yamls containing the values of parameters which change for each micrograph, such as defocus)
   4. Generate SPA micrographs (which may update parakeet config object on-the-fly using parameter lists)
3. Tools to visualise and investigate truth particles (locations gained by parsing parakeet stdout) and picked particles (at the moment relion star file input is accepted)

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

## Plan of Action (updated 25/01/203)
- wrapper to drive the program
- 

# Licensing
Need to figure out what kind of license we want and when/how we need to get it for the repo. 

# flow chart of current structure of Roodmus
![flowchart](flowchart.png)


