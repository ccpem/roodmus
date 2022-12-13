# roodmus

## Intended Utility After Combining Scripts

From Joel point of view, would be good to keep convention where micrograph images are in .mrc format and are labelled using an index of the form *000000.mrc -> *XXXXXX.mrc (ie: in python):
`str(index).zfill(6)`

Envisioning separate utilities of toolset to be:
1. Sample pdb/mmcif models from trajectory dataset
2. Generate SPA dataset using Parakeet python API:
   1. Parse arguments for SPA generation utility and use to 
   2. Set/randomly generate Parakeet configuration parameter lists (ie: defoci, lists of conformations to generate for each micrograph)
   3. Save Parakeet configuration to file (along with yamls containing the values of parameters which change for each micrograph, such as defocus)
   4. Generate SPA micrographs (which may update parakeet config object on-the-fly using parameter lists)
3. Tools to visualise and investigate truth particles (locations gained by parsing parakeet stdout) and picked particles (at the moment relion star file input is accepted)
4. Any tools related to heterogeneous reconstruction algorithms and validation/analysis thereof??? Need your advice on this Maarten!

I think any code related to reconstruction pipelines should not be in this repo (in my case I make use of ccpem-pipeliner repo). 

The utilities need to be sufficient to allow users to make use of Parakeet to create and investigate their datasets but should make use of Parakeet as an external piece of software.

--Maarten
My proposal would be to generally split the toolset into three parts:
1. preparation of trajectory for Parakeet
This requires sampling the trajectory and saving them into a (list of) temporary .pdb or .mmcif files to allow parakeet to run with this as an input. Sampling can be done equispaced or through waymarking. The user would have to set the number of frame to save or a condition for when a frame should be saved if waymarking. 
2. running Parakeet
A default run of the program should perhaps output something like a set of N micrographs containing a total of K particles. The parameters used for all micrographs can be constant, except the defocus which must vary between micrographs. We then need to make a choice on how the frames from the trajectory are distributed across the micrographs. By default, in my scripts K=n_frames, meaning each frame is only turned into a single particle in the data set. We could also allow frames to be present multiple times in the data set, or even multiple times in a single micrograph.
3. validation and visualisation
In addition to what Joel proposed, it may also be useful to mimic some of the processing steps in RELION/cryosparc such as CTF estimation and 3D refinement by plotting what the ground truth of these steps should look like (i.e. the Fourier transform of the micrographs and the ditribution of orientations).

In general I think it should not be too hard to port Joel's and my code if we agree on some structure for the final product.

## Plan of Action (updated 9/12/22)
1. Upload scripts 
2. Identify and agree on intended utilities in repo
3. Identify any utilities which scripts/code does not yet exist for
4. Agree on how to organise repo 
5. Agree on starting point for combined code and how to integrate scripts together for each utility
6. Division of labour

# Licensing
Need to figure out what kind of license we want and when/how we need to get it for the repo. 


