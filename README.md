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

## Plan of Action (updated 9/12/22)
1. Upload scripts 
2. Identify and agree on intended utilities in repo
3. Identify any utilities which scripts/code does not yet exist for
4. Agree on how to organise repo 
5. Agree on starting point for combined code and how to integrate scripts together for each utility
6. Division of labour

# Licensing
Need to figure out what kind of license we want and when/how we need to get it for the repo. 


