# N2O_SF6_Xe

Source files and python scripts for running CHARMM NVT simulations of 
a single N2O molecule in SF6 or xenon solvent at difference solvent
densities and model simulation at specific initial conditions for 
a single N2O molecule in gas phase.

## Requirements

- CHARMM c47a2 or higher with compiled usersb.F90 file from the 'charmm' directory
- packmol
- python 3.8 or higher 
- MDAnalysis

## Order

First run the MD simulation of N2O in Xe and SF6 as well as the N2O model simulation 
(see subdirectories).
After completion, run all the evaluation scripts in the subdirectories in the order
as described there. 
Last run the evaluation scripts in the main directory to generate result figures for publication.
