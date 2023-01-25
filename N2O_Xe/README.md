# N2O_Xe

Running Script_NVT_scan_samples.py will prepare the working directories for the NVT simulations of 
N2O in Xe at different system volumes representing the solvent density.
Currently it is scripted for a slurm queueing system with a run script template file in directory templates.

After the simmulations are complete, following evaluation script files can be executed to postprocess the simulations.

Script_evaluate_samples_n2o.py: Main evaluation script that collect general time series about the solutes N2O properties
and generate a 1DIR spectra from the Fourier Transform of the dipole-dipole correlation function.

Script_paper_figures.py: After running Script_evaluate_samples_n2o.py, this script will produce 1DIR figures for
publication.

Script_evaluate_samples_frq.py: Evaluate the instantaneous and quenched normal mode results.

Script_evaluate_samples_lcl.py: Generate solute-solvent radial distribution function and plot.

Script_evaluate_samples_slv.py: Generate solvent-solvent radial distribution function and plot.
