#!/bin/bash
#SBATCH --job-name=N2O
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=1200

#--------------------------------------
# Modules
#--------------------------------------

module load intel/2019-compiler-intel-openmpi-4.1.2
my_charmm=/data/toepfer/Project_VibRotSpec/intel_c47a2_n2o/build/cmake/charmm
ulimit -s 10420

#--------------------------------------
# Prepare Run
#--------------------------------------

export SLURMFILE=slurm-$SLURM_JOBID.out

#--------------------------------------
# Run equilibrium and production 
# simulations
#--------------------------------------

srun $my_charmm -i dyna.inp -o dyna.out

# We succeeded, reset trap and clean up normally.
trap - EXIT
exit 0
