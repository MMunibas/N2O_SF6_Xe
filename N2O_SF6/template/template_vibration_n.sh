#!/bin/bash
#SBATCH --job-name=v_SLVU_TTT_VVV_DDD
#SBATCH --partition=vshort
#SBATCH --nodes=1
#SBATCH --ntasks=CCC
#SBATCH --mem-per-cpu=MMM
#SBATCH --exclude=node[49-64]

#--------------------------------------
# Modules
#--------------------------------------

module load intel/2019-compiler-intel-openmpi-4.1.2
my_charmm=/data/toepfer/Project_VibRotSpec/intel_c47a2_n2o/build/cmake/charmm
# module load gcc/gcc-9.2.0
# my_charmm=/data/toepfer/Project_VibRotSpec/gcc_c47a2_n2o/build/cmake/charmm
ulimit -s 10420

#--------------------------------------
# Prepare Run
#--------------------------------------

export SLURMFILE=slurm-$SLURM_JOBID.out

#--------------------------------------
# Run equilibrium and production 
# simulations
#--------------------------------------

# Run CHARMM job
srun $my_charmm -i FINP -o FOUT

# We succeeded, reset trap and clean up normally.
trap - EXIT
exit 0
