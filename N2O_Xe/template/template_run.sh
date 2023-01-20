#!/bin/bash
#SBATCH --job-name=r_SLVU_TTT_VVV
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=CCC
#SBATCH --mem-per-cpu=MMM
#SBATCH --exclude=node[101-124]

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

# Generate CHARMM files
if test -f "generate.out"; then
    out=$(grep "CHARMM" generate.out | tail -n 1)
    if grep -iq "STOP" <<< "$out"; then
        echo "Initialization already done"
    else
        echo "Restart initialization"
        srun $my_charmm -i generate.inp -o generate.out
    fi
else
    echo "Start initialization"
    srun $my_charmm -i generate.inp -o generate.out
fi

# Run equilibrium and production
if test -f "dyna.out"; then
    out=$(grep "CHARMM" dyna.out | tail -n 1)
    if grep -iq "STOP" <<< "$out"; then
        echo "Production already done"
    else
        echo "Restart production"
        srun $my_charmm -i dyna.inp -o dyna.out
    fi
else
    echo "Start production"
    srun $my_charmm -i dyna.inp -o dyna.out
fi

# We succeeded, reset trap and clean up normally.
trap - EXIT
exit 0
