#!/bin/bash
#SBATCH --job-name=e_SLVU_TTT_VVV
#SBATCH --partition=long
#SBATCH --nodes=1
#SBATCH --ntasks=CCC
#SBATCH --mem-per-cpu=MMM
#SBATCH --exclude=node[77,109-128]

#--------------------------------------
# Modules
#--------------------------------------

module load intel/2019-compiler-intel-openmpi-4.1.2
my_charmm=/data/toepfer/Project_VibRotSpec/intel_c47a2_n2o/build/cmake/charmm
#module load gcc/gcc-9.2.0
#my_charmm=/data/toepfer/Project_VibRotSpec/gcc_c47a2_n2o/build/cmake/charmm
ulimit -s 10420

#--------------------------------------
# Prepare Run
#--------------------------------------

export SLURMFILE=slurm-$SLURM_JOBID.out

#--------------------------------------
# Parameters
#--------------------------------------

NPROD=NDYP
NSTR=NDYS
NEND=NDYE
VMODE=VMD
SLEEP_TIME=60s;

#--------------------------------------
# Evaluate production runs
#--------------------------------------

for i in $(seq $NSTR $NEND); do
    j=$(($i-1))
    stepdone=0
    while [ "$stepdone" -eq 0 ]; do
        diff=$(($NPROD-$i))
        if test -f "vibmode_"$VMODE"_forces.$j.dat"; then
            echo "Force projection evaluation $j already done"
            stepdone=1
        elif test -f "dyna_crd.$i.dcd"; then
            sed "s/NNN/$j/g" forces.inp > forces.$j.inp
            sed -i "s/NMD/$VMODE/g" forces.$j.inp
            echo "Start force projection evaluation $j"
            srun $my_charmm -i forces.$j.inp -o forces.$j.out
            grep "MODE   "$VMODE"  FREQ=" forces.$j.out > vibmode_"$VMODE"_forces.$j.dat
            grep -A3 '   EIGENVECTOR:' forces.$j.out > vibmode_"$VMODE"_projection.$j.dat
            grep -A3 '               FREQUENCIES' forces.$j.out > vibmode_frequencies.$j.dat
            rm -f forces.$j.inp forces.$j.out
            stepdone=1
        elif [ "$diff" -eq 0 ]; then
            if test -f "dyna.out"; then
                out=$(grep "CHARMM" dyna.out | tail -n 1)
                if grep -iq "STOP" <<< "$out"; then
                    sed "s/NNN/$j/g" forces.inp > forces.$j.inp
                    sed -i "s/NMD/$VMODE/g" forces.$j.inp
                    echo "Start force projection evaluation $j"
                    srun $my_charmm -i forces.$j.inp -o forces.$j.out
                    grep "MODE   "$VMODE"  FREQ=" forces.$j.out > vibmode_"$VMODE"_forces.$j.dat
                    grep -A3 '   EIGENVECTOR:' forces.$j.out > vibmode_"$VMODE"_projection.$j.dat
                    grep -A3 '               FREQUENCIES' forces.$j.out > vibmode_frequencies.$j.dat
                    rm -f forces.$j.inp forces.$j.out
                    stepdone=1
                fi
            fi
        else
            echo "Go to bed for $SLEEP_TIME"
            sleep $SLEEP_TIME
        fi
    done
done

# We succeeded, reset trap and clean up normally.
trap - EXIT
exit 0
