import os
import sys
import subprocess
import numpy as np

import MDAnalysis

import time
import getpass
from itertools import product
from glob import glob

#------------
# Parameters
#------------

# Trajectory data
dcdnames = 'dyna_crd.*.dcd'
dcdsplit = [['.', 1]]
dcdnum = {NDCD}

# Maximum number of tasks running
Ntasks_max = {NTSK}

# Step size per dcd file
Nruns_dcd = {NRND}
Nsteps_dcd = {NSTD}

# Template files
tempinp = 'vibration_n.inp'
temprun = 'vibration_n.sh'

# Working files
workrun = 'vibration_{:d}_{:d}.sh'
workinp = 'vibration_{:d}_{:d}.inp'
workout = 'vibration_{:d}_{:d}.out'

# Frequency results file and directory
rsltfile = 'inm_{:d}_{:d}.dat'
rsltdir = 'inm_results'
if not os.path.exists(rsltdir):
    os.mkdir(rsltdir)

#-----------------------
# Read function
#-----------------------

def read_results(idcd, istp):
    
    # Read results
    listfreq = ''
    grms = 0.0
    ivib = 0
    complete = False
    with open(workout.format(idcd, istp), 'r') as fout:
        for line in fout.readlines():
            if 'MINI>' in line:
                grms = float(line.split()[4])
            if '  VIBRATION MODE ' in line:
                if ivib>int(line.split()[2]):
                    listfreq += '  {:10.8f}'.format(grms) + '\n'
                elif ivib!=0:
                    listfreq += '  '
                ivib = int(line.split()[2])
                try:
                    freq = '{0:=10.6f}'.format(float(line.split()[4]))
                except:
                    if '*' in line:
                        freq = '{0:=10.6f}'.format(0.0)
                    elif not '='==line.split()[3][-1]:
                        freq = '{0:=10.6f}'.format(
                            float(line.split()[3].split("=")[1]))
                listfreq += freq
            if 'NORMAL TERMINATION BY NORMAL STOP' in line:
                complete = True
        listfreq += '  {:10.8f}'.format(grms) + '\n'
    
    # Write results
    rfile = os.path.join(rsltdir, rsltfile.format(idcd, istp))
    with open(rfile, 'w') as fres:
        fres.write(listfreq)
    
    return complete
    
#-----------------------
# Loop
#-----------------------

# Task ids and script file list
tskids = []
tsksrc = []
tskdcd = []
tskstp = []

done = False
while not done:
    
    # Get current dcd files
    dcdfiles = np.array(glob(dcdnames))
    idcds = np.zeros(len(dcdfiles), dtype=int)
    for idcd, dcdfile in enumerate(dcdfiles):
        d = dcdfile.split('/')[-1]
        for splt in dcdsplit:
            d = d.split(splt[0])[splt[1]]
        idcds[idcd] = int(d)

    # Sort dcd files
    dcdsort = np.argsort(idcds)
    dcdfiles = dcdfiles[dcdsort]
    idcds = idcds[dcdsort]

    for idcd, dcdfile in enumerate(dcdfiles):

        # Skip if not last and next dcd file is not created 
        if idcd != (dcdnum - 1) and not (idcd + 1) in idcds:
            continue

        # If last job, check if dcd file size is valid
        if idcd == (dcdnum - 1):
            if os.path.getsize(dcdfile) != os.path.getsize(dcdfiles[0]):
                continue
        
        for istp in range(Nruns_dcd):
            
            # Get result file
            rfile = os.path.join(rsltdir, rsltfile.format(idcd, istp))

            # Skip if result file already exists and is not empty
            if os.path.exists(rfile):
                if os.path.getsize(rfile) > 0:
                    continue
                
            # Check if CHARMM output file exist and complete
            if os.path.exists(workout.format(idcd, istp)):
                
                # Read results
                complete = read_results(idcd, istp)
                
                # Skip if complete
                if os.path.exists(rfile):
                    if complete and os.path.getsize(rfile) > 0:
                        continue
                        

            # Prepare input file
            with open(tempinp, "r") as ftemp:
                inplines = ftemp.read()

            # Prepare parameters
            # DCD file number
            inplines = inplines.replace('NDF', '{:d}'.format(idcd))
            # Start and last frame
            dframe = Nsteps_dcd//Nruns_dcd
            inplines = inplines.replace('NSTART', '{:d}'.format(
                int(istp*dframe)))
            if istp == (Nsteps_dcd - 1):
                inplines = inplines.replace('NLAST', '{:d}'.format(Nsteps_dcd))
            else:
                inplines = inplines.replace('NLAST', '{:d}'.format(
                int((istp + 1)*dframe)))
            
            # Write input file
            ifile = workinp.format(idcd, istp)
            with open(ifile, "w") as fwork:
                fwork.write(inplines)
            
            # Prepare run file
            with open(temprun, "r") as ftemp:
                inplines = ftemp.read()

            # Prepare parameters
            inplines = inplines.replace('DDD', '{:d}'.format(idcd))
            inplines = inplines.replace('FINP', ifile)
            inplines = inplines.replace('FOUT', workout.format(idcd, istp))
            
            # Write run file
            sfile = workrun.format(idcd, istp)
            with open(sfile, "w") as fwork:
                fwork.write(inplines)
            
            # Execute CHARMM
            task = subprocess.run(["sbatch", sfile], capture_output=True)
            tskids.append(int(task.stdout.decode().split()[-1]))
            tsksrc.append((workinp, workrun))
            tskdcd.append(idcd)
            tskstp.append(istp)
            print(task.stdout.decode())
            
            # Check if maximum task number is reached 
            if len(tskids) < Ntasks_max:
                continue
            
            max_cap = True
            while max_cap:
                
                # Get current tasks ids
                tsklist = subprocess.run(['squeue'], capture_output=True)
                idslist = [
                    int(job.split()[0])
                    for job in tsklist.stdout.decode().split('\n')[1:-1]]
                
                # Check if task is still running
                for it, tid in enumerate(tskids):
                    
                    if not tid in idslist:
                        
                        # Remove task id and run file
                        del tskids[it]
                        if os.path.exists(tsksrc[it][0]):
                            os.remove(tsksrc[it][0])
                        if os.path.exists(tsksrc[it][1]):
                            os.remove(tsksrc[it][1])
                        
                        # Read results
                        complete = read_results(tskdcd[it], tskstp[it])
                        
                        # Remove trajectory from current task list
                        del tskdcd[it]
                        del tskstp[it]
                        if len(tskids) <= Ntasks_max:
                            max_cap = False
                        break
                    
                # Wait 30 seconds
                if max_cap:
                    time.sleep(30)
            
    # Check if all result files are written (even if empty)
    rfile_last = os.path.join(rsltdir, rsltfile.format(
        idcds[-1], Nsteps_dcd - 1))
    if os.path.exists(rfile_last):
        done = True
        
    # Check stop flag
    if os.path.exists("stop"):
        done = True
        
    # Wait else
    time.sleep(600)
    


