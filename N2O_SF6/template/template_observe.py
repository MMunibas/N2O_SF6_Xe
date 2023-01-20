#!/usr/bin/python

# Basics
import os
import subprocess
import numpy as np

# Miscellaneous
import time
import MDAnalysis
import getpass

runfile = "%RFILE%"
inpfile = "%IFILE%"
outfile = "%OFILE%"
maxndcd = %MAXND%

indices = np.array([0, 1, 2])
psffile = "init.n2o_sf6.psf"

def check_status():
    
    with open(inpfile, 'r') as f:
        
        # Read lines
        inplines = f.read()
        
    # Check heat output
    if os.path.exists("heat.dcd"):
        inplines = inplines.replace("set mini 0", "set mini 1")
    
    # Check NVE output
    if os.path.exists("nver.dcd"):
        inplines = inplines.replace("set nver 0", "set nver 1")
    
    # Check equi output
    if os.path.exists("equi.dcd"):
        inplines = inplines.replace("set heat 0", "set heat 1")
    
    # Check dcd output
    if os.path.exists("dyna_crd.0.dcd"):
        inplines = inplines.replace("set equi 0", "set equi 1")
            
    # Write modified input
    with open(inpfile, 'w') as f:
        f.write(inplines)

# Start/Continue simulation
check_status()
task = subprocess.run(['sbatch', runfile], capture_output=True)
tskid = int(task.stdout.split()[-1])

# Create continue file (if deleted, while loop stops)
with open("continue", 'w') as f:
    f.write("Continue to observe")

finished = False
error = False
latest_step = 0
checked_dcd = 0
while not finished:
    
    # Wait 5/10/15 minutes
    time.sleep(900)
    
    # Check stop flag
    if not os.path.exists("continue"):
        finished = True
        break
    
    # Get task id list
    user = getpass.getuser()
    tsklist = subprocess.run(
        ['squeue', '-u', user], capture_output=True)
    idslist = [
        int(line.split()[0])
        for line in tsklist.stdout.decode().split('\n')[1:-1]]
    statuslist = [
        line.split()[4]
        for line in tsklist.stdout.decode().split('\n')[1:-1]]
    
    # Get status of the job
    for ii, idi in enumerate(idslist):
        if tskid == idi:
            if statuslist[ii]=="R":
                job_is_running = True
            else:
                job_is_running = False
            break
    if not tskid in idslist:
        job_is_running = False
    
    # Current dcd file
    dcdfile = "dyna_crd.{:d}.dcd".format(checked_dcd)
    
    if os.path.exists(dcdfile) and job_is_running:
        
        # Open current dcd file
        try:
            dcd = MDAnalysis.Universe(psffile, dcdfile)
        except OSError: # empty file
            continue
        
        masses = dcd._topology.masses.values
        
        # Iterate over frames
        for ii, frame in enumerate(dcd.trajectory):
            
            cell = frame._unitcell
            pos = frame._pos[indices]
            mss = masses[indices]
            
            com = np.sum(mss.reshape(-1, 1)*pos, axis=0)/np.sum(mss)
                    
            # Check Center of Mass location and change
            if ii==0:
                old_com = com.copy()
                
            oob_threshold = 10.0
            oob = any(abs(com) > cell[:3]/2. + oob_threshold)
            if oob and not oob_once:
                print("Target out of box: ", checked_dcd, dcdfile, ii)
                print(os.getcwd())
                print("Box size: ", cell[:3])
                print("Old COM: ", old_com)
                print("New COM: ", com)
                oob_once = True
                error = True
                break
            
            diff_com = old_com - com
            diff_threshold = 0.5
            jump = False
            if any(abs(diff_com) > diff_threshold):
                for ij in range(3):
                    diff_ij = abs(diff_com[ij])
                    if diff_ij > diff_threshold:
                        if diff_ij < cell[ij] - diff_threshold:
                            jump = True
            if jump:
                print("Position jump: ", checked_dcd, dcdfile, ii)
                print(os.getcwd())
                print("Box size: ", cell[:3])
                print("Old COM: ", old_com)
                print("New COM: ", com)
                print(abs(diff_com) > diff_threshold)
                print(abs(diff_com) < cell[:3] - diff_threshold)
                jump = False
                error = True
                break
            
            # Save current COM
            old_com = com.copy()
            
    # If eror occurs, adopt current step and rerun simulation.
    # Else, check for completeness
    if error:
        
        # Cancel job
        subprocess.run(['scancel', '{:d}'.format(tskid)])
        
        # Update current step in input file
        with open(inpfile, 'r') as f:
            
            # Read lines
            inplines = f.read()
        
        for line in inplines.split('\n'):
            if 'set ndcd ' in line:
                defline = line
                break
        
        inplines = inplines.replace(defline, 'set ndcd {:d}'.format(
            checked_dcd))
        
        # Write modified input
        with open(inpfile, 'w') as f:
            f.write(inplines)
        
        # Restart simulation
        check_status()
        task = subprocess.run(['sbatch', runfile], capture_output=True)
        tskid = int(task.stdout.split()[-1])
        
        # Reset error flag
        error = False
            
    # If task is still running (and no error in dcd) ...
    if tskid in idslist:
        continue
    else:
        finished = True
        
    # Check if last step is reached
    if maxndcd == checked_dcd + 1:
        finished = True
        
    # Check if current step is complete
    next_dcdfile = "dyna_crd.{:d}.dcd".format(checked_dcd + 1)
    if os.path.exists(next_dcdfile):
        checked_dcd += 1
    
                    
        
        
