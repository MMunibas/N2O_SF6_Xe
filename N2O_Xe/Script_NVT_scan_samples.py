import os
import sys
import numpy as np

import subprocess

import MDAnalysis

from ase import units
from shutil import copyfile


#------------
# Parameters
#------------

# Temperatures [K]
T = [291.18]

# Critical temperature and density of Xe
solvent = 'XE'
M = 131.293 # g/mol
Tcrit = 289.73 # K
rhocrit = 1.10 # g/cm**3

# Relative density
rhostar = [
    [0.04, 0.10, 0.15, 0.37, 0.49, 0.62, 0.66, 0.75, 0.87, 0.93, 1.33, 1.55]]

# Molare Volume
Vm = []
for rhotemp in rhostar:
    Vm.append(M/np.array(rhotemp)/rhocrit)

# Number of solvent molecules
N = 600

# Time step [ps]
dt = 0.001

# Number of sample runs per property
Nsmpl = 10

# Propagation steps
# Heating steps
Nheat = 100000
# NVE run steps
Nnver = 100000
# Equilibration steps
Nequi = 100000
# Production runs and dynamic steps per sample run
Nprod = 100
Ndyna = 100000

# Evaluation
# Vibrational vectors of mode 
Imode = 9
# Number of parallel runs
Nparr = 20

# Number of frame skip
Nskip = 5   # each 5 fs

# Step size for storing coordinates, velocities of N2O
Nsave = [1] + [50]*(Nsmpl - 1)

# FFT points per box edge range
Nfft = {
    (0.0, 36.0): 32,        # 2**5
    (36.0, 96.0): 64,       # 2**6
    (96.0, 192.0): 128      # 2**7
    }

# Number of CPUs - dynamics
cpus = 1

# Number of CPUs -evaluation
cpus_eval = 1

# Memory per CPU
mems = 1200

# Main directory
maindir = os.getcwd()

# Template directory
tempdir = 'template'

# Source directory
sourdir = 'source'

# Source files to be copied in working directory
sourfls = [
    'n2o.top', 'n2o.par', 
    '{:s}.par'.format(solvent.lower()), '{:s}.top'.format(solvent.lower()),
    'pes1_rRz.csv', 
    'crystal_image.str', 
    'n2o.psf', 
    'n2o_{:s}.dcm'.format(solvent.lower())]

#-----------------------------
# Preparations - General
#-----------------------------

# Iterate over system conditions
workdirs = []
abox = []
for it, Ti in enumerate(T):
    
    # Make temperature directory
    if not os.path.exists(os.path.join(maindir, 'T{:d}'.format(int(Ti)))):
        os.mkdir(os.path.join(maindir, 'T{:d}'.format(int(Ti))))
    
    workdirs.append([])
    abox.append([])
    
    for iv, Vmi in enumerate(Vm[it]):
        
        workdirs[it].append([])
        
        for ismpl in range(Nsmpl):
        
            # Make working directory
            workdir = os.path.join(
                maindir, 
                'T{:d}'.format(int(Ti)), 
                'V{:d}_{:d}'.format(int(Vmi), ismpl))
            if not os.path.exists(workdir):
                os.mkdir(workdir)
            workdirs[it][iv].append(workdir)
            
        # Compute simulation box size [AA**3] and edge length [AA]
        VN = Vmi*(1e-2)**3/units.mol
        Vbox = N*VN*(1e10)**3
        abox[it].append(np.cbrt(Vbox))

# Center box size for initial N2O position
cbox = 2.0

#-----------------------------
# Preparations - Packmol
#-----------------------------

# Iterate over system conditions
for it, Ti in enumerate(T):
    
    for iv, Vmi in enumerate(Vm[it]):
        
        for ismpl in range(Nsmpl):
        
            # Working directory
            workdir = workdirs[it][iv][ismpl]
            
            # Simulation box edge length
            aboxi = abox[it][iv]
            
            # Prepare input file - packmol.inp
            ftemp = open(os.path.join(
                maindir, tempdir, 'template_packmol.inp'), 'r')
            inplines = ftemp.read()
            ftemp.close()
            
            # Prepare parameters
            # Output file
            ffile = os.path.join(
                maindir, workdir, 'init.n2o_{:s}.pdb'.format(solvent.lower()))
            inplines = inplines.replace('FFF', ffile)
            # Random seed
            seed = np.random.randint(1e6)
            inplines = inplines.replace('DDD', '{:d}'.format(seed))
            # Source directory
            sdir = os.path.join(maindir, sourdir)
            inplines = inplines.replace('SSS', sdir)
            # N2O Molecule number
            inplines = inplines.replace('NNN1', str(1))
            # Box size range
            rxyz = '{:6.4f} '*6
            rxyz = rxyz.format(
                -cbox/2., -cbox/2., -cbox/2., 
                cbox/2.,  cbox/2.,  cbox/2.)
            inplines = inplines.replace('RRR1', str(rxyz))
            # Solvent molecule number
            inplines = inplines.replace('NNN2', str(N))
            # Solvent name
            inplines = inplines.replace('SLV', solvent.lower())
            # Box size range
            rxyz = '{:6.4f} '*6
            rxyz = rxyz.format(
                -aboxi/2. + 1.0, -aboxi/2. + 1.0, -aboxi/2. + 1.0, 
                aboxi/2. - 1.0,  aboxi/2. - 1.0,  aboxi/2. - 1.0)
            inplines = inplines.replace('RRR2', str(rxyz))
            
            # Write input file
            finp = open(os.path.join(workdir, 'packmol.inp'), 'w')
            finp.write(inplines)
            finp.close()
            
            # Execute packmol
            subprocess.run(
                'cd {:s} ; packmol < packmol.inp'.format(workdir), shell=True)

#-----------------------------
# Preparations - Generation
#-----------------------------

# Iterate over system conditions
for it, Ti in enumerate(T):
    
    for iv, Vmi in enumerate(Vm[it]):
        
        for ismpl in range(Nsmpl):
        
            # Working directory
            workdir = workdirs[it][iv][ismpl]
        
            # Simulation box edge length
            aboxi = abox[it][iv]
            
            for f in sourfls:
                src = os.path.join(maindir, sourdir, f)
                trg = os.path.join(maindir, workdir, f)
                copyfile(src, trg)
            
            # Get single pdb files
            fsys = open(
                os.path.join(
                    workdir, 'init.n2o_{:s}.pdb'.format(solvent.lower())), 'r')
            syslines = fsys.readlines()
            fsys.close()
            
            # N2O
            pdb_n2o = ''
            for line in syslines:
                if 'ATOM' in line:
                    if 'N2O' in line:
                        pdb_n2o += line
                else:
                    pdb_n2o += line  
            fn2o = open(os.path.join(workdir, 'n2o.pdb'), 'w')
            fn2o.write(pdb_n2o)
            fn2o.close()
            
            # Solvent
            pdb_solvent = ''
            for line in syslines:
                if 'ATOM' in line:
                    if solvent in line:
                        pdb_solvent += line
                else:
                    pdb_solvent += line  
            fsolvent = open(
                os.path.join(workdir, '{:s}.pdb'.format(solvent.lower())), 'w')
            fsolvent.write(pdb_solvent)
            fsolvent.close()
            
            # Prepare input file - generate.inp
            ftemp = open(os.path.join(
                maindir, tempdir, 'template_generate.inp'), 'r')
            inplines = ftemp.read()
            ftemp.close()
            
            # Prepare parameters
            # Solvent - lower and upper case
            inplines = inplines.replace('SLVL', solvent.lower())
            inplines = inplines.replace('SLVU', solvent.upper())
            # Initial system tag
            inplines = inplines.replace(
                'ISYS', 'init.n2o_{:s}'.format(solvent.lower()))
            # Box edge length
            inplines = inplines.replace('AAA', '{:6.4f}'.format(aboxi))
            
            # Write input file
            finp = open(os.path.join(workdir, 'generate.inp'), 'w')
            finp.write(inplines)
            finp.close()
 
#-----------------------------
# Preparations - Production
#-----------------------------

# Iterate over system conditions
for it, Ti in enumerate(T):
    
    # Get initial N2O velocities for current temperature
    from source.initial_conditions import get_velocities
    init_vel = get_velocities(
        #j = [0, 0, 0],
        #nu_as = 0,
        #nu_s = 0,
        #nu_d = 0,
        temperature=Ti
        )
    
    for iv, Vmi in enumerate(Vm[it]):
        
        for ismpl in range(Nsmpl):
        
            # Working directory
            workdir = workdirs[it][iv][ismpl]
        
            # Simulation box edge length
            aboxi = abox[it][iv]
            
            # Prepare input file - dyna.inp
            ftemp = open(os.path.join(
                maindir, tempdir, 'template_dyna.inp'), 'r')
            inplines = ftemp.read()
            ftemp.close()
            
            # Prepare parameters
            # Solvent - lower and upper case
            inplines = inplines.replace('SLVL', solvent.lower())
            inplines = inplines.replace('SLVU', solvent.upper())
            # Initial system tag
            inplines = inplines.replace(
                'ISYS', 'init.n2o_{:s}'.format(solvent.lower()))
            # Box edge length
            inplines = inplines.replace('AAA', '{:6.4f}'.format(aboxi))
            # Number of FFT points
            nffti = 32
            for key, item in Nfft.items():
                if aboxi > key[0] and aboxi <= key[1]:
                    nffti = item
                    break
            #nffti = int(aboxi/dfft) + 1
            inplines = inplines.replace('NFT', '{:d}'.format(int(nffti)))
            # N2O molecules
            inplines = inplines.replace('NNN', '{:d}'.format(1))
            # Time step
            inplines = inplines.replace('TDT', '{:5.4f}'.format(dt))
            # Temperature
            inplines = inplines.replace('TMN', '{:d}'.format(100))
            inplines = inplines.replace('TTT', '{:d}'.format(int(Ti)))
            inplines = inplines.replace('THM', '{:d}'.format(100))
            inplines = inplines.replace('FBT', '{:.1f}'.format(0.1))
            # Heating steps
            inplines = inplines.replace('HTS', '{:d}'.format(Nheat))
            # Assign velocities
            inplines = inplines.replace('{LVXN1}', '{:.6f}'.format(
                init_vel[0, 0]))
            inplines = inplines.replace('{LVYN1}', '{:.6f}'.format(
                init_vel[0, 1]))
            inplines = inplines.replace('{LVZN1}', '{:.6f}'.format(
                init_vel[0, 2]))
            inplines = inplines.replace('{LVXN2}', '{:.6f}'.format(
                init_vel[1, 0]))
            inplines = inplines.replace('{LVYN2}', '{:.6f}'.format(
                init_vel[1, 1]))
            inplines = inplines.replace('{LVZN2}', '{:.6f}'.format(
                init_vel[1, 2]))
            inplines = inplines.replace('{LVXO3}', '{:.6f}'.format(
                init_vel[2, 0]))
            inplines = inplines.replace('{LVYO3}', '{:.6f}'.format(
                init_vel[2, 1]))
            inplines = inplines.replace('{LVZO3}', '{:.6f}'.format(
                init_vel[2, 2]))
            # NVE test run steps
            inplines = inplines.replace('CES', '{:d}'.format(Nnver))
            # Equilibration steps
            inplines = inplines.replace('EQS', '{:d}'.format(Nequi))
            # Production runs
            inplines = inplines.replace('NDY', '{:d}'.format(Nprod))
            # Dynamic steps
            inplines = inplines.replace('DYS', '{:d}'.format(Ndyna))
            # Writing interval
            inplines = inplines.replace('NSV', '{:d}'.format(Nsave[ismpl]))
            
            # Write input file
            finp = open(os.path.join(workdir, 'dyna.inp'), 'w')
            finp.write(inplines)
            finp.close()

#-----------------------------
# Prepare Run Script
#-----------------------------        

# Iterate over system conditions
for it, Ti in enumerate(T):
    
    for iv, Vmi in enumerate(Vm[it]):
        
        for ismpl in range(Nsmpl):
        
            # Working directory
            workdir = workdirs[it][iv][ismpl]
        
            # Prepare run file - run.sh
            ftemp = open(os.path.join(
                maindir, tempdir, 'template_run.sh'), 'r')
            inplines = ftemp.read()
            ftemp.close()
            
            # Prepare parameters
            # Solvent
            inplines = inplines.replace('SLVU', solvent.upper())
            # Temperature
            inplines = inplines.replace('TTT', '{:d}'.format(int(Ti)))
            # Volume
            inplines = inplines.replace('VVV', '{:d}'.format(int(Vmi)))
            # CPUs
            inplines = inplines.replace('CCC', '{:d}'.format(cpus))
            # Memory
            inplines = inplines.replace('MMM', '{:d}'.format(mems))
            
            # Write run file
            runfile = os.path.join(workdir, 'run.sh')
            finp = open(runfile, 'w')
            finp.write(inplines)
            finp.close()
            
            # Run job
            subprocess.run(
                'cd {:s} ; sbatch run.sh'.format(workdir), shell=True)

#-----------------------------
# Prepare Observation Script
#-----------------------------        

# Iterate over system conditions
for it, Ti in enumerate(T):
    
    for iv, Vmi in enumerate(Vm[it]):
        
        for ismpl in range(Nsmpl):
            
            # Working directory
            workdir = workdirs[it][iv][ismpl]
        
            # Prepare run file - run.sh
            ftemp = open(os.path.join(
                maindir, tempdir, 'template_observe.py'), 'r')
            inplines = ftemp.read()
            ftemp.close()
            
            # Prepare parameters
            # Run script file
            inplines = inplines.replace('%RFILE%', 'run.sh')
            # Input file
            inplines = inplines.replace('%IFILE%', 'dyna.inp')
            # Output file
            inplines = inplines.replace('%OFILE%', 'dyna.out')
            # Production runs
            inplines = inplines.replace('%MAXND%', '{:d}'.format(Nprod))
            
            # Write observation file
            obsfile = os.path.join(workdir, 'observe.py')
            finp = open(obsfile, 'w')
            finp.write(inplines)
            finp.close()
            
            # Run observation script
            #subprocess.run(
            #    'cd {:s} ; python observe.py &'.format(workdir), shell=True)


#-----------------------------
# Preparations - Forces
#-----------------------------

# Iterate over system conditions
for it, Ti in enumerate(T):
    
    for iv, Vmi in enumerate(Vm[it]):
        
        for ismpl in range(Nsmpl):
            
            for ipar in range(Nparr):
                
                if ipar < 5:
                    continue
                
                # Working directory
                workdir = workdirs[it][iv][ismpl]
            
                # Simulation box edge length
                aboxi = abox[it][iv]
                
                # Prepare input file - forces.inp
                ftemp = open(os.path.join(
                    maindir, tempdir, 'template_forces.inp'), 'r')
                inplines = ftemp.read()
                ftemp.close()
                
                # Prepare parameters
                # Solvent - lower and upper case
                inplines = inplines.replace('SLVL', solvent.lower())
                inplines = inplines.replace('SLVU', solvent.upper())
                # Initial system tag
                inplines = inplines.replace(
                    'ISYS', 'init.n2o_{:s}'.format(solvent.lower()))
                # Box edge length
                inplines = inplines.replace('AAA', '{:6.4f}'.format(aboxi))
                # Number of FFT points
                nffti = 32
                for key, item in Nfft.items():
                    if aboxi > key[0] and aboxi <= key[1]:
                        nffti = item
                        break
                #nffti = int(aboxi/dfft) + 1
                inplines = inplines.replace('NFT', '{:d}'.format(int(nffti)))
                # Write input file
                finp = open(os.path.join(workdir, 'forces.inp'), 'w')
                finp.write(inplines)
                finp.close()

#-----------------------------
# Run Script - Forces
#-----------------------------        

# Iterate over system conditions
for it, Ti in enumerate(T):
    
    for iv, Vmi in enumerate(Vm[it]):
        
        for ismpl in range(Nsmpl):
            
            for ipar in range(Nparr):
                
                if ipar < 5:
                    continue
                
                # Working directory
                workdir = workdirs[it][iv][ismpl]
            
                # Prepare run file - run.sh
                ftemp = open(os.path.join(
                    maindir, tempdir, 'template_forces.sh'), 'r')
                inplines = ftemp.read()
                ftemp.close()
                
                # Prepare parameters
                # Solvent
                inplines = inplines.replace('SLVU', solvent.upper())
                # Temperature
                inplines = inplines.replace('TTT', '{:d}'.format(int(Ti)))
                # Volume
                inplines = inplines.replace('VVV', '{:d}'.format(int(Vmi)))
                # CPUs
                inplines = inplines.replace('CCC', '{:d}'.format(cpus_eval))
                # Memory
                inplines = inplines.replace('MMM', '{:d}'.format(mems))
                # Production runs
                if ipar==0:
                    Nstart = 1
                else:
                    Nstart = int(Nprod*ipar/Nparr) + 1
                Nend = int(Nprod*(ipar + 1)/Nparr)
                inplines = inplines.replace('NDYP', '{:d}'.format(Nprod))
                inplines = inplines.replace('NDYS', '{:d}'.format(Nstart))
                inplines = inplines.replace('NDYE', '{:d}'.format(Nend))
                # Vibrational mode request
                inplines = inplines.replace('VMD', '{:d}'.format(Imode))
                
                # Write run file
                evlfile = os.path.join(workdir, 'forces_{:d}.sh'.format(Nstart))
                finp = open(evlfile, 'w')
                finp.write(inplines)
                finp.close()
                
                # Run job
                #subprocess.run(
                #    'cd {:s} ; sbatch forces_{:d}.sh'.format(
                #        workdir, Nstart), 
                #    shell=True)



#-----------------------------
# Preparations - INM analysis
#-----------------------------

# Iterate over system conditions
for it, Ti in enumerate(T):
    
    for iv, Vmi in enumerate(Vm[it]):
        
        for ismpl in range(Nsmpl):
        
            # Working directory
            workdir = workdirs[it][iv][ismpl]
        
            # Simulation box edge length
            aboxi = abox[it][iv]
            
            # Prepare input file - vibration_n.inp
            ftemp = open(os.path.join(
                maindir, tempdir, 'template_vibration_n.inp'), 'r')
            inplines = ftemp.read()
            ftemp.close()
            
            # Prepare parameters
            inplines = inplines.replace('SLVL', solvent.lower())
            inplines = inplines.replace('SLVU', solvent.upper())
            # Initial system tag
            inplines = inplines.replace(
                'ISYS', 'init.n2o_{:s}'.format(solvent.lower()))
            # Box edge length
            inplines = inplines.replace('AAA', '{:6.4f}'.format(aboxi))
            # Production runs
            inplines = inplines.replace('NDY', '{:d}'.format(Nprod))
            # Dynamic steps
            inplines = inplines.replace('DYS', '{:d}'.format(Ndyna))
            # Writing interval
            inplines = inplines.replace('NSV', '{:d}'.format(Nsave[ismpl]))
            # Skip rate for dcd frames
            inplines = inplines.replace('SKP', '{:d}'.format(Nskip))
            
            
            # Write input file
            finp = open(os.path.join(workdir, 'vibration_n.inp'), 'w')
            finp.write(inplines)
            finp.close()
            
            
            # Prepare run file - vibration_n.sh
            ftemp = open(os.path.join(
                maindir, tempdir, 'template_vibration_n.sh'), 'r')
            inplines = ftemp.read()
            ftemp.close()
            
            # Prepare parameters
            # Solvent
            inplines = inplines.replace('SLVU', solvent.upper())
            # Temperature
            inplines = inplines.replace('TTT', '{:d}'.format(int(Ti)))
            # Volume
            inplines = inplines.replace('VVV', '{:d}'.format(int(Vmi)))
            # CPUs
            inplines = inplines.replace('CCC', '{:d}'.format(1))
            # Memory
            inplines = inplines.replace('MMM', '{:d}'.format(mems))
            
            # Write run file
            finp = open(os.path.join(workdir, 'vibration_n.sh'), 'w')
            finp.write(inplines)
            finp.close()
            
            
            # Prepare control file - vibration.py
            ftemp = open(os.path.join(
                maindir, tempdir, 'template_vibration.py'), 'r')
            inplines = ftemp.read()
            ftemp.close()
            
            # Prepare parameters
            # Production runs
            inplines = inplines.replace('{NDCD}', '{:d}'.format(Nprod))
            # Maximum number of submitted jobs
            inplines = inplines.replace('{NTSK}', '{:d}'.format(Nparr))
            
            # Write control file
            finp = open(os.path.join(workdir, 'vibration.py'), 'w')
            finp.write(inplines)
            finp.close()

