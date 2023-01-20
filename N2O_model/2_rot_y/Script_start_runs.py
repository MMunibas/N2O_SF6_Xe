import os
import sys
import numpy as np

import subprocess

from ase import units
from shutil import copyfile

from source.initial_conditions import get_velocities

#------------------------
# Setup Parameters
#------------------------

# Maximum range of J states
maxJ = None

# Rotational temperature
rot_temp = 321.9

# Vibrational temperature
vib_temp = 321.9

# Rotational axis
# rot_idx: int or None
#   int: rotation around axis of inertia with index
#   None: rotation around all of inertia with index
rot_idx = 1 # rotation around y axis

# Bending direction perpendicular to bond axus
# bend_idx: float [0, 1] or None
#   float: mixing of bending direction defined by axis of inertia
#   None: random mixing of bending direction defined by axis of inertia
bend_idx = 0.0 # bending along x direction


# Number of samples per J state
Nsmpl = 1

# Number of simulation time steps (of 1 fs)
Nstps = 100000

# Working directories
Jdirs = "J{:d}"
smpldirs = "J{:d}_S{:d}"

# Simulation files
srcdir = "source"
inpfile = "dyna.inp"
srcfiles = [
    "n2o.crd",
    "n2o.par",
    "n2o.psf",
    "n2o.top",
    "pes1_rRz.csv",
    "run.sh"
    ]

# Conversion parameter
a02A = units.Bohr
kcalmol2Ha = units.kcal/units.mol/units.Hartree
kcalmol2J = units.kcal/units.mol/units.J
u2kg = units._amu
ms2Afs = 1e-5

u2au = units._amu/units._me
ms2au = units._me*units.Bohr*1e-10*2.0*np.pi/units._hplanck

# Time for speed of light in vacuum to travel 1 cm in fs
jiffy = 0.01/units._c*1e15

#------------------------
# Rotational Parameters
#------------------------

# N2O - RKHS equilibrium data
n2o_atms = ["N", "N", "O"]
n2o_peq = np.array([-1.128620, 0.0, 1.185497])/a02A
n2o_msss = np.array([14.0067, 14.0067, 15.9994])*u2au
n2o_com = np.sum(n2o_peq*n2o_msss)/np.sum(n2o_msss)
n2o_re = np.abs(n2o_peq - n2o_com)
n2o_I = np.sum(n2o_msss*n2o_re**2)
n2o_B = 1.0**2/(2.0*n2o_I)
Jmax = 1000
if maxJ is not None:
    if maxJ*2 > Jmax:
        Jmax = maxJ*2
n2o_J = np.arange(Jmax, dtype=float)
n2o_EJ = n2o_B*n2o_J*(n2o_J + 1)

# Boltzmann constant in atomic units
kB = units.kB/units.Hartree

# Boltzmann-Maxwell distribution function
def boltzmann(J, T, E=n2o_EJ):
    
    # Rotational quantum number
    J = np.arange(len(E))
    
    # State function
    Z = sum([
        (2*Ji + 1)*np.exp(-(E[Ji] - E[0])/(kB*T)) for Ji in J])
    
    # Boltzmann-Maxwell distribution
    D = [(2*Ji + 1)*np.exp(-(E[Ji] - E[0])/(kB*T))/Z for Ji in J]
    
    return D

P_J = boltzmann(n2o_J, rot_temp)

#------------------------
# Setup Sample Runs
#------------------------

if maxJ is None:
    
    if rot_temp == 0.0:
        Ji = 0
    else:
        Ji = int(np.argmax(P_J)*1.0)
    
    # Create working directory
    workdir = Jdirs.format(Ji)
    if not os.path.exists(workdir):
        os.mkdir(workdir)
    
    # Iterate over samples
    for ismpl in range(Nsmpl):
        
        # Create working directory
        workdir = os.path.join(Jdirs.format(Ji), smpldirs.format(Ji, ismpl))
        if not os.path.exists(workdir):
            os.mkdir(workdir)
            
        # Get initial conditions
        init_vel_lines = get_velocities(
            j = Ji,
            rot_idx = rot_idx,    
            nu_as = None,
            nu_s = None,
            nu_d = None,
            nu_d_idx = bend_idx,
            temperature=vib_temp,
            )
        
        # Copy files
        for f in srcfiles:
            src = os.path.join(srcdir, f)
            trg = os.path.join(workdir, f)
            if not os.path.exists(trg):
                copyfile(src, trg)
        
        # Prepare input file - dyna.inp
        with open(os.path.join(srcdir, inpfile), 'r') as f:
            inplines = f.read()
        
        # Prepare parameters
        inplines = inplines.replace("NSTP", "{:d}".format(Nstps))
        inplines = inplines.replace("SIC", "{:s}".format(init_vel_lines))
        
        # Write input file - dyna.inp
        with open(os.path.join(workdir, inpfile), 'w') as f:
            f.write(inplines)
            
        # Start sample run
        subprocess.run(
            'cd {:s} ; sbatch run.sh'.format(workdir), shell=True)
    
    
    
else:

    # Iterate over J states
    for Ji in range(maxJ):
        
        # Create working directory
        workdir = Jdirs.format(Ji)
        if not os.path.exists(workdir):
            os.mkdir(workdir)
        
        # Iterate over samples
        for ismpl in range(Nsmpl):
        
            # Create working directory
            workdir = os.path.join(Jdirs.format(Ji), smpldirs.format(Ji, ismpl))
            if not os.path.exists(workdir):
                os.mkdir(workdir)
            
            # Get initial conditions
            init_vel_lines = get_velocities(
                j = Ji,
                rot_idx = rot_idx,
                nu_as = None,
                nu_s = None,
                nu_d = None,
                nu_d_idx = bend_idx,
                temperature=vib_temp,
                )
            
            # Copy files
            for f in srcfiles:
                src = os.path.join(srcdir, f)
                trg = os.path.join(workdir, f)
                if not os.path.exists(trg):
                    copyfile(src, trg)
            
            # Prepare input file - dyna.inp
            with open(os.path.join(srcdir, inpfile), 'r') as f:
                inplines = f.read()
            
            # Prepare parameters
            inplines = inplines.replace("NSTP", "{:d}".format(Nstps))
            inplines = inplines.replace("SIC", "{:s}".format(init_vel_lines))
            
            # Write input file - dyna.inp
            with open(os.path.join(workdir, inpfile), 'w') as f:
                f.write(inplines)
                
            # Start sample run
            subprocess.run(
                'cd {:s} ; sbatch run.sh'.format(workdir), shell=True)
        
        

        
        
