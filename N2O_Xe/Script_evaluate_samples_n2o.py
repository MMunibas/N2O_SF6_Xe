# Basics
import os
import sys
import numpy as np
from glob import glob
import scipy
from scipy.optimize import curve_fit

# ASE - Basics
from ase import Atoms

# MDAnalysis
import MDAnalysis

# Statistics
from statsmodels.tsa.stattools import acovf

# Multiprocessing

from multiprocessing import Pool
# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Miscellaneous
import ase.units as units
from ase.visualize import view
import pickle as pkl
from scipy.io import FortranFile

#-----------------------------
# Parameters
#-----------------------------

# Number of parallel tasks
tasks = 4

# Requested jobs
request_reading = True
request_reappending = True
request_sequence = True
request_IRspectra = True
recalc_IRspectra = True

# Temperatures [K]
T = [291.18]

# Critical temperature and density of Xe
M = 131.293 # g/mol
Tcrit = 289.73 # K
rhocrit = 1.10 # g/cm**3

# Relative density
rhostar = [[0.04, 0.10, 0.15, 0.37, 0.49, 0.62, 0.66, 0.75, 0.87, 0.93, 1.33, 1.55]]

# Experimental rotational lifetime
rho_exp =      [0.04, 0.10, 0.15, 0.37, 0.49, 0.62, 0.66, 0.75, 0.87, 0.93, 1.33]
con_exp =      [0.37, 0.81, 1.26, 3.11, 4.12, 5.21, 5.60, 6.31, 7.30, 7.81, 11.17]
tau_exp =      [40.0, 22.8, 16.8,  5.8,  4.4,  3.0,  1.9,  1.2,  1.0,  0.8, 0.7]
err_exp =      [ 9.1,  1.8,  1.4,  0.7,  0.1,  0.3,  0.2,  0.2,  0.15,  0.1, 0.1]
tau_coll_exp = [19.9, 10.2,  6.6,  2.7,  2.0,  1.6,  1.5,  1.3,  1.2,  1.1,  0.8]
Texp = 291.2

# Experimental IR spectra lines for N2O in gas phase (P-, Q- and R-branch)
exp_ir = [2211, 2221, 2236]

# Molare Volume
Vm = []
for rhotemp in rhostar:
    Vm.append(M/np.array(rhotemp)/rhocrit)

# Concentration
concrit = 1000.*rhocrit/M
CM = [np.array(rhostar_i)*concrit for rhostar_i in rhostar]

# System information
sys_trgt = 'N2O'
sys_solv = 'XE'
sys_resi = ['N2O', 'XE']
sys_Natm = {
    'N2O': 3,
    'SF6': 7,
    'XE': 1}
sys_tag = {
    'N2O': r'N$_2$O',
    'SF6': r'SF$_6$',
    'XE': 'Xe'}
sys_bnds = [[0, 1], [1, 2], [0, 1, 2]]

# Main directory
maindir = os.getcwd()

# Result directory
rsltdir = 'results'

# Trajectory source
sys_fdcd = 'dyna_crd.*.dcd'
sys_scrd = ['.', 1]
sys_fvel = 'dyna_vel.*.dcd'
sys_svel = ['.', 1]

# System psf file
traj_psffile = 'init.n2o_xe.psf'

# System coord file
sys_fcrd = 'init.n2o_xe.crd'

# Fixed Time step if needed
fixdt = None
#fixdt = {
    #tuple(0): 0.002,
    #tuple(range(1,10)): 0.050}

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

# Fontsize
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

dpi = 200

color_scheme = [
    'b', 'r', 'g', 'purple', 'orange', 'magenta', 'brown', 'darkblue',
    'darkred', 'darkgreen', 'darkgrey', 'olive']


#-----------------------------
# Preparations
#-----------------------------

# Sample runs for temperature/molar volume conditions
Smpls = {}
# Frame number list per sample runs for temperature/molar volume conditions
Nframes_list = {}

# Get residue and atom information
# Number of residue
sys_Nres = {}
# Atom indices of residues
sys_ires = {}

# Iterate over systems
for it, Ti in enumerate(T):
    
    # Add temperature to sample and frame number list
    Smpls[Ti] = {}
    Nframes_list[Ti] = {}
    
    for iv, Vmi in enumerate(Vm[it]):
        
        # Add molar volume to frame number list
        Nframes_list[Ti][Vmi] = []
        
        # Detect samples
        smpldirs = glob(
            os.path.join(
                maindir, 
                'T{:d}'.format(int(Ti)), 
                'V{:d}_*'.format(int(Vmi))))
        ismpls = [int(smpldir.split("_")[-1]) for smpldir in smpldirs]
        
        # Add sample numbers to list
        Smpls[Ti][Vmi] = ismpls
        
        # Working directory
        workdir = os.path.join(
            maindir, 
            'T{:d}'.format(int(Ti)), 
            'V{:d}_{:d}'.format(int(Vmi), Smpls[Ti][Vmi][0]))
        
        # System tag
        tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
    
        # Read residue information
        list_lres = {}
        sys_Nres[tag] = {}
        for ires, res in enumerate(sys_resi):
            info_res = []
            with open(os.path.join(workdir, sys_fcrd), 'r') as f:
                for line in f:
                    if res in line and not '*'==line[0]:
                        info_res.append(line)
            list_lres[res] = info_res
            sys_Nres[tag][res] = int(len(info_res)/sys_Natm[res])
            
        # Get residue atom numbers
        sys_ires[tag] = {}
        for ires, res in enumerate(sys_resi):
            atomsinfo = np.zeros(
                [sys_Nres[tag][res], sys_Natm[res]], dtype=int)
            for ir in range(sys_Nres[tag][res]):
                ri = sys_Natm[res]*ir
                for ia in range(sys_Natm[res]):
                    info = list_lres[res][ri + ia].split()
                    atomsinfo[ir, ia] = int(info[0]) - 1
            sys_ires[tag][res] = atomsinfo
            
if not os.path.exists(os.path.join(maindir, rsltdir)):
    os.mkdir(os.path.join(maindir, rsltdir))

#-----------------------------
# Read trajectories
#-----------------------------

# Temperature and molar volume combinations
T_Vm_Smpls = [
    [Ti, Vmi, ismpl]
    for it, Ti in enumerate(T) 
    for iv, Vmi in enumerate(Vm[it])
    for ismpl in Smpls[Ti][Vmi]]

def read(i):
    
    # Temperature and molar volume
    Ti = T_Vm_Smpls[i][0]
    Vmi = T_Vm_Smpls[i][1]
    ismpl = T_Vm_Smpls[i][2]

    # Working directory
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(Ti)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
    
    # Get dcd files
    dcdfiles = np.array(glob(os.path.join(workdir, sys_fdcd)))
    iruns = np.array([
        int(dcdfile.split('/')[-1].split(sys_scrd[0])[sys_scrd[1]])
        for dcdfile in dcdfiles])
    velfiles = np.array(glob(os.path.join(workdir, sys_fvel)))
    ivels = np.array([
        int(velfile.split('/')[-1].split(sys_svel[0])[sys_svel[1]])
        for velfile in velfiles])
    psffile = os.path.join(workdir, traj_psffile)
    
    # Sort dcd files
    dcdsort = np.argsort(iruns)
    dcdfiles = dcdfiles[dcdsort]
    iruns = iruns[dcdsort]
    velsort = np.argsort(ivels)
    velfiles = velfiles[velsort]
    
    # Initialize trajectory time counter in ps
    traj_time_dcd = 0.0
    
    # Iterate over dcd files
    to_small_once = False
    for ir, idcd in enumerate(iruns):
        
        # Distance file - x(atom-COM)
        xfile_i = os.path.join(
            workdir, 'dists_{:s}_{:d}.npy'.format(tag, idcd))
        
        # Bond distance file
        rfile_i = os.path.join(
            workdir, 'bonds_{:s}_{:d}.npy'.format(tag, idcd))
        
        # Spherical angles file
        sfile_i = os.path.join(
            workdir, 'sphrc_{:s}_{:d}.npy'.format(tag, idcd))
        
        # Angular momentum file
        lfile_i = os.path.join(
            workdir, 'angmo_{:s}_{:d}.npy'.format(tag, idcd))
        
        # Angular velocity file
        wfile_i = os.path.join(
            workdir, 'velos_{:s}_{:d}.npy'.format(tag, idcd))
        
        # Dipole file
        dfile_i = os.path.join(
            workdir, 'dipos_{:s}_{:d}.npy'.format(tag, idcd))
        
        # Time file
        tfile_i = os.path.join(
            workdir, 'times_{:s}_{:d}.npy'.format(tag, idcd))
        
        to_small = False
        if os.path.exists(xfile_i):
            xlst = np.load(xfile_i)
            if ir == 0:
                nsize = xlst.shape[0]
            else:
                if xlst.shape[0] < nsize:
                    to_small = True
                    to_small_once = True
        
        if not os.path.exists(xfile_i) or to_small or to_small_once or False:
            
            # Distances list
            xlst = []
            
            # Bond distances list
            rlst = []
            
            # Spherical angles list
            slst = []
            
            # Angular momentum list
            llst = []
            
            # Angular velocities list
            wlst = []
            
            # Dipoles list
            dlst = []
            
            # Times list
            tlst = []
            
            # Open dcd file
            dcdfile = dcdfiles[ir]
            dcd = MDAnalysis.Universe(psffile, dcdfile)
            
            # Open velocity file
            fvel = FortranFile(velfiles[ir])
            header = fvel.read_reals(np.int32)
            _ = fvel.read_reals(np.float32)
            Natoms = fvel.read_reals(np.int32)[0]
            
            # Get trajectory parameter
            Nframes = len(dcd.trajectory)
            Nskip = int(dcd.trajectory.skip_timestep)
            dt = np.round(
                float(dcd.trajectory._ts_kwargs['dt']), decimals=8)/Nskip
            if fixdt is not None:
                dt = fixdt
            
            # Get masses
            masses = dcd._topology.masses.values
            
            # Get atom charges
            charges = dcd._topology.charges.values
            charges = np.array(
                [-0.0852103666, -0.8646406122+2.3068475322, 0.9673935417-2.3243900952] 
                + [0.0]*sys_Nres[tag]['XE'])
            
            # Iterate over frames
            for ii, frame in enumerate(dcd.trajectory):
                
                # Update time
                traj_time = traj_time_dcd + Nskip*dt*ii*1e3
                
                # Initialize frame result arrays
                Ntrgt = sys_Nres[tag][sys_trgt]
                Natms = sys_Natm[sys_trgt]
                Nbnds = len(sys_bnds)
                xarr = np.zeros([Ntrgt, Natms], dtype=np.float32)
                rarr = np.zeros([Ntrgt, Nbnds], dtype=np.float32)
                sarr = np.zeros([Ntrgt, Natms, 2], dtype=np.float32)
                larr = np.zeros([Ntrgt, 3], dtype=np.float32)
                warr = np.zeros([Ntrgt, Natms], dtype=np.float32)
                darr = np.zeros([Ntrgt, 3], dtype=np.float32)
                
                # Initialize velocity array
                vel = np.zeros([3, Natoms], dtype=np.float32)
                
                # Get cell information
                cell = frame._unitcell/a02A
                if cell[0]!=0.0:
                    box = fvel.read_reals(float)/a02A
                    
                # Atom velocity x, y, z
                vel[0, :] = fvel.read_reals(np.float32)
                vel[1, :] = fvel.read_reals(np.float32)
                vel[2, :] = fvel.read_reals(np.float32)
                vel = vel.T
                
                # Iterate over target molecules
                for itrgt in range(Ntrgt):
                    
                    # Atom indices
                    indcs = sys_ires[tag][sys_trgt][itrgt]
                    
                    # Atom positions (atom, cart) in Bohr
                    pstns = frame._pos[indcs]/a02A
                    
                    # Atom velocities (atom, cart) in m/s #A/fs #Bohr/fs
                    vlcts = vel[indcs]
                    vlcts *= np.sqrt(kcalmol2J/u2kg)#*ms2Afs#/a02A
                    
                    # Atom velocities (atom, cart) in a.u.
                    vlcts *= ms2au
                    
                    # Atom masses (atom) in a.u. (me)
                    m = masses[indcs]*u2au
                    
                    # Atom momentum (atom, cart) in a.u. #u*A/fs
                    mntms = m.reshape(-1, 1)*vlcts
                    
                    # Center of Mass
                    com = (
                        np.sum(m.reshape(-1, 1)*pstns, axis=0)
                        /np.sum(m))
                    
                    # Center of Mass velocity
                    pcom = np.sum(mntms, axis=0)
                    vcom = pcom/np.sum(m)
                    
                    # Correct momentum and velocities
                    mntms -= (
                        m.reshape(-1, 1)
                        *pcom/np.sum(m))
                    vlcts = mntms/m.reshape(-1, 1)
                    
                    # Distances in Bohr
                    dstns = np.sqrt(np.sum((pstns - com)**2, axis=1))
                    xarr[itrgt, :] = dstns
                    
                    # Bond distances and angles
                    for ib, bnds in enumerate(sys_bnds):
                        if len(bnds) == 2:
                            rarr[itrgt, ib] = np.sqrt(np.sum(
                                (pstns[bnds[0]] - pstns[bnds[1]])**2))
                        elif len(bnds) == 3:
                            a = pstns[bnds[1]] - pstns[bnds[0]]
                            a = a/np.sqrt(np.sum(a**2))
                            b = pstns[bnds[1]] - pstns[bnds[2]]
                            b = b/np.sqrt(np.sum(b**2))
                            rarr[itrgt, ib] = np.dot(a, b)
                        else:
                            raise ValueError("Wrong system bond definition!")
                    
                    # Spherical angles
                    for ia in range(Natms):
                        vec = pstns[ia] - com
                        theta = np.arccos(vec[2]/dstns[ia])
                        phi = np.arctan2(vec[1], vec[0])
                        if phi < 0:
                            phi = 2.0*np.pi + phi
                        sarr[itrgt, ia, 0] = theta
                        sarr[itrgt, ia, 1] = phi
                        
                    # Angular momentum a.u. #u*radians/fs
                    angmo = np.sum(
                        np.cross(
                            pstns - com, 
                            mntms), 
                        axis=0)
                    larr[itrgt, :] = angmo
                    
                    # Angular velocities a.u. #radians/fs
                    wvlct = np.cross(pstns - com, vlcts)
                    wvlct = np.sqrt(np.sum(wvlct**2, axis=1))
                    warr[itrgt, :] = wvlct
                    
                    # Dipoles
                    dples = np.dot(charges[indcs], pstns)
                    darr[itrgt, :] = dples
                    
                # Append frame results
                xlst.append(xarr)
                rlst.append(rarr)
                slst.append(sarr)
                llst.append(larr)
                wlst.append(warr)
                dlst.append(darr)
                tlst.append(traj_time)
            
            # Update trajectory time
            traj_time_dcd = traj_time + Nskip*dt*1e3
            
            # Close velocity file
            fvel.close()
        
            # Save dcd results
            np.save(xfile_i, np.array(xlst, dtype=np.float32))
            np.save(rfile_i, np.array(rlst, dtype=np.float32))
            np.save(sfile_i, np.array(slst, dtype=np.float32))
            np.save(lfile_i, np.array(llst, dtype=np.float32))
            np.save(wfile_i, np.array(wlst, dtype=np.float32))
            np.save(dfile_i, np.array(dlst, dtype=np.float32))
            np.save(tfile_i, np.array(tlst, dtype=np.float32))
            
        else:
            
            # Load trajectory time
            try:
                
                tlst = np.load(tfile_i)
                
            except (ValueError,AttributeError):
                
                # Get dt
                dcdfile = dcdfiles[ir]
                dcd = MDAnalysis.Universe(psffile, dcdfile)
                Nskip = int(dcd.trajectory.skip_timestep)
                dt = np.round(
                    float(dcd.trajectory._ts_kwargs['dt']), decimals=8)/Nskip
                if fixdt is not None:
                    dt = fixdt
                dt = dt*1e3
                
                # Regain time list
                xlst = np.load(xfile_i)
                tlst = (
                    np.arange(xlst.shape[0])*Nskip*dt 
                    + traj_time_dcd + dt*Nskip)
            
            # Set tlst start to traj_time_dcd
            tlst = tlst - tlst[0] + traj_time_dcd
            
            # Save trajectory time
            np.save(tfile_i, np.array(tlst, dtype=np.float32))
            
            # Update trajectory time
            traj_time_dcd = 2.0*tlst[-1] - tlst[-2]
            

if request_reading and tasks==1:
    for i in range(0, len(T_Vm_Smpls)):
        read(i)
elif request_reading:    
    if __name__ == '__main__':
        pool = Pool(tasks)
        pool.imap(read, range(0, len(T_Vm_Smpls)))
        pool.close()
        pool.join()


# Append results

for it, Ti in enumerate(T):
    
    for iv, Vmi in enumerate(Vm[it]):
        
        for ismpl in Smpls[Ti][Vmi]:
            
            # Working directory
            workdir = os.path.join(
                maindir, 
                'T{:d}'.format(int(Ti)), 
                'V{:d}_{:d}'.format(int(Vmi), ismpl))
            
            # System tag
            tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
            
            # Get dcd files
            dcdfiles = np.array(glob(os.path.join(workdir, sys_fdcd)))
            iruns = np.array([
                int(dcdfile.split('/')[-1].split(sys_scrd[0])[sys_scrd[1]])
                for dcdfile in dcdfiles])
            velfiles = np.array(glob(os.path.join(workdir, sys_fvel)))
            
            # Sort dcd files
            dcdsort = np.argsort(iruns)
            dcdfiles = dcdfiles[dcdsort]
            iruns = iruns[dcdsort]
            
            # Distance file - x(atom-COM)
            xfile = os.path.join(
                workdir, 'dists_{:s}.npy'.format(tag))
            
            # Bond distance file
            rfile = os.path.join(
                workdir, 'bonds_{:s}.npy'.format(tag))
            
            # Spherical angles file
            sfile = os.path.join(
                workdir, 'sphrc_{:s}.npy'.format(tag))
            
            # Angular momentum file
            lfile = os.path.join(
                workdir, 'angmo_{:s}.npy'.format(tag))
            
            # Angular velocity file
            wfile = os.path.join(
                workdir, 'velos_{:s}.npy'.format(tag))
            
            # Dipole file
            dfile = os.path.join(
                workdir, 'dipos_{:s}.npy'.format(tag))
            
            # Time file
            tfile = os.path.join(
                workdir, 'times_{:s}.npy'.format(tag))
            
            if not os.path.exists(xfile) or request_reappending:
                
                xfile_0 = os.path.join(
                    workdir, 'dists_{:s}_{:d}.npy'.format(tag, 0))
                if not os.path.exists(xfile_0):
                    print("Skip results for T{:d}/V{:d}_{:d}".format(
                        int(Ti), int(Vmi), ismpl))
                    continue
                else:
                    print("Append results for T{:d}/V{:d}_{:d}".format(
                        int(Ti), int(Vmi), ismpl))
                    
                # Distances list
                tmplst = []
                for ir, idcd in enumerate(iruns):
                    xfile_i = os.path.join(
                        workdir, 'dists_{:s}_{:d}.npy'.format(tag, idcd))
                    if os.path.exists(xfile_i):
                        tmplst.append(np.load(xfile_i))
                    #os.remove(xfile_i)
                if len(tmplst):
                    np.save(xfile, np.concatenate(tmplst))
                
                # Bond distances list
                tmplst = []
                for ir, idcd in enumerate(iruns):
                    rfile_i = os.path.join(
                        workdir, 'bonds_{:s}_{:d}.npy'.format(tag, idcd))
                    if os.path.exists(rfile_i):
                        tmplst.append(np.load(rfile_i))
                    #os.remove(rfile_i)
                if len(tmplst):
                    np.save(rfile, np.concatenate(tmplst))
                
                # Spherical angles list
                tmplst = []
                for ir, idcd in enumerate(iruns):
                    sfile_i = os.path.join(
                        workdir, 'sphrc_{:s}_{:d}.npy'.format(tag, idcd))
                    if os.path.exists(sfile_i):
                        tmplst.append(np.load(sfile_i))
                    #os.remove(sfile_i)
                if len(tmplst):
                    np.save(sfile, np.concatenate(tmplst))
                
                # Angular momentum list
                tmplst = []
                for ir, idcd in enumerate(iruns):
                    lfile_i = os.path.join(
                        workdir, 'angmo_{:s}_{:d}.npy'.format(tag, idcd))
                    if os.path.exists(lfile_i):
                        tmplst.append(np.load(lfile_i))
                    #os.remove(lfile_i)
                if len(tmplst):
                    np.save(lfile, np.concatenate(tmplst))
                
                # Angular velocities list
                tmplst = []
                for ir, idcd in enumerate(iruns):
                    wfile_i = os.path.join(
                        workdir, 'velos_{:s}_{:d}.npy'.format(tag, idcd))
                    if os.path.exists(wfile_i):
                        tmplst.append(np.load(wfile_i))
                    #os.remove(wfile_i)
                if len(tmplst):
                    np.save(wfile, np.concatenate(tmplst))
                
                # Dipoles list
                tmplst = []
                for ir, idcd in enumerate(iruns):
                    dfile_i = os.path.join(
                        workdir, 'dipos_{:s}_{:d}.npy'.format(tag, idcd))
                    if os.path.exists(dfile_i):
                        tmplst.append(np.load(dfile_i))
                    #os.remove(dfile_i)
                if len(tmplst):
                    np.save(dfile, np.concatenate(tmplst))
                
                # Times list
                tmplst = []
                for ir, idcd in enumerate(iruns):
                    tfile_i = os.path.join(
                        workdir, 'times_{:s}_{:d}.npy'.format(tag, idcd))
                    if os.path.exists(tfile_i):
                        tmplst.append(np.load(tfile_i))
                    #os.remove(tfile_i)
                if len(tmplst):
                    tmplst = np.concatenate(tmplst)
                    tmax = tmplst[-1]
                    dt = tmplst[-1] - tmplst[-2]
                    np.save(tfile, tmplst)
                    
                    print(
                        "Done - t = {:.3f} ps (dt = {:.3f} ps) in total".format(
                            tmax, dt))
                
            # Open time list file
            tmplst = np.load(tfile)
            
            # Add frame number to list
            Nframes_list[Ti][Vmi].append(len(tmplst))

#-----------------------------
# Plot Sequence
#-----------------------------

# Figure arrangement
figsize = (12, 8)
left = 0.12
bottom = 0.15
row = np.array([0.22, 0.03])
column = np.array([0.83, 0.10])

# Iterate over systems
for it, Ti in enumerate(T):
    
    # Plot absorption spectra
    fig = plt.figure(figsize=figsize)
    
    # Initialize axes
    axs1 = fig.add_axes([left, bottom + 2*np.sum(row), column[0], row[0]])
    axs2 = fig.add_axes([left, bottom + 1*np.sum(row), column[0], row[0]])
    axs3 = fig.add_axes([left, bottom + 0*np.sum(row), column[0], row[0]])
    
    fig.suptitle(
        r'$\Theta$ sequence of N(0) atom of N$_2$O in {:s} at '.format(
            sys_tag[sys_solv])
        + '{:.1f} K\n'.format(float(Ti)), 
        fontweight='bold')
    
    # Lowest
    iv = 0
    Vmi = Vm[it][iv]
    ismpl = Smpls[Ti][Vmi][0]
            
    # Working directory
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(Ti)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
    
    # Spherical angles file
    sfile = os.path.join(
        workdir, 'sphrc_{:s}.npy'.format(tag))
    
    # Time file
    tfile = os.path.join(
        workdir, 'times_{:s}.npy'.format(tag))
    
    # Load results
    tlst = np.load(tfile)
    select = tlst < 50.0*1e3
    slst = np.load(sfile)[select]
    tlst = tlst[select]
    
    # Pressure ratio
    p = M/Vmi/rhocrit
    
    # Plot
    label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
    axs1.plot(
        tlst*1e-3, slst[:, 0, 0, 0]*180.0/np.pi, '-', 
        color=color_scheme[iv], label=label)
    
    axs1.set_xlim([0, 50.0])
    axs1.set_ylim([0, 180.])
    
    axs1.set_xticklabels([])
    
    axs1.legend(
        loc='upper right', 
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
    
    
    
    
    # Middle
    iv = len(Vm[it])//2
    Vmi = Vm[it][iv]
    ismpl = Smpls[Ti][Vmi][0]
            
    # Working directory
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(Ti)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
    
    # Spherical angles file
    sfile = os.path.join(
        workdir, 'sphrc_{:s}.npy'.format(tag))
    
    # Time file
    tfile = os.path.join(
        workdir, 'times_{:s}.npy'.format(tag))
    
    # Load results
    tlst = np.load(tfile)
    select = tlst < 50.0*1e3
    slst = np.load(sfile)[select]
    tlst = tlst[select]
    
    # Pressure ratio
    p = M/Vmi/rhocrit
    
    # Plot
    label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
    select = tlst < 50.0*1e3
    axs2.plot(
        tlst*1e-3, slst[:, 0, 0, 0]*180.0/np.pi, '-', 
        color=color_scheme[iv], label=label)
    
    axs2.set_xlim([0, 50.0])
    axs2.set_ylim([0, 180.])
    
    axs2.set_xticklabels([])
    
    axs2.set_ylabel(r'$\Theta (^\circ)$', fontweight='bold')
    axs2.get_yaxis().set_label_coords(-0.10, 0.50)
    
    axs2.legend(
        loc='upper right', 
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
        
    
    
    # Highest
    iv = len(Vm[it]) - 1
    Vmi = Vm[it][iv]
    ismpl = Smpls[Ti][Vmi][0]
            
    # Working directory
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(Ti)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
    
    # Spherical angles file
    sfile = os.path.join(
        workdir, 'sphrc_{:s}.npy'.format(tag))
    
    # Time file
    tfile = os.path.join(
        workdir, 'times_{:s}.npy'.format(tag))
    
    # Load results
    tlst = np.load(tfile)
    select = tlst < 50.0*1e3
    slst = np.load(sfile)[select]
    tlst = tlst[select]
    
    # Pressure ratio
    p = M/Vmi/rhocrit
    
    # Plot
    label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
    select = tlst < 50.0*1e3
    axs3.plot(
        tlst*1e-3, slst[:, 0, 0, 0]*180.0/np.pi, '-', 
        color=color_scheme[iv], label=label)
    
    axs3.set_xlim([0, 50.0])
    axs3.set_ylim([0, 180.])
    
    axs3.set_xlabel(r'Time (ps)', fontweight='bold')
    axs3.get_xaxis().set_label_coords(0.5, -0.3)
    
    axs3.legend(
        loc='upper right', 
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
    
    plt.savefig(
        os.path.join(
            maindir, rsltdir, 'Seq_spec_{:d}K.png'.format(int(Ti))),
        format='png', dpi=dpi)
    plt.close()
    


# Figure arrangement
figsize = (12, 8)
left = 0.12
bottom = 0.15
row = np.array([0.22, 0.03])
column = np.array([0.83, 0.10])

# Iterate over systems
for it, Ti in enumerate(T):
    
    # Skip if not requested
    if not request_sequence:
        continue
    
    # Plot absorption spectra
    fig = plt.figure(figsize=figsize)
    
    # Initialize axes
    axs1 = fig.add_axes([left, bottom + 2*np.sum(row), column[0], row[0]])
    axs2 = fig.add_axes([left, bottom + 1*np.sum(row), column[0], row[0]])
    axs3 = fig.add_axes([left, bottom + 0*np.sum(row), column[0], row[0]])
    
    fig.suptitle(
        r'$|\vec{L}|$ sequence of N$_2$O in ' + '{:s} at '.format(
            sys_tag[sys_solv])
        + '{:.1f} K\n'.format(float(Ti)), 
        fontweight='bold')
    
    # Lowest
    iv = 2  
    Vmi = Vm[it][iv]
    ismpl = Smpls[Ti][Vmi][0]
            
    # Working directory
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(Ti)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
    
    # Angular momentum file
    lfile = os.path.join(
        workdir, 'angmo_{:s}.npy'.format(tag))
    
    # Time file
    tfile = os.path.join(
        workdir, 'times_{:s}.npy'.format(tag))
    
    # Load results
    tlst = np.load(tfile)
    select = tlst < 50.0*1e3
    llst = np.sqrt(np.sum(np.load(lfile)[select]**2, axis=2))
    tlst = tlst[select]
    
    # Pressure ratio
    p = M/Vmi/rhocrit
    
    # Plot
    label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
    select = tlst < 50.0*1e3
    axs1.plot(
        tlst*1e-3, llst[:, 0], '-', 
        color=color_scheme[iv], label=label)
    
    axs1.set_xlim([0, 50.0])
    axs1.set_xticklabels([])
    
    axs1.legend(
        loc='upper right', 
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
    
    
    
    
    # Middle
    iv = len(Vm[it]) - 3
    Vmi = Vm[it][iv]
    ismpl = Smpls[Ti][Vmi][0]
            
    # Working directory
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(Ti)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
    
    # Angular momentum file
    lfile = os.path.join(
        workdir, 'angmo_{:s}.npy'.format(tag))
    
    # Time file
    tfile = os.path.join(
        workdir, 'times_{:s}.npy'.format(tag))
    
    # Load results
    tlst = np.load(tfile)
    select = tlst < 50.0*1e3
    llst = np.sqrt(np.sum(np.load(lfile)[select]**2, axis=2))
    tlst = tlst[select]
    
    # Pressure ratio
    p = M/Vmi/rhocrit
    
    # Plot
    label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
    select = tlst < 50.0*1e3
    axs2.plot(
        tlst*1e-3, llst[:, 0], '-', 
        color=color_scheme[iv], label=label)
    
    
    axs2.set_xlim([0, 50.0])
    #axs2.set_ylim([0, 180.])
    
    axs2.set_xticklabels([])
    
    axs2.set_ylabel(r'$\left| \vec{L} \right|$ (a.u.)', fontweight='bold')
    axs2.get_yaxis().set_label_coords(-0.10, 0.50)
    
    axs2.legend(
        loc='upper right', 
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
        
    
    
    # Highest
    iv = len(Vm[it]) - 1
    Vmi = Vm[it][iv]
    ismpl = Smpls[Ti][Vmi][0]
            
    # Working directory
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(Ti)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
    
    # Angular momentum file
    lfile = os.path.join(
        workdir, 'angmo_{:s}.npy'.format(tag))
    
    # Time file
    tfile = os.path.join(
        workdir, 'times_{:s}.npy'.format(tag))
    
    # Load results
    tlst = np.load(tfile)
    select = tlst < 50.0*1e3
    llst = np.sqrt(np.sum(np.load(lfile)[select]**2, axis=2))
    tlst = tlst[select]
    
    # Pressure ratio
    p = M/Vmi/rhocrit
    
    # Plot
    label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
    select = tlst < 50.0*1e3
    axs3.plot(
        tlst*1e-3, llst[:, 0], '-', 
        color=color_scheme[iv], label=label)
    
    
    axs3.set_xlim([0, 50.0])
    #axs3.set_ylim([0, 180.])
    
    axs3.set_xlabel(r'Time (ps)', fontweight='bold')
    axs3.get_xaxis().set_label_coords(0.5, -0.3)
    
    axs3.legend(
        loc='upper right', 
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
    
    plt.savefig(
        os.path.join(
            maindir, rsltdir, 'Seq_angmom_{:d}K.png'.format(int(Ti))),
        format='png', dpi=dpi)
    plt.close()
    
    
# Figure arrangement
figsize = (12, 8)
left = 0.12
bottom = 0.15
row = np.array([0.22, 0.03])
column = np.array([0.83, 0.10])

# Iterate over systems
for it, Ti in enumerate(T):
    
    # Plot absorption spectra
    fig = plt.figure(figsize=figsize)
    
    # Initialize axes
    axs1 = fig.add_axes([left, bottom + 2*np.sum(row), column[0], row[0]])
    axs2 = fig.add_axes([left, bottom + 1*np.sum(row), column[0], row[0]])
    axs3 = fig.add_axes([left, bottom + 0*np.sum(row), column[0], row[0]])
    
    fig.suptitle(
        r'Angular velocity $\omega$ sequence of N$_2$O in SF$_6$ at ' 
        + '{:.1f} K\n'.format(float(Ti)), 
        fontweight='bold')
    
    # Lowest
    iv = 0
    Vmi = Vm[it][iv]
    ismpl = Smpls[Ti][Vmi][0]
            
    # Working directory
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(Ti)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
    
    # Angular velocity file
    wfile = os.path.join(
        workdir, 'velos_{:s}.npy'.format(tag))
    
    # Time file
    tfile = os.path.join(
        workdir, 'times_{:s}.npy'.format(tag))
    
    # Load results
    tlst = np.load(tfile)
    select = tlst < 50.0*1e3
    wlst = np.load(wfile)[select]
    tlst = tlst[select]
    
    # Pressure ratio
    p = M/Vmi/rhocrit
    
    # Plot
    label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
    
    axs1.plot(
        tlst*1e-3, wlst[:, 0, 0], '-', 
        color=color_scheme[iv], label=label)
    
    axs1.set_xlim([0, 50.0])
    axs1.set_xticklabels([])
    
    axs1.legend(
        loc='upper right', 
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
    
    
    
    
    # Middle
    iv = len(Vm[it])//2
    Vmi = Vm[it][iv]
    ismpl = Smpls[Ti][Vmi][0]
            
    # Working directory
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(Ti)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
    
    # Angular velocity file
    wfile = os.path.join(
        workdir, 'velos_{:s}.npy'.format(tag))
    
    # Time file
    tfile = os.path.join(
        workdir, 'times_{:s}.npy'.format(tag))
    
    # Load results
    tlst = np.load(tfile)
    select = tlst < 50.0*1e3
    wlst = np.load(wfile)[select]
    tlst = tlst[select]
    
    # Pressure ratio
    p = M/Vmi/rhocrit
    
    # Plot
    label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
    
    axs2.plot(
        tlst*1e-3, wlst[:, 0, 0], '-', 
        color=color_scheme[iv], label=label)
    
    axs2.set_xlim([0, 50.0])
    #axs2.set_ylim([0, 180.])
    
    axs2.set_xticklabels([])
    
    axs2.set_ylabel(r'$\omega$ (a.u.)', fontweight='bold')
    axs2.get_yaxis().set_label_coords(-0.10, 0.50)
    
    axs2.legend(
        loc='upper right', 
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
        
    
    
    # Highest
    iv = len(Vm[it]) - 1
    Vmi = Vm[it][iv]
    ismpl = Smpls[Ti][Vmi][0]
            
    # Working directory
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(Ti)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
    
     # Angular velocity file
    wfile = os.path.join(
        workdir, 'velos_{:s}.npy'.format(tag))
    
    # Time file
    tfile = os.path.join(
        workdir, 'times_{:s}.npy'.format(tag))
    
    # Load results
    tlst = np.load(tfile)
    select = tlst < 50.0*1e3
    wlst = np.load(wfile)[select]
    tlst = tlst[select]
    
    # Pressure ratio
    p = M/Vmi/rhocrit
    
    # Plot
    label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
    
    axs3.plot(
        tlst*1e-3, wlst[:, 0, 0], '-', 
        color=color_scheme[iv], label=label)
    
    axs3.set_xlim([0, 50.0])
    #axs3.set_ylim([0, 180.])
    
    axs3.set_xlabel(r'Time (ps)', fontweight='bold')
    axs3.get_xaxis().set_label_coords(0.5, -0.3)
    
    axs3.legend(
        loc='upper right', 
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
    
    plt.savefig(
        os.path.join(
            maindir, rsltdir, 'Seq_angvel_{:d}K.png'.format(int(Ti))),
        format='png', dpi=dpi)
    plt.close()


#-----------------------------
# Plot IR
#-----------------------------

avgfreq = 1.0
def moving_average(data_set, periods=9):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, 'same')


# Figure arrangement
figsize = (12, 8)
left = 0.10
bottom = 0.15
row = np.array([0.70, 0.00])
column = np.array([0.35, 0.15])

# Iterate over systems
for it, Ti in enumerate(T):
    
    if not request_IRspectra:
        continue
    
    # Plot absorption spectra
    fig = plt.figure(figsize=figsize)
    
    # Initialize axes
    axs1 = fig.add_axes([left, bottom, column[0], row[0]])
    axs2 = fig.add_axes([left + np.sum(column), bottom, column[0], row[0]])
    
    for iv, Vmi in enumerate(Vm[it]):
        
        # Choose number of frame condition
        #Nframes, counts = np.unique(Nframes_list[Ti][Vmi], return_counts=True)
        #iframes = np.argmax(counts)
        Nframes_choice = np.max(Nframes_list[Ti][Vmi])
        Nsmpls = 0
        
        # Prepare frequency and average spectra arrays
        Nfreq = int(Nframes_choice/2) + 1
        avgspec = np.zeros(Nfreq, dtype=float)
        ravg = [0.0, 0.0]
        rstd = [0.0, 0.0]
        #avgdabs = [0.0, 0.0, 0.0]
        #stddabs = [0.0, 0.0, 0.0]
        
        # Only sample 0
        for ismpl in Smpls[Ti][Vmi]:
            
            # Working directory
            workdir = os.path.join(
                maindir, 
                'T{:d}'.format(int(Ti)), 
                'V{:d}_{:d}'.format(int(Vmi), ismpl))
            
            # System tag
            tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
            
            # IR spectra file
            ifile = os.path.join(
                workdir, 'irspc_{:s}.npy'.format(tag))
            
            # Dipole file
            dfile = os.path.join(
                workdir, 'dipos_{:s}.npy'.format(tag))
            
            # Bond distance file
            rfile = os.path.join(
                workdir, 'bonds_{:s}.npy'.format(tag))
            
            # Time file
            tfile = os.path.join(
                workdir, 'times_{:s}.npy'.format(tag))
            
            # Skip if input file does not exist
            if not os.path.exists(dfile):
                continue
            
            # Number of frames and frequency points
            tlst = np.load(tfile)
            Nframes_smpl = len(tlst)
            Nfreq_smpl = int(Nframes_smpl/2) + 1
            
            # Check number of frames
            if Nframes_choice!=Nframes_smpl:
                continue
            
            # Frequency array
            dtime = tlst[1] - tlst[0]
            freq = np.arange(Nfreq)/float(Nframes_smpl)/dtime*jiffy
            if not os.path.exists(ifile) or recalc_IRspectra:
                
                # Load results
                dlst = np.load(dfile)
                
                # Weighting constant
                beta = 1.0/3.1668114e-6/float(Ti)
                hbar = 1.0
                cminvtoau = 1.0/2.1947e5
                const = beta*cminvtoau*hbar
                
                # Compute IR spectra
                
                acvx = acovf(dlst[:, 0, 0], fft=True)
                acvy = acovf(dlst[:, 0, 1], fft=True)
                acvz = acovf(dlst[:, 0, 2], fft=True)
                acv = acvx + acvy + acvz
                
                acv = acv*np.blackman(Nframes_smpl)
                spec = np.abs(np.fft.rfftn(acv))*np.tanh(const*freq/2.)
                
                # Save spectra
                np.save(ifile, spec)
                
            else:
                
                # Load results
                dlst = np.load(dfile)
                
                # Load results
                spec = np.load(ifile)
                tlst = np.load(tfile)
                
            avgspec += spec
            Nsmpls += 1
            
            # Load bond distance and angle information
            rlst = np.load(rfile)
            rlst[:, 0, 2][rlst[:, 0, 2] < -1.0] = -1.0
            ravg[0] += np.mean(rlst[:, 0, 0])
            ravg[1] += np.mean(np.arccos(rlst[:, 0, 2])*180.0/np.pi)
            rstd[0] += np.std(rlst[:, 0, 0])
            rstd[1] += np.std(np.arccos(rlst[:, 0, 2])*180.0/np.pi)
            
            #for ii in range(3):
                #avgdabs[ii] += np.mean(dlst[:, 0, ii])
                #stddabs[ii] += np.std(dlst[:, 0, ii])
            
        # Divide average spectra by number of samples
        avgspec /= Nsmpls
        
        # Apply moving average
        Nave = int(avgfreq/(freq[1] - freq[0]))
        avgspec = moving_average(avgspec, Nave)
        
        # Scale avgspec
        select = np.logical_and(
            freq > 300, freq < 2300)
        avgspec /= np.max(avgspec[select])
        
        # Pressure ratio
        p = M/Vmi/rhocrit
        
        # Plot
        label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
        rngr1 = [300, 2100]
        select1 = np.logical_and(
            freq > rngr1[0], freq < rngr1[1])
        sspec = avgspec/np.max(avgspec[select1])
        subselect = np.logical_and(
            freq > 800, freq < rngr1[1])
        subscale1 = 1./np.max(avgspec[subselect])
        sspec[subselect] = sspec[subselect]*subscale1
        
        axs1.plot(
            freq[select1], sspec[select1] + iv, '-', 
            color=color_scheme[iv], label=label)
        axs1.plot([800, 800], [iv, iv + 1], ':k')
        
        axs1.text(
            rngr1[0] + (rngr1[1] - rngr1[0])*0.01, 
            iv + 0.1, 'x{:.1f}'.format(1.0),
            fontdict={
                'family' : 'monospace',
                'style'  : 'italic',
                'weight' : 'light',
                'size'   : 'x-small'})
        axs1.text(
            800 + (rngr1[1] - rngr1[0])*0.01, 
            iv + 0.1, 'x{:.1f}'.format(subscale1),
            fontdict={
                'family' : 'monospace',
                'style'  : 'italic',
                'weight' : 'light',
                'size'   : 'x-small'})
        
        rnge2 = [2100, 2500]
        select2 = np.logical_and(
            freq > rnge2[0], freq < rnge2[1])
        subscale2 = 1./np.max(avgspec[select2])
        avgspec *= subscale2
        
        axs2.plot(
            freq[select2], avgspec[select2] + iv, '-', 
            color=color_scheme[iv], label=label)
        
        axs2.plot([exp_ir[0], exp_ir[0]], [iv, iv + 1], ':k')
        axs2.plot([exp_ir[1], exp_ir[1]], [iv, iv + 1], '--k')
        axs2.plot([exp_ir[2], exp_ir[2]], [iv, iv + 1], ':k')
        
        axs2.text(
            rnge2[0] + (rnge2[1] - rnge2[0])*0.01, 
            iv + 0.1, 'x{:.1f}'.format(subscale2),
            fontdict={
                'family' : 'monospace',
                'style'  : 'italic',
                'weight' : 'light',
                'size'   : 'x-small'})
            
        dtext = (
            r"$\left< d\mathrm{(N-N)} \right>$" 
            + r" = {:.3f} $\pm$ {:.3f} ".format(ravg[0]*a02A, rstd[0]*a02A) 
            + r"$\mathrm{\AA}$"
            + "\n"
            r"$\left< \theta\mathrm{(ONN)} \right>$"
            + r" = {:.1f} $\pm$ {:.1f}".format(ravg[1], rstd[1])
            + r"$^\circ$")
        #dtext = (
            #r"std$\left( \mu_x \right) = {:.1E}$".format(stddabs[0])
            #+ "\n"
            #r"std$\left( \mu_y \right) = {:.1E}$".format(stddabs[1])
            #+ "\n"
            #r"std$\left( \mu_z \right) = {:.1E}$".format(stddabs[2]))
        axs2.text(
            rnge2[0] + (rnge2[1] - rnge2[0])*0.5, 
            iv + 0.1, dtext,
            fontdict={
                'family' : 'monospace',
                'style'  : 'italic',
                'weight' : 'light',
                'size'   : 'x-small'})
        
    axs1.set_xlim(rngr1)
    axs2.set_xlim(rnge2)
    axs1.set_ylim([0, len(Vm[it])])
    axs2.set_ylim([0, len(Vm[it])])
    axs1.set_yticklabels([])
    axs2.set_yticklabels([])
    
    fave = (freq[1] - freq[0])*Nave
    fig.suptitle(
        r'IR Spectra of N$_2$O in {:s} at '.format(sys_tag[sys_solv]) 
        + '{:.1f} K\n'.format(float(Ti))
        + '(moving average of {:.2} '.format(fave) + r'cm$^{-1}$)', 
        fontweight='bold')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = list(range(len(Vm[it])))[::-1]
    axs1.legend(
        [handles[idx] for idx in order], [labels[idx] for idx in order],
        loc=(0.75, 0.3),  framealpha=1.0,
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
    
    axs1.set_xlabel(r'Frequency (cm$^{-1}$)', fontweight='bold')
    axs1.get_xaxis().set_label_coords(1.1, -0.1)
    axs1.set_ylabel('Intensity', fontweight='bold')
    axs1.get_yaxis().set_label_coords(-0.15, 0.50)
    
    #plt.show()
    plt.savefig(
        os.path.join(
            maindir, rsltdir, 'IR_spec_{:d}K.png'.format(int(Ti))),
        format='png', dpi=dpi)
    plt.close()
    
    
        
        

