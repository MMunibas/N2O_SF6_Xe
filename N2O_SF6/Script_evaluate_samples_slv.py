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
from MDAnalysis.analysis.distances import \
    distance_array, self_distance_array

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

# Temperatures [K]
T = [321.93]

# Critical temperature and density of SF6
M = 146.05 # g/mol
Tcrit = 273.15 + 45.6 # K
rhocrit = 0.74 # g/cm**3

# Relative density
#rhostar = [[0.16, 0.30, 0.67, 0.86, 0.99, 1.17, 1.36, 1.51, 1.87]]
rhostar = [[0.04, 0.10, 0.16, 0.30, 0.67, 0.86, 0.99, 1.17, 1.36, 1.51, 1.87]]

# Experimental rotational lifetime
rho_exp =      [0.16, 0.30, 0.67, 0.86, 0.99, 1.17, 1.36, 1.51, 1.87]
con_exp =      [0.82, 1.51, 3.43, 4.41, 5.06, 5.79, 6.94, 7.70, 9.58]
tau_exp =      [9.5,  6.0,  2.8,  2.4,  2.3,  1.9,  1.4,  0.9, np.nan]
err_exp =      [0.5,  0.4,  0.3,  0.2,  0.1,  0.1,  0.1,  0.1, np.nan]
tau_coll_exp = [6.7,  3.6,  1.7,  1.5,  1.4,  1.0,  0.8,  0.7, np.nan]
Texp = 321.9

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
sys_solv = 'SF6'
sys_resi = ['N2O', 'SF6']
sys_Natm = {
    'N2O': 3,
    'SF6': 7,
    'XE': 1}
sys_tag = {
    'N2O': r'N$_2$O',
    'SF6': r'SF$_6$',
    'XE': 'Xe'}
sys_bnds = [[0, 1], [1, 2], [0, 1, 2]]

# System residue atoms
sys_atm = {
    'N2O': ['N', 'N', 'O'],
    'SF6': ['S', 'F', 'F', 'F', 'F', 'F', 'F'],
    'XE': ['Xe']}

# System residue center atom or center of mass
sys_cnt = {
    'N2O': [0, 1, 2],
    'SF6': 0,
    'XE': 0}

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
traj_psffile = 'init.n2o_sf6.psf'

# System coord file
sys_fcrd = 'init.n2o_sf6.crd'

# Fixed Time step if needed
fixdt = None
#fixdt = {
    #tuple(0): 0.002,
    #tuple(range(1,10)): 0.050}



# Evaluation time step skip
eval_skip = 50   # 50 fs

# Distance binning density in 1/Angstrom
dst_step = 0.1




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
# Read Trajectories 
#-----------------------------

# Temperature and molar volume combinations
T_Vm_Smpls = [
    [Ti, Vmi, ismpl]
    for it, Ti in enumerate(T) 
    for iv, Vmi in enumerate(Vm[it])
    for ismpl in Smpls[Ti][Vmi]]

def read_local(isys):
    
    # Temperature and molar volume
    Ti = T_Vm_Smpls[isys][0]
    Vmi = T_Vm_Smpls[isys][1]
    ismpl = T_Vm_Smpls[isys][2]

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
    psffile = os.path.join(workdir, traj_psffile)
    
    # Sort dcd files
    dcdsort = np.argsort(iruns)
    dcdfiles = dcdfiles[dcdsort]
    iruns = iruns[dcdsort]
    
    # Number of target molecules
    Ntrgt = sys_Nres[tag][sys_trgt]
    
    # Number of solvent molecules
    Nsolv = sys_Nres[tag][sys_solv]
    
    # Target atom indices
    indcs_trgt = sys_ires[tag][sys_trgt]
        
    # Target atom indices
    indcs_solv = sys_ires[tag][sys_solv]
    
    # Initialize trajectory time counter in ps
    traj_time_dcd = 0.0
    
    # Iterate over dcd files
    for ir, idcd in enumerate(iruns):
        
        # Solute - Solvent distance histogram file
        nfile_i = os.path.join(
            workdir, 'nhslv_{:s}_{:d}.npy'.format(tag, idcd))
        
        # Time file
        tfile_i = os.path.join(
            workdir, 'ntslv_{:s}_{:d}.npy'.format(tag, idcd))
        
        if not os.path.exists(nfile_i) or True:
            
            # Distance histogram list
            nlst = []
            
            # Times list
            tlst = []
            
            # Open dcd file
            dcdfile = dcdfiles[ir]
            dcd = MDAnalysis.Universe(psffile, dcdfile)
            
            print(idcd, dcdfile)
            
            # Get trajectory parameter
            Nframes = len(dcd.trajectory)
            Nskip = int(dcd.trajectory.skip_timestep)
            dt = np.round(
                float(dcd.trajectory._ts_kwargs['dt']), decimals=8)/Nskip
            if fixdt is not None:
                dt = fixdt
            
            # Get masses
            masses = dcd._topology.masses.values
            
            # Get atom types
            atoms = np.array([ai for ai in dcd._topology.names.values])
            
            # System center auxiliaries
            # Ntrgt, Ncnt
            cidcs_trgt = indcs_trgt[:, np.array(sys_cnt[sys_trgt])].reshape(
                Ntrgt, -1)
            # Ntrgt, Ncnt
            aux_trgt = (
                masses[cidcs_trgt]
                /np.sum(
                    masses[cidcs_trgt], 
                    axis=1).reshape(-1, 1)
                )
            # Nsolv, Ncnt, 1
            cidcs_solv = indcs_solv[:, np.array(sys_cnt[sys_solv])].reshape(
                Nsolv, -1)
            # Nsolv, Ncnt, 1
            aux_solv = (
                masses[cidcs_solv]
                /np.sum(
                    masses[cidcs_solv], 
                    axis=1).reshape(-1, 1)
                )
            
            # Iterate over frames
            for iframe, frame in enumerate(dcd.trajectory):
                
                # Update time
                traj_time = traj_time_dcd + Nskip*dt*iframe*1e3
                
                if not iframe%eval_skip:
                
                    # Atom positions (atom, cart)
                    pos = frame._pos
                    
                    # Get cell information
                    cell = frame._unitcell
                    if iframe == 0:
                        dst_lim = np.min(cell[:3])/2.
                    
                    # Target atom positions
                    # Ntrgt, 3
                    pos_trgt = np.sum(
                        pos[cidcs_trgt]*aux_trgt.reshape(Ntrgt, -1, 1), 
                        axis=1)
                    
                    # Solvent atom positions
                    pos_solv = np.sum(
                        pos[cidcs_solv]*aux_solv.reshape(Nsolv, -1, 1), 
                        axis=1)
                    
                    # Compute distances
                    dsts_atm = (
                        self_distance_array(pos_solv, box=cell)
                        ).reshape(-1)
                    
                    # Get distance histogram
                    nhst, _ = np.histogram(
                        dsts_atm, bins=np.arange(0.0, dst_lim, dst_step))
                    
                    # Append frame result
                    nlst.append(nhst.astype(float))
                    tlst.append(traj_time)
                
            # Update trajectory time
            traj_time_dcd = traj_time + Nskip*dt*1e3
            
            # Save results
            np.save(nfile_i, np.array(nlst, dtype=float))
            np.save(tfile_i, np.array(tlst, dtype=np.float32))
            
        else:
            
            # Open dcd file
            dcdfile = dcdfiles[ir]
            dcd = MDAnalysis.Universe(psffile, dcdfile)
            
            # Get trajectory parameter
            Nskip = int(dcd.trajectory.skip_timestep)
            dt = np.round(
                float(dcd.trajectory._ts_kwargs['dt']), decimals=8)/Nskip
            if fixdt is not None:
                dt = fixdt
            
            # Update trajectory time
            tlst = np.load(tfile_i)
            traj_time_dcd = tlst[-1] + Nskip*dt*1e3
       

if request_reading and tasks==1:
    for i in range(0, len(T_Vm_Smpls)):
        read_local(i)
elif request_reading:    
    if __name__ == '__main__':
        pool = Pool(tasks)
        pool.imap(read_local, range(0, len(T_Vm_Smpls)))
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
            
            # Sort dcd files
            dcdsort = np.argsort(iruns)
            dcdfiles = dcdfiles[dcdsort]
            iruns = iruns[dcdsort]
            
            # Solute - Solvent distance histogram file
            nfile = os.path.join(
                workdir, 'nhslv_{:s}.npy'.format(tag))
            
            # Time file
            tfile = os.path.join(
                workdir, 'ntslv_{:s}.npy'.format(tag))
            
            if not os.path.exists(nfile) or request_reappending:
                
                nfile_0 = os.path.join(
                    workdir, 'nhslv_{:s}_{:d}.npy'.format(tag, 0))
                if not os.path.exists(nfile_0):
                    print("Skip results for T{:d}/V{:d}_{:d}".format(
                        int(Ti), int(Vmi), ismpl))
                    continue
                else:
                    print("Append results for T{:d}/V{:d}_{:d}".format(
                        int(Ti), int(Vmi), ismpl))
                    
                # Solute - Solvent distance histogram list
                tmplst = []
                for ir, idcd in enumerate(iruns):
                    nfile_i = os.path.join(
                        workdir, 'nhslv_{:s}_{:d}.npy'.format(tag, idcd))
                    if os.path.exists(nfile_i):
                        tmplst.append(np.load(nfile_i))
                    #os.remove(nfile_i)
                if len(tmplst):
                    np.save(nfile, np.concatenate(tmplst))
                
                # Time list
                tmplst = []
                for ir, idcd in enumerate(iruns):
                    tfile_i = os.path.join(
                        workdir, 'ntslv_{:s}_{:d}.npy'.format(tag, idcd))
                    if os.path.exists(tfile_i):
                        tmplst.append(np.load(tfile_i))
                    #os.remove(tfile_i)
                if len(tmplst):
                    tmplst = np.concatenate(tmplst)
                    tmax = tmplst[-1]
                    np.save(tfile, tmplst)
                
            #else:
                
                ## Open time list file
                #tmplst = np.load(tfile)
                
                ## Add frame number to list
                #Nframes_list[Ti][Vmi].append(len(tmplst))
                


#-----------------------------
# Radial Solvent Distribution
#-----------------------------

# Figure arrangement
figsize = (12, 8)
left = 0.12
bottom = 0.15
row = np.array([0.70, 0.00])
column = np.array([0.50, 0.10])

# Iterate over systems
for it, Ti in enumerate(T):
    
    # Initialize axes
    fig = plt.figure(figsize=figsize)
    axs1 = fig.add_axes([left, bottom, column[0], row[0]])
    
    # Max values
    gmin = 0.0
    gmax = 0.0
    
    # Max plotting distance
    dst_mplt = 15.0
    
    for iv, Vmi in enumerate(Vm[it]):
        
        # Count number of valid samples
        Nsmpls = 0
        
        # Radial distribution functions of each sample
        gsmpls = []
        
        for ismpl in Smpls[Ti][Vmi]:
            
            # Working directory
            workdir = os.path.join(
                maindir, 
                'T{:d}'.format(int(Ti)), 
                'V{:d}_{:d}'.format(int(Vmi), ismpl))
            
            # System tag
            tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
            
            # Radial distribution function
            gfile = os.path.join(
                workdir, 'g_slv_{:s}.npy'.format(tag))
            
            # Solute - Solvent distance histogram file
            nfile = os.path.join(
                workdir, 'nhslv_{:s}.npy'.format(tag))
            
            # Time file
            tfile = os.path.join(
                workdir, 'ntslv_{:s}.npy'.format(tag))
            
            # Skip if input file does not exist
            if not os.path.exists(nfile):
                continue
            
            if not os.path.exists(gfile) or True:
                
                # Load results
                nlst = np.load(nfile)
                tlst = np.load(tfile)
                
                # Sum time steps
                nlst = np.sum(nlst, axis=0)
                #print(Vmi, nlst)
                # Radial grid
                dst_rnge = [0.0, len(nlst)*dst_step]
                dst_bins = np.arange(0.0, dst_rnge[-1] + dst_step/2., dst_step)
                dst_cntr = dst_bins[:-1] + dst_step/2.
                
                # Normalize
                N = np.sum(nlst)
                V = 4./3.*np.pi*dst_rnge[1]**3
                g = V*nlst/dst_step/N
                g = g/(4.0*np.pi*dst_cntr**2)
                
                # Save radial distribution function
                np.save(gfile, g)
                
            else:
                
                # Load radial distribution function
                g = np.load(gfile)
                
                # Radial grid
                dst_rnge = [0.0, len(g)*dst_step]
                dst_cntr = np.array(dst_rnge[:-1]) + dst_step/2.
                
            # Append result to list
            gsmpls.append(g)
            Nsmpls += 1
        
        # Get sample average
        dst_Nstp = np.min([len(gsmpl) for gsmpl in gsmpls if len(gsmpl)])
        g = np.zeros(dst_Nstp, dtype=float)
        for gsmpl in gsmpls:
            g += gsmpl[:dst_Nstp]/Nsmpls
        
        # Radial grid
        dst_rnge = [0.0, len(g)*dst_step]
        dst_bins = np.arange(0.0, dst_rnge[-1] + dst_step/2., dst_step)
        dst_cntr = dst_bins[:-1] + dst_step/2.
        
        # Local minima and maxima
        ilocmin = scipy.signal.argrelmin(g, order=2)
        ilocmax = scipy.signal.argrelmax(g, order=2)
        rlocmin = np.sort(dst_cntr[ilocmin])
        rlocmax = np.sort(dst_cntr[ilocmax])
        if len(rlocmin) >= 3: 
            llocmin = '{:.2f}, {:.2f}, {:.2f}'.format(*rlocmin[:3])
        else:
            llocmin = ('{:.2f}, '*len(rlocmin)).format(*rlocmin)[:-2]
        if len(rlocmax) >= 2: 
            llocmax = '{:.2f}, {:.2f}'.format(*rlocmax[:2])
        else:
            llocmax = ('{:.2f}, '*len(rlocmax)).format(*rlocmax)[:-2]
        
        # Pressure ratio
        p = M/Vmi/rhocrit
        label = (
            r'{:.2f} [{:.2f}]'.format(CM[it][iv], p)
            + '\n'
            + '[{:s}]/[{:s}]'.format(llocmin, llocmax))
        
        # Plot g
        axs1.plot(
            dst_cntr, g, color=color_scheme[iv], label=label)
        
        if np.max(g) > gmax:
            gmax = np.max(g)
        if np.min(g) > gmin:
            gmin = np.min(g)
        
    title = r'Radial distribution function $g(r)$ between ' + '\n'
    if len(np.array(sys_cnt[sys_solv]).shape):
        title += r'{:s} (center of mass) '.format(sys_tag[sys_solv])
    else:
        title += r'{:s} ({:s})'.format(
            sys_tag[sys_solv], sys_atm[sys_solv][sys_cnt[sys_solv]])
    title += 'solvent molecules at {:.1f} K'.format(float(Ti))
    fig.suptitle(title, fontweight='bold')
    
    axs1.set_xlabel(r'Distance $r$ ($\mathrm{\AA}$)', fontweight='bold')
    axs1.get_xaxis().set_label_coords(0.5, -0.1)
    axs1.set_ylabel(
        r'$g(r)$', fontweight='bold')
    axs1.get_yaxis().set_label_coords(-0.08, 0.50)
    
    axs1.set_xlim([0.0, dst_mplt])
    axs1.set_ylim([gmin, gmax*1.1])
    
    axs1.legend(
        loc=[1.05, -0.20], 
        title=(
            r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv])
            + '\n(Local minima/maxima)'))
    
    plt.savefig(
        os.path.join(
            maindir, rsltdir, 'g_solvent_solvent_{:s}.png'.format(sys_solv)),
        format='png', dpi=dpi)
    plt.close()
    
