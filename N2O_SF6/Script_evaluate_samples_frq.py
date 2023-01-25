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
tasks = 1

# Requested jobs
request_reading = True

request_histogram = True
request_summary = True
request_correlation = True

recalc_correlation = True

# Temperatures [K]
T = [321.93]

# Critical temperature and density of SF6
M = 146.05 # g/mol
Tcrit = 273.15 + 45.6 # K
rhocrit = 0.74 # g/cm**3

# Relative density
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

# Main directory
maindir = os.getcwd()

# Result directory
rsltdir = 'results'

# Trajectory source
sys_fdcd = 'dyna_crd.*.dcd'
sys_scrd = ['.', 1]
sys_fvel = 'dyna_vel.*.dcd'
sys_svel = ['.', 1]

# Frequency source
sys_finm = 'vibmode_frequencies.*.dat'
sys_sinm = ['.', 1]
sys_fqnm = 'inm_results/inm_*.dat'
sys_sqnm = [['_', 1], ['.', 0]]

# System psf file
traj_psffile = 'init.n2o_sf6.psf'

# System coord file
sys_fcrd = 'init.n2o_sf6.crd'

# Fixed Time step if needed
fixdt = None
#fixdt = 0.001

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
#color_scheme = [
    #'b', 'r', 'orange', 'purple', 'green', 'magenta', 'cyan', 'olive', 'black']


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
    psffile = os.path.join(workdir, traj_psffile)
    
    # Get instantaneous and quenched normal mode files
    inmfiles = np.array(glob(os.path.join(workdir, sys_finm)))
    iinms = np.zeros(inmfiles.shape[0], dtype=int)
    if isinstance(sys_sinm[0], str):
        sinm = [sys_sinm]
    else:
        sinm = sys_sinm
    for ii, inmfile in enumerate(inmfiles):
        inmfile = inmfile.split('/')[-1]
        for s in sinm:
            inmfile = inmfile.split(s[0])[s[1]]
        iinms[ii] = int(inmfile)
    qnmfiles = np.array(glob(os.path.join(workdir, sys_fqnm)))
    iqnms = np.zeros(qnmfiles.shape[0], dtype=int)
    if isinstance(sys_sqnm[0], str):
        sqnm = [sys_sqnm]
    else:
        sqnm = sys_sqnm
    for ii, qnmfile in enumerate(qnmfiles):
        qnmfile = qnmfile.split('/')[-1]
        for s in sqnm:
            qnmfile = qnmfile.split(s[0])[s[1]]
        iqnms[ii] = int(qnmfile)
    
    # Sort files
    dcdsort = np.argsort(iruns)
    dcdfiles = dcdfiles[dcdsort]
    iruns = iruns[dcdsort]
    inmsort = np.argsort(iinms)
    inmfiles = inmfiles[inmsort]
    iinms = iinms[inmsort]
    qnmsort = np.argsort(iqnms)
    qnmfiles = qnmfiles[qnmsort]
    iqnms = iqnms[qnmsort]
    
    # Open first dcd file
    dcdfile = dcdfiles[0]
    dcd = MDAnalysis.Universe(psffile, dcdfile)
    
    # Get trajectory parameter
    Nframes = len(dcd.trajectory)
    Nskip = int(dcd.trajectory.skip_timestep)
    dt = np.round(
        float(dcd.trajectory._ts_kwargs['dt']), decimals=8)/Nskip
    if fixdt is not None:
        dt = fixdt
    
    # Instantaneous normal mode frequencies file
    ifile = os.path.join(
        workdir, 'inmfr_{:s}.npy'.format(tag))
    
    # Quenched normal mode frequencies file
    qfile = os.path.join(
        workdir, 'qnmfr_{:s}.npy'.format(tag))
    
    # File auxiliary parameter
    nline_ninm = 5
        
    if not os.path.exists(ifile) or True:
        
        # Initialize array
        inmarr = np.zeros(
            [len(iruns)*Nframes, 3*sys_Natm[sys_trgt]], 
            dtype=np.float32)
        qnmarr = np.zeros(
            [len(iruns)*Nframes, 3*sys_Natm[sys_trgt]], 
            dtype=np.float32)
        
        # Iterate over dcd files
        for ir, idcd in enumerate(iruns):
            
            # Run index
            irun = ir*Nframes
            
            if idcd in iinms:
                
                iinm = np.where(iinms==idcd)[0][0]
                
                # Open instantaneous normal mode frequencies file
                with open(inmfiles[iinm], 'r') as f:
                    frqs = f.readlines()
                
                # Iterate over frames
                Nlines = len(frqs)
                for ii in range(Nframes):
                    if (nline_ninm*(ii + 1)) >= Nlines:
                        inmarr[(irun + ii):, :] = np.nan
                        break
                    frqlines = " ".join(
                        frqs[nline_ninm*ii + 2:nline_ninm*(ii + 1) - 1])
                    inmarr[irun + ii, :] = frqlines.split()[1::2]
                    
                    
            else:
                
                inmarr[irun:, :] = np.nan
                
            if idcd in iqnms:
                
                iqnm = np.where(iqnms==idcd)[0][0]
            
                # Open quenched normal mode frequencies file
                with open(qnmfiles[iqnm], 'r') as f:
                    frqs = f.readlines()
                    
                # Iterate over frames
                Nlines = len(frqs)
                for ii in range(Nframes):
                    if ii >= Nlines:
                        qnmarr[ii:, :] = np.nan
                        break
                    jj = 0
                    sfrqs = frqs[ii].split()
                    for fi in range(3*sys_Natm[sys_trgt]):
                        if sfrqs[jj] == "-":
                            qnmarr[irun + ii, fi] = -float(sfrqs[jj + 1])
                            jj += 2
                        else:
                            qnmarr[irun + ii, fi] = float(sfrqs[jj])
                            jj += 1
                
            else:
                
                qnmarr[irun:, :] = np.nan
            
        # Save results
        np.save(ifile, inmarr)
        np.save(qfile, qnmarr)
        
if request_reading and tasks==1:
    for i in range(0, len(T_Vm_Smpls)):
        read(i)
elif request_reading:    
    if __name__ == '__main__':
        pool = Pool(tasks)
        pool.imap(read, range(0, len(T_Vm_Smpls)))
        pool.close()
        pool.join()


#-----------------------------
# Plot Histogram
#-----------------------------

# Figure arrangement
figsize = (12, 8)
left = 0.10
bottom = 0.15
row = np.array([0.70, 0.00])
column = np.array([0.15, 0.04])

# Frequency binning
dfrq = 1.0
rfrq = [-500.0, 3000.0]

frq_selection = [
     [-200.,  200.],
     [ 500.,  700.],
     [1100., 1500.],
     [2100., 2350.]]

bins_frqs = np.arange(rfrq[0], rfrq[1] + dfrq/2., dfrq)
cntr_frqs = bins_frqs[1:] - dfrq/2.
            
# Iterate over systems
for it, Ti in enumerate(T):
    
    if not request_histogram:
        continue
    
    # Plot absorption spectra
    fig = plt.figure(figsize=figsize)
    
    # Initialize axes
    axs1 = fig.add_axes([left + 0.0*np.sum(column), bottom, column[0], row[0]])
    axs2 = fig.add_axes([left + 1.0*np.sum(column), bottom, column[0], row[0]])
    axs3 = fig.add_axes([left + 2.0*np.sum(column), bottom, column[0], row[0]])
    axs4 = fig.add_axes([left + 3.0*np.sum(column), bottom, column[0], row[0]])
    axs = [axs1, axs2, axs3, axs4]
    
    # Frequency histogram
    hist_frqs = np.zeros(cntr_frqs.shape[0], dtype=int)
    
    for iv, Vmi in enumerate(Vm[it]):
        
        # Only sample 0
        for ismpl in Smpls[Ti][Vmi]:
        
            # Working directory
            workdir = os.path.join(
                maindir, 
                'T{:d}'.format(int(Ti)), 
                'V{:d}_{:d}'.format(int(Vmi), ismpl))
            
            # System tag
            tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
        
            # Instantaneous normal mode frequencies file
            ifile = os.path.join(
                workdir, 'inmfr_{:s}.npy'.format(tag))
            
            # Load data
            inmlst = np.load(ifile)
            
            # Correct list - keep 1900 < vas(2200) < 2500
            mask = np.logical_and(
                inmlst[:, -1] > 2500.,
                inmlst[:, -1] < 1700.)
            inmlst[mask, :] = np.nan
            inmlst = inmlst[~np.isnan(inmlst)]
            
            # Bin frequencies
            hist_frqs = hist_frqs + np.histogram(
                inmlst, bins=bins_frqs)[0]
            
        # Scale histogram
        hist_frqs = hist_frqs.astype(float)/np.sum(hist_frqs)
        hist_frqs = hist_frqs/np.max(hist_frqs)
        
        # Pressure ratio
        p = M/Vmi/rhocrit
        
        # Plot
        label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
        
        for ia, srnge in enumerate(frq_selection):
            
            selection = np.logical_and(
                cntr_frqs > srnge[0],
                cntr_frqs < srnge[1])
            axs[ia].plot(
                cntr_frqs[selection], 
                hist_frqs[selection]/np.nanmax(hist_frqs[selection]) + iv, 
                '-', color=color_scheme[iv], label=label)
            
            axs[ia].set_xlim(srnge)
            axs[ia].set_ylim([0, len(Vm[it])])
    
    fig.suptitle(
        r'Instantaneous normal modes of N$_2$O in {:s} at '.format(
            sys_tag[sys_solv]) 
        + '{:.1f} K\n'.format(float(Ti)), 
        fontweight='bold')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = list(range(len(Vm[it])))[::-1]
    axs[3].legend(
        [handles[idx] for idx in order], [labels[idx] for idx in order],
        loc=(0.70, 0.3),  framealpha=1.0,
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
    
    axs[1].set_xlabel(r'INM Frequency (cm$^{-1}$)', fontweight='bold')
    axs[1].get_xaxis().set_label_coords(1.0, -0.1)
    axs[0].set_ylabel('Probability', fontweight='bold')
    axs[0].get_yaxis().set_label_coords(-0.15, 0.50)
    
    axs[0].set_yticklabels([])
    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])
    axs[3].set_yticklabels([])
    
    plt.savefig(
        os.path.join(
            maindir, rsltdir, 'INM_hist_{:d}K.png'.format(int(Ti))),
        format='png', dpi=dpi)
    plt.close()
    
exit()



frq_selection = [
     [-50.,  150.],
     [ 600.,  650.],
     [1250., 1350.],
     [2150., 2200.]]

        
# Iterate over systems
for it, Ti in enumerate(T):
    
    if not request_histogram:
        continue
    
    # Plot absorption spectra
    fig = plt.figure(figsize=figsize)
    
    # Initialize axes
    axs1 = fig.add_axes([left + 0.0*np.sum(column), bottom, column[0], row[0]])
    axs2 = fig.add_axes([left + 1.0*np.sum(column), bottom, column[0], row[0]])
    axs3 = fig.add_axes([left + 2.0*np.sum(column), bottom, column[0], row[0]])
    axs4 = fig.add_axes([left + 3.0*np.sum(column), bottom, column[0], row[0]])
    axs = [axs1, axs2, axs3, axs4]
    
    # Frequency histogram
    hist_frqs = np.zeros(cntr_frqs.shape[0], dtype=int)
    
    for iv, Vmi in enumerate(Vm[it]):
        
        # Only sample 0
        for ismpl in Smpls[Ti][Vmi]:
        
            # Working directory
            workdir = os.path.join(
                maindir, 
                'T{:d}'.format(int(Ti)), 
                'V{:d}_{:d}'.format(int(Vmi), ismpl))
            
            # System tag
            tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
        
            # Instantaneous normal mode frequencies file
            qfile = os.path.join(
                workdir, 'qnmfr_{:s}.npy'.format(tag))
            
            # Load data
            qnmlst = np.load(qfile)
            
            # Correct list - keep 1900 < vas(2200) < 2500
            mask = np.logical_and(
                qnmlst[:, -1] > 2500.,
                qnmlst[:, -1] < 1700.)
            qnmlst[mask, :] = np.nan
            qnmlst = qnmlst[~np.isnan(qnmlst)]
            
            # Bin frequencies
            hist_frqs = hist_frqs + np.histogram(
                qnmlst, bins=bins_frqs)[0]
            
        # Scale histogram
        hist_frqs = hist_frqs.astype(float)/np.sum(hist_frqs)
        hist_frqs = hist_frqs/np.max(hist_frqs)
        
        # Pressure ratio
        p = M/Vmi/rhocrit
        
        # Plot
        label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
        
        for ia, srnge in enumerate(frq_selection):
            
            selection = np.logical_and(
                cntr_frqs > srnge[0],
                cntr_frqs < srnge[1])
            axs[ia].plot(
                cntr_frqs[selection], 
                hist_frqs[selection]/np.nanmax(hist_frqs[selection]) + iv, 
                '-', color=color_scheme[iv], label=label)
            
            axs[ia].set_xlim(srnge)
            axs[ia].set_ylim([0, len(Vm[it])])
    
    fig.suptitle(
        r'Quenched normal modes of N$_2$O in {:s} at '.format(
            sys_tag[sys_solv]) 
        + '{:.1f} K\n'.format(float(Ti)), 
        fontweight='bold')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = list(range(len(Vm[it])))[::-1]
    axs[3].legend(
        [handles[idx] for idx in order], [labels[idx] for idx in order],
        loc=(0.70, 0.3),  framealpha=1.0,
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
    
    axs[1].set_xlabel(r'INM Frequency (cm$^{-1}$)', fontweight='bold')
    axs[1].get_xaxis().set_label_coords(1.0, -0.1)
    axs[0].set_ylabel('Probability', fontweight='bold')
    axs[0].get_yaxis().set_label_coords(-0.15*0.35/column[0], 0.50)
    
    axs[0].set_yticklabels([])
    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])
    axs[3].set_yticklabels([])
    
    plt.savefig(
        os.path.join(
            maindir, rsltdir, 'QNM_hist_{:d}K.png'.format(int(Ti))),
        format='png', dpi=dpi)
    plt.close()
    

#-----------------------------
# Plot Summary Histogram
#-----------------------------

# Figure arrangement
figsize = (12, 8)
left = 0.10
bottom = 0.15
row = np.array([0.70, 0.00])
column1 = np.array([0.40, 0.04])
column2 = np.array([0.40/3., 0.04/3.])

frq_selection = [
     [-50.,  200.],
     [ 610.,  630.],
     [1285., 1305.],
     [2170., 2190.]]

# Iterate over systems
for it, Ti in enumerate(T):
    
    if not request_summary:
        continue
    
    # Plot absorption spectra
    fig = plt.figure(figsize=figsize)
    
    # Initialize axes
    axs1 = fig.add_axes([left, bottom, column1[0]/2., row[0]])
    axs2 = fig.add_axes([
        left + np.sum(column1) + 0.0*np.sum(column2), bottom, 
        column2[0], row[0]])
    axs3 = fig.add_axes([
        left + np.sum(column1) + 1.0*np.sum(column2), bottom, 
        column2[0], row[0]])
    axs4 = fig.add_axes([
        left + np.sum(column1) + 2.0*np.sum(column2), bottom, 
        column2[0], row[0]])
    axs = [axs1, axs2, axs3, axs4]
    
    # Frequency histogram
    hist_inms = np.zeros(cntr_frqs.shape[0], dtype=int)
    hist_inms_trans = np.zeros(cntr_frqs.shape[0], dtype=int)
    hist_inms_rot = np.zeros(cntr_frqs.shape[0], dtype=int)
    hist_qnms = np.zeros(cntr_frqs.shape[0], dtype=int)
    
    for iv, Vmi in enumerate(Vm[it]):
        
        # Only sample 0
        for ismpl in Smpls[Ti][Vmi]:
        
            # Working directory
            workdir = os.path.join(
                maindir, 
                'T{:d}'.format(int(Ti)), 
                'V{:d}_{:d}'.format(int(Vmi), ismpl))
            
            # System tag
            tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
        
            # Instantaneous normal mode frequencies file
            ifile = os.path.join(
                workdir, 'inmfr_{:s}.npy'.format(tag))
            
            # Instantaneous normal mode frequencies file
            qfile = os.path.join(
                workdir, 'qnmfr_{:s}.npy'.format(tag))
    
            # Load data
            inmlst = np.load(ifile)
            qnmlst = np.load(qfile)
            
            # Correct list - keep 1900 < vas(2200) < 2500
            mask = np.logical_and(
                inmlst[:, -1] > 2500.,
                inmlst[:, -1] < 1900.)
            inmlst[mask, :] = np.nan
            inmlst = inmlst[~np.isnan(inmlst[:, -1]), :]
            mask = np.logical_and(
                qnmlst[:, -1] > 2500.,
                qnmlst[:, -1] < 1900.)
            qnmlst[mask, :] = np.nan
            qnmlst = qnmlst[~np.isnan(qnmlst[:, -1]), :]
            
            # Bin frequencies
            hist_inms = hist_inms + np.histogram(
                inmlst, bins=bins_frqs)[0]
            hist_inms_trans = hist_inms_trans + np.histogram(
                inmlst[:, :3], bins=bins_frqs)[0]
            hist_inms_rot = hist_inms_rot + np.histogram(
                inmlst[:, 3:5], bins=bins_frqs)[0]
            hist_qnms = hist_qnms + np.histogram(
                qnmlst, bins=bins_frqs)[0]
            
        # Scale histogram
        hist_inms_trans = hist_inms_trans.astype(float)/np.sum(hist_inms)
        hist_inms_rot = hist_inms_rot.astype(float)/np.sum(hist_inms)
        hist_inms = hist_inms.astype(float)/np.sum(hist_inms)
        hist_qnms = hist_qnms.astype(float)/np.sum(hist_qnms)
        
        # Pressure ratio
        p = M/Vmi/rhocrit
        
        # Plot
        
        label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
        
        selection = np.logical_and(
            cntr_frqs > frq_selection[0][0],
            cntr_frqs < frq_selection[0][1])
        inms_max = np.nanmax(hist_inms[selection])
        axs[0].plot(
            cntr_frqs[selection], 
            hist_inms_trans[selection]/inms_max + iv, 
            '--', color=color_scheme[iv])
        axs[0].plot(
            cntr_frqs[selection], 
            hist_inms_rot[selection]/inms_max + iv, 
            ':', color=color_scheme[iv])
        axs[0].plot(
            cntr_frqs[selection], 
            hist_inms[selection]/inms_max + iv, 
            '-', color=color_scheme[iv], label=label)
        
        axs[0].set_xlim(frq_selection[0])
        axs[0].set_ylim([0, len(Vm[it])])
        
        for ia, srnge in enumerate(frq_selection[1:]):
            
            selection = np.logical_and(
                cntr_frqs > srnge[0],
                cntr_frqs < srnge[1])
            axs[ia + 1].plot(
                cntr_frqs[selection], 
                hist_qnms[selection]/np.nanmax(hist_qnms[selection]) + iv, 
                '-', color=color_scheme[iv], label=label)
            
            axs[ia + 1].set_xlim(srnge)
            axs[ia + 1].set_ylim([0, len(Vm[it])])
    
    fig.suptitle(
        'Instantaneous (left) and quenched (right) normal modes\n'
        + r'of N$_2$O in {:s} at '.format(sys_tag[sys_solv]) 
        + '{:.1f} K\n'.format(float(Ti)), 
        fontweight='bold')
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = list(range(len(Vm[it])))[::-1]
    axs[0].legend(
        [handles[idx] for idx in order], [labels[idx] for idx in order],
        loc=(1.10, 0.3),  framealpha=1.0,
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
    
    axs[0].set_xlabel(r'INM Frequency (cm$^{-1}$)', fontweight='bold')
    axs[0].get_xaxis().set_label_coords(0.5, -0.1)
    axs[2].set_xlabel(r'QNM Frequency (cm$^{-1}$)', fontweight='bold')
    axs[2].get_xaxis().set_label_coords(0.5, -0.1)
    axs[0].set_ylabel('Probability', fontweight='bold')
    axs[0].get_yaxis().set_label_coords(-0.15*0.30/column[0], 0.50)
    
    axs[0].set_yticklabels([])
    axs[1].set_yticklabels([])
    axs[2].set_yticklabels([])
    axs[3].set_yticklabels([])
    
    plt.savefig(
        os.path.join(
            maindir, rsltdir, 'INM_QNM_hist_{:d}K.png'.format(int(Ti))),
        format='png', dpi=dpi)
    plt.close()
    
#-----------------------------
# Plot QNM Correlation
#-----------------------------

# Figure arrangement
figsize = (12, 8)
left = 0.12
bottom = 0.15
row = np.array([0.70, 0.00])
column = np.array([0.50, 0.10])

plt_log = False
pltt = 1.0

corr_nmd = 9

# Iterate over systems
for it, Ti in enumerate(T):
    
    if not request_correlation:
        continue
    
    # Plot Correlation
    fig = plt.figure(figsize=figsize)
    
    # Initialize axes
    axs1 = fig.add_axes([left, bottom, column[0], row[0]])
    
    # MinMax values
    cmin = 0.0
    cmax = 0.0
    
    for iv, Vmi in enumerate(Vm[it]):
        
        # Only sample 0
        for ismpl in Smpls[Ti][Vmi]:
        
            # Working directory
            workdir = os.path.join(
                maindir, 
                'T{:d}'.format(int(Ti)), 
                'V{:d}_{:d}'.format(int(Vmi), ismpl))
            
            # System tag
            tag = '{:d}_{:d}'.format(int(Ti), int(Vmi))
        
            # Instantaneous normal mode frequencies file
            ifile = os.path.join(
                workdir, 'inmfr_{:s}.npy'.format(tag))
            
            # Quenched normal mode frequencies file
            qfile = os.path.join(
                workdir, 'qnmfr_{:s}.npy'.format(tag))
            
            # Time file
            tfile = os.path.join(
                workdir, 'times_{:s}.npy'.format(tag))
            
            # Load time list
            tlst = np.load(tfile)
            
            if ismpl == 0:
                
                # Correlation arrays
                corr_inms = np.zeros(tlst.shape[0], dtype=float)
                corr_qnms = np.zeros(tlst.shape[0], dtype=float)
                
            # Quenched normal mode frequency correlation file
            cfile = os.path.join(
                workdir, 'crqnm_{:s}_{:d}.npy'.format(tag, corr_nmd))
            
            if not os.path.exists(cfile) or recalc_correlation:
                
                # Load data
                #inmlst = np.load(ifile)
                qnmlst = np.load(qfile)
                # Correct list - keep 1900 < vas(2200) < 2500
                #mask = np.logical_and(
                    #inmlst[:, -1] > 2500.,
                    #inmlst[:, -1] < 1900.)
                #inmlst[mask, :] = np.nan
                #inmlst = inmlst[~np.isnan(inmlst[:, -1]), :]
                mask = np.logical_and(
                    qnmlst[:, -1] > 2500.,
                    qnmlst[:, -1] < 1900.)
                qnmlst[mask, :] = np.nan
                #qnmlst = qnmlst[~np.isnan(qnmlst[:, -1]), corr_nmd - 1]
                qnmlst = qnmlst[:, corr_nmd - 1]
                print(qnmlst.shape)
                # Compute correlation
                corr_qnm = acovf(qnmlst, fft=True, missing='conservative')
                
                # Save correlation
                np.save(cfile, corr_qnm)
                
            else:
                
                corr_qnm = np.load(cfile)
                
            # Add correlation
            corr_qnms[:len(corr_qnm)] += corr_qnm
            
        # Average correlation
        #corr_qnms /= len(Smpls[Ti][Vmi])
        
        # Pressure ratio
        p = M/Vmi/rhocrit
            
        # Plot selection
        select_plt = np.logical_and(
            np.logical_not(np.isnan(corr_qnms)),
            np.logical_and(
                tlst*1e-3 < pltt,
                tlst*1e-3 > 0.0))
            
        # Plot
        
        label = r'{:.0f} ($\rho*$={:.2f})'.format(Vmi, p)
        
        if plt_log:
            
            log_corr_qnms = np.zeros_like(corr_qnms)
            log_corr_qnms[corr_qnms > 0.0] = np.log(corr_qnms[corr_qnms > 0.0])
            log_corr_qnms[corr_qnms <= 0.0] = np.nan
            
            axs1.plot(
                tlst[select_plt]*1e-3, log_corr_qnms[select_plt], '-', 
                color=color_scheme[iv], label=label)
            
        else:
            
            axs1.plot(
                tlst[select_plt]*1e-3, corr_qnms[select_plt], '-', 
                color=color_scheme[iv], label=label)
            
        if np.nanmax(corr_qnms[select_plt]) > cmax:
            cmax = np.nanmax(corr_qnms[select_plt])
        if np.nanmin(corr_qnms[select_plt]) < cmin:
            cmin = np.nanmin(corr_qnms[select_plt])
        
    axs1.set_xlim([-pltt/20.0, pltt])
    
    fig.suptitle(
        'Quenched normal mode {:d} autocorrelation function of '.format(
            corr_nmd)
        + r'N$_2$O ' + '\n'
        + 'in {:s} at '.format(sys_tag[sys_solv]) 
        + '{:.1f} K'.format(float(Ti)),
        fontweight='bold')
    
    axs1.legend(
        loc=[1.05, -0.20], 
        title=r'$V_m$ of {:s} (cm$^3$/mol)'.format(sys_tag[sys_solv]))
    
    axs1.set_xlabel(r'Time (ps)', fontweight='bold')
    axs1.get_xaxis().set_label_coords(0.5, -0.1)
    if plt_log:
        ylabel = (
            r'ln($\left< \left| \mu_{as}(0) \right|~\left| \mu_{as} \right| \right>$) '
            + r'(ln(ps$^{-1}$))')
        axs1.set_ylabel(r'{:s}'.format(ylabel), fontweight='bold')
    else:
        ylabel = (
            r'$\left< \left| \mu_{as}(0) \right|~\left| \mu_{as}(t) \right| \right>$ ' 
            + r'(ps$^{-1}$)')
        axs1.set_ylabel(ylabel, fontweight='bold')
    axs1.get_yaxis().set_label_coords(-0.08, 0.50)
    
    if plt_log:
        axs1.set_ylim([-5, 5.0])
    #else:
        #dc = cmax - cmin
        #axsi.set_ylim([cmin - dc*0.1, cmax + dc*0.1])
    
    if plt_log:
        figtitle = 'QNM_{:d}_corr_{:d}K_log.png'.format(corr_nmd, int(Ti))
    else:
        figtitle = 'QNM_{:d}_corr_{:d}K.png'.format(corr_nmd, int(Ti))
    
    fig.savefig(
        os.path.join(
            maindir, rsltdir, figtitle),
        format='png', dpi=dpi)
    plt.close(fig)

    
    
