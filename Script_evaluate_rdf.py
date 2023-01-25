# Basics
import os
import sys
import numpy as np
from glob import glob
import scipy
from scipy import stats

# ASE - Basics
from ase import Atoms

# MDAnalysis
import MDAnalysis

# Statistics
from statsmodels.tsa.stattools import acovf

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

# Result directory
resdir = "paper_figures"
if not os.path.exists(resdir):
    os.mkdir(resdir)

# System information
sys_sdir = [
    'N2O_Xe',
    'N2O_SF6']
sys_solv = ['XE', 'SF6']
sys_trgt = 'N2O'
sys_tag = {
    'N2O': r'N$_2$O',
    'SF6': r'SF$_6$',
    'XE': 'Xe'}
ismpls = [0]

# Temperatures [K]
T = [291.18, 321.93]

# Critical temperature and density of sys_solv
M = [131.293, 146.05] # g/mol
Tcrit = [289.73, 273.15 + 45.6] # K
rhocrit = [1.10, 0.74] # g/cm**3

# Relative density
rhostar = [
    [0.04, 0.10, 0.15, 0.37, 0.49, 0.62, 0.66, 0.75, 0.87, 0.93, 1.33, 1.55],
    [0.04, 0.10, 0.16, 0.30, 0.67, 0.86, 0.99, 1.17, 1.36, 1.51, 1.87]
    ]

# State
slv_states = [
    [
        'Gas', 'Gas', 'Gas', 'Gas', 'Gas', 
        'SCF', 'SCF', 'SCF', 'SCF', 'SCF', 'SCF', 
        'Liquid'],
    [
        'Gas', 'Gas', 'Gas', 'Gas', 
        'SCF', 'SCF', 'SCF', 'SCF', 'SCF', 'SCF',
        'Liquid']
    ]

# Molare Volume and concentration
Vm = []
CM = []
for islv, slvi in enumerate(sys_solv):
    Vmi = []
    for rhotemp in rhostar[islv]:
        Vmi.append(M[islv]/np.array(rhotemp)/rhocrit[islv])
    Vm.append(Vmi)
    
    concrit = 1000.*rhocrit[islv]/M[islv]
    CM.append([np.array(rhostar_i)*concrit for rhostar_i in rhostar[islv]])

# Simulated critical concentrations
ccrit_sim = [
    5.19,
    5.06]

# Solvent phase change
slv_change = [
    [(CM[0][5] + CM[0][4])/2., (CM[0][11] + CM[0][10])/2.],
    [(CM[1][4] + CM[1][3])/2., (CM[1][10] + CM[1][9])/2.]
    ]

# First solvation shell range
slv_sol_dfrst = [6.5, 7.5]
slv_slv_dfrst = [6.6, 8.0]

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
TINY_SIZE = 8
XSMALL_SIZE = 10
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize

dpi = 200

color_scheme = [
    'b', 'r', 'g', 'purple', 'orange', 'magenta', 'brown', 'darkblue',
    'darkred', 'darkgreen', 'darkgrey', 'olive']


#---------------------------------
# Plot Histogram - Coordination
#---------------------------------

# Figure arrangement
figsize = (6, 4)
left = 0.15
bottom = 0.15
row = np.array([0.70, 0.00])
column = np.array([0.35, 0.10])

# Distance binning density in 1/Angstrom
dst_step = 0.1

# Max plotting distance
dst_mplt = 10.0

# Plot Coordination number
fig = plt.figure(figsize=figsize)

figlabels = ['A', 'B']

# Iterate over systems
for islv, slvi in enumerate(sys_solv):
    
    # Initialize axes
    axs1 = fig.add_axes([left + islv*np.sum(column), bottom, column[0], row[0]])
    
    # Coordination number first solvation shell
    nloc = []
    
    # Radial distribution function
    nlfile = os.path.join(
        resdir, 'n_all_{:s}.npy'.format(slvi))
    
    if not os.path.exists(nlfile):
    
        for iv, Vmi in enumerate(Vm[islv]):
            print(slvi, Vmi)
            # Radial distribution functions of each sample
            gsmpls = []
            rsmpls = []
            
            # Sample counter
            Nsmpls = 0
            
            for ismpl in ismpls:
                
                # Working directory
                workdir = os.path.join(
                    sys_sdir[islv], 
                    'T{:d}'.format(int(T[islv])), 
                    'V{:d}_{:d}'.format(int(Vmi), ismpl))
                
                # System tag
                tag = '{:d}_{:d}'.format(int(T[islv]), int(Vmi))
                
                # Radial distribution function
                gfile = os.path.join(
                    workdir, 'g_all_{:s}.npy'.format(tag))
                
                # Solute - Solvent distance histogram file
                nfile = os.path.join(
                    workdir, 'nhist_{:s}.npy'.format(tag))
                
                # Time file
                tfile = os.path.join(
                    workdir, 'ntist_{:s}.npy'.format(tag))
                
                # Load data
                g = np.load(gfile)
                nlst = np.load(nfile)
                tlst = np.load(tfile)
                
                # Sum time steps
                nlst = np.sum(nlst, axis=0)
                
                # Radial grid
                dst_rnge = [0.0, len(nlst)*dst_step]
                dst_bins = np.arange(0.0, dst_rnge[-1] + dst_step/2., dst_step)
                dst_cntr = dst_bins[:-1] + dst_step/2.
                
                # Normalize
                N = np.sum(nlst)/len(tlst)
                V = 4./3.*np.pi*dst_rnge[1]**3
                
                # Append result to list
                gsmpls.append(g)
                rsmpls.append(N/V)
                Nsmpls += 1
                
            # Get sample average
            dst_Nstp = np.min([len(gsmpl) for gsmpl in gsmpls if len(gsmpl)])
            g = np.zeros(dst_Nstp, dtype=float)
            for gsmpl in gsmpls:
                g += gsmpl[:dst_Nstp]/Nsmpls
            rho_local = np.mean(rsmpls)
            
            # Radial grid
            dst_rnge = [0.0, len(g)*dst_step]
            dst_bins = np.arange(0.0, dst_rnge[-1] + dst_step/2., dst_step)
            dst_cntr = dst_bins[:-1] + dst_step/2.
            
            # Average N in First solvation shell
            n_first = 4.0*np.pi*rho_local*(
                np.sum(
                    g[dst_cntr <= slv_sol_dfrst[islv]]
                    *dst_cntr[dst_cntr <= slv_sol_dfrst[islv]]**2)
                *dst_step)
            
            nloc.append(n_first)
        
        np.save(nlfile, nloc)
        
    else:
        
        nloc = np.load(nlfile)
    
    # Max values
    CMmax = np.max([np.max(CMi) for CMi in CM])
    intcmax = int(np.ceil(CMmax)) + 2
    
    # Fit linear
    x = np.array([-intcmax*0.02, intcmax*1.02])
    res1 = stats.linregress(CM[islv][:3], nloc[:3])
    res2 = stats.linregress(CM[islv][-3:], nloc[-3:])
    print(res1)
    print(res2)
    fitfunc = lambda x, p0: p0 * x  
    p1, pcov1 = scipy.optimize.curve_fit(
        fitfunc, CM[islv][:3], nloc[:3])
    fitfunc = lambda x, p0, p1: p0 * x + p1  
    p2, pcov2 = scipy.optimize.curve_fit(
        fitfunc, CM[islv][-3:], nloc[-3:])
    print(p1)
    print(p2)
    
    #axs1.plot(x, res1.intercept + res1.slope*x, '--r', lw=1)
    #axs1.plot(x, res2.intercept + res2.slope*x, '--b', lw=1)
    axs1.plot(
        x, p1[0]*x, 'r', lw=1, ls='dashed', 
        label=r"$N(c) = {:.2f} c$".format(p1[0]))
    axs1.plot(
        x, p2[1] + p2[0]*x, 'b', lw=1, ls='dashdot', 
        label=r"$N(c) = {:.2f} c + {:.2f}$".format(*p2))
    
    # Plot Coordination number
    axs1.plot(CM[islv], nloc, 'o-k')
    
    axs1.set_xticks([ii for ii in range(0, int(intcmax), 4)])
    nmax = int(np.ceil(np.max(nloc))) 
    axs1.set_yticks([ii for ii in range(0, nmax + 1, 2)])

    axs1.set_xlim([-intcmax*0.02, intcmax*1.02])
    axs1.set_ylim([0.0, np.ceil(nmax*1.0)*1.1])
    
    axs1.set_title(r'N$_2$O:{:s}'.format(sys_tag[slvi]), fontweight='bold')
    axs1.set_xlabel(r'$c$ (M)', fontweight='bold')
    axs1.get_xaxis().set_label_coords(0.5, -0.10)
    if islv == 0:
        axs1.set_ylabel(
            r"$\langle N(r') \rangle$",#.format(slv_sol_dfrst[islv]), 
            fontweight='bold')
        axs1.get_yaxis().set_label_coords(-0.20, 0.50)

    tbox = TextArea(
        figlabels[islv], textprops=dict(color='k', fontsize=BIGGER_SIZE))

    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.05, 0.95),
        bbox_transform=axs1.transAxes, borderpad=0.)

    axs1.add_artist(anchored_tbox)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = list(range(2))[::-1]
    axs1.legend(
        [handles[idx] for idx in order], [labels[idx] for idx in order],
        loc='lower right', fontsize=TINY_SIZE, framealpha=1.0)
    
    axs1.plot(
        [ccrit_sim[islv], ccrit_sim[islv]], [0.0, np.ceil(nmax*1.0)*1.1], 
        'k', ls='dotted',
        label=r'c$_\mathrm{crit}$ (Sim.)')
    
    #concrit = 1000.*rhocrit[islv]/M[islv]
    #axs1.plot(
        #[concrit, concrit], [0.0, np.ceil(nmax*1.0)*1.1], 
        #'k', ls=(0, (5, 5)),
        #label=r'c$_\mathrm{crit}$ (Exp.)')
    
    axs1.plot([slv_change[islv][0]]*2, [0.0, np.ceil(np.max(nloc)*1.1)], '--k')
    axs1.plot([slv_change[islv][1]]*2, [0.0, np.ceil(np.max(nloc)*1.1)], '--k')
    
    # Add system state
    state_dict = {
        'style'  : 'normal',
        'weight' : 'bold',
        'size'   : TINY_SIZE}
    axs1.text(
        slv_change[islv][0]/2.2, 
        8.0, 'Gas',
        va='bottom', ha='center',
        fontdict=state_dict,
        bbox=dict(
            boxstyle="round",
            ec="lightgray",
            fc=(1.0, 1.0, 1.0))
        )
    if islv == 0:
        axs1.text(
            (slv_change[islv][1] + slv_change[islv][0])/2.3, 
            9.0, 'SCF',
            va='bottom', ha='center',
            fontdict=state_dict,
            bbox=dict(
                boxstyle="round",
                ec="lightgray",
                fc=(1.0, 1.0, 1.0))
            )
    elif islv == 1:
        axs1.text(
            (slv_change[islv][1] + slv_change[islv][0])/2.5, 
            9.0, 'SCF',
            va='bottom', ha='center',
            fontdict=state_dict,
            bbox=dict(
                boxstyle="round",
                ec="lightgray",
                fc=(1.0, 1.0, 1.0))
            )
        axs1.text(
            (slv_change[islv][1] + slv_change[islv][0])/0.9, 
            8.0, 'Liquid',
            va='bottom', ha='center',
            fontdict=state_dict,
            bbox=dict(
                boxstyle="round",
                ec="lightgray",
                fc=(1.0, 1.0, 1.0))
            )
        
    

plt.savefig(os.path.join(resdir, 'n_solute_solvent.png'), format='png', dpi=dpi)
plt.close()
exit()

#---------------------------------
# Plot Histogram - Solute-Solvent
#---------------------------------

# Figure arrangement
figsize = (12, 6)
left = 0.05
bottom = 0.15
row = np.array([0.80, 0.00])
column = np.array([0.27, 0.20])

# Distance binning density in 1/Angstrom
dst_step = 0.1

# Max plotting distance
dst_mplt = 10.0

# Max values
gmin = 0.0
gmax = 0.0

# Plot INM histogram
fig = plt.figure(figsize=figsize)

figlabels = ['A', 'B', 'C', 'D']

# Iterate over systems
for islv, slvi in enumerate(sys_solv):
    
    # Initialize axes
    axs2 = fig.add_axes([
        left + islv*np.sum(column) + column[0]*1.20, 
        bottom + 3*row[0]/4., 
        column[1]*0.60, row[0]/4.])
    axs1 = fig.add_axes([left + islv*np.sum(column), bottom, column[0], row[0]])
    
    # Amplitude g
    gamp = []
    gcm = []
    
    # Coordination number first solvation shell
    nloc = []
    
    for iv, Vmi in enumerate(Vm[islv]):
        print(slvi, Vmi)
        # Radial distribution functions of each sample
        gsmpls = []
        rsmpls = []
        
        # Sample counter
        Nsmpls = 0
        
        for ismpl in ismpls:
            
            # Working directory
            workdir = os.path.join(
                sys_sdir[islv], 
                'T{:d}'.format(int(T[islv])), 
                'V{:d}_{:d}'.format(int(Vmi), ismpl))
            
            # System tag
            tag = '{:d}_{:d}'.format(int(T[islv]), int(Vmi))
            
            # Radial distribution function
            gfile = os.path.join(
                workdir, 'g_all_{:s}.npy'.format(tag))
            
            # Solute - Solvent distance histogram file
            nfile = os.path.join(
                workdir, 'nhist_{:s}.npy'.format(tag))
            
            # Time file
            tfile = os.path.join(
                workdir, 'ntist_{:s}.npy'.format(tag))
            
            # Load data
            g = np.load(gfile)
            nlst = np.load(nfile)
            tlst = np.load(tfile)
            
            # Sum time steps
            nlst = np.sum(nlst, axis=0)
            
            # Radial grid
            dst_rnge = [0.0, len(nlst)*dst_step]
            dst_bins = np.arange(0.0, dst_rnge[-1] + dst_step/2., dst_step)
            dst_cntr = dst_bins[:-1] + dst_step/2.
            
            # Normalize
            N = np.sum(nlst)/len(tlst)
            V = 4./3.*np.pi*dst_rnge[1]**3
            
            # Append result to list
            gsmpls.append(g)
            rsmpls.append(N/V)
            Nsmpls += 1
        
        # Get sample average
        dst_Nstp = np.min([len(gsmpl) for gsmpl in gsmpls if len(gsmpl)])
        g = np.zeros(dst_Nstp, dtype=float)
        for gsmpl in gsmpls:
            g += gsmpl[:dst_Nstp]/Nsmpls
        rho_local = np.mean(rsmpls)
        
        # Radial grid
        dst_rnge = [0.0, len(g)*dst_step]
        dst_bins = np.arange(0.0, dst_rnge[-1] + dst_step/2., dst_step)
        dst_cntr = dst_bins[:-1] + dst_step/2.
        
        # Pressure ratio
        p = M[islv]/Vmi/rhocrit[islv]
        
        label = r'{:.2f} [{:.2f}]'.format(CM[islv][iv], p)
        
        overlap = 1.5
        axs1.plot(
            dst_cntr, g + iv, '-', color=color_scheme[iv], label=label)
        
        # Average N in First solvation shell
        n_first = 4.0*np.pi*rho_local*(
            np.sum(
                g[dst_cntr <= slv_sol_dfrst[islv]]
                *dst_cntr[dst_cntr <= slv_sol_dfrst[islv]]**2)
            *dst_step)
        
        gamp.append(np.max(g))
        nloc.append(n_first)
        gcm.append(CM[islv][iv])
        
        # Add system state
        state_dict = {
            'style'  : 'normal',
            'weight' : 'bold',
            'size'   : 'small'}
        axs1.text(
            0.20, 
            iv + 0.1, '{:s}'.format(slv_states[islv][iv]),
            va='bottom', ha='left',
            fontdict=state_dict)
        
    
        
    axs1.set_xlim([0.0, dst_mplt])
    axs1.set_ylim([0.0, len(Vm[islv]) - 1 + np.max(g)*1.1])
    
    axs1.set_xticks([ii for ii in range(0, int(dst_mplt), 2)])
    
    intgmax = int(np.ceil(np.max(g)))
    axs1.set_yticks([ii for ii in range(len(Vm[islv]) - 1 + intgmax)])
    axs1.set_yticklabels([])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = list(range(len(Vm[islv])))[::-1]
    axs1.legend(
        [handles[idx] for idx in order], [labels[idx] for idx in order],
        loc=(1.05, -0.17),  framealpha=1.0,
        title=(
            #r'RDF (N$_2$O - {:s})'.format(sys_tag[slvi]) + '\n'
            r'$c$ (M) of {:s} [$\rho*$]'.format(sys_tag[slvi])
            )
        )
        
    axs1.set_xlabel(r'Distance $r$ ($\mathrm{\AA}$)', fontweight='bold')
    axs1.get_xaxis().set_label_coords(0.5, -0.1)
    if islv == 0:
        axs1.set_ylabel('g(r)', fontweight='bold')
        axs1.get_yaxis().set_label_coords(-0.10, 0.50)
    
    axs1.set_title(r'N$_2$O-{:s}'.format(sys_tag[slvi]), fontweight='bold')
    
    tbox = TextArea(
        figlabels[islv*2], textprops=dict(color='k', fontsize=MEDIUM_SIZE))

    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.02, 0.95),
        bbox_transform=axs1.transAxes, borderpad=0.)
    
    axs1.add_artist(anchored_tbox)
    print(nloc)
    #axs2.plot(gcm, gamp, 'o-k')
    axs2.plot(gcm, nloc, 'o-k')
    CMmax = np.max([np.max(CMi) for CMi in CM])
    
    intcmax = int(np.ceil(CMmax))
    axs2.set_xticks([ii for ii in range(0, int(intcmax), 4)])
    #intgmax = int(np.ceil(np.max(gamp)))
    #axs2.set_yticks([ii for ii in range(1, intgmax + 1)])
    intnloc = int(np.ceil(np.max(nloc)))
    axs2.set_yticks([ii for ii in range(0, intnloc + 1, 2)])
    
    axs2.set_xlim([-intcmax*0.02, intcmax*1.02])
    #axs2.set_ylim([1.0, np.ceil(np.max(gamp)*1.0)])
    axs2.set_ylim([0.0, np.ceil(np.max(nloc)*1.0)])
    
    axs2.set_xlabel(r'$c$ (M)', fontweight='bold', fontsize=SMALL_SIZE)
    axs2.get_xaxis().set_label_coords(0.5, -0.25)
    #axs2.set_ylabel(r'$g_\mathrm{max}$', fontweight='bold', fontsize=SMALL_SIZE)
    axs2.set_ylabel(
        r"$\langle N(r') \rangle$",#.format(slv_sol_dfrst[islv]), 
        fontweight='bold', fontsize=SMALL_SIZE)
    axs2.get_yaxis().set_label_coords(-0.25, 0.50)
    
    tbox = TextArea(
        figlabels[islv*2 + 1], textprops=dict(color='k', fontsize=MEDIUM_SIZE))

    #anchored_tbox = AnchoredOffsetbox(
        #loc='upper right', child=tbox, pad=0., frameon=False,
        #bbox_to_anchor=(0.95, 0.90),
        #bbox_transform=axs2.transAxes, borderpad=0.)
    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.05, 0.90),
        bbox_transform=axs2.transAxes, borderpad=0.)
    
    axs2.add_artist(anchored_tbox)
    
    axs2.plot([slv_change[islv][0]]*2, [0.0, np.ceil(np.max(nloc)*1.0)], ':k')
    axs2.plot([slv_change[islv][1]]*2, [0.0, np.ceil(np.max(nloc)*1.0)], ':k')
    
    # Add system state
    state_dict = {
        'style'  : 'normal',
        'weight' : 'bold',
        'size'   : 'xx-small'}
    axs2.text(
        slv_change[islv][0]/2., 
        6.0, 'Gas',
        va='bottom', ha='center',
        fontdict=state_dict)
    axs2.text(
        (slv_change[islv][1] + slv_change[islv][0])/2., 
        2.0, 'SCF',
        va='bottom', ha='center',
        fontdict=state_dict)
    
    
    
    
plt.savefig(os.path.join(resdir, 'g_solute_solvent.png'), format='png', dpi=dpi)
plt.close()




#----------------------------------
# Plot Histogram - Solvent-Solvent
#----------------------------------

# Figure arrangement
figsize = (12, 6)
left = 0.05
bottom = 0.15
row = np.array([0.80, 0.00])
column = np.array([0.27, 0.20])

# Distance binning density in 1/Angstrom
dst_step = 0.1

# Max plotting distance
dst_mplt = 15.0

# Max values
gmin = 0.0
gmax = 0.0

# Plot INM histogram
fig = plt.figure(figsize=figsize)

figlabels = ['A', "A'", 'B', "B'"]

# Iterate over systems
for islv, slvi in enumerate(sys_solv):
    
    # Initialize axes
    #axs2 = fig.add_axes([
        #left + islv*np.sum(column) + column[0]*1.20, 
        #bottom + 3*row[0]/4., 
        #column[1]*0.60, row[0]/4.])
    axs1 = fig.add_axes([left + islv*np.sum(column), bottom, column[0], row[0]])
    
    # Amplitude g
    gamp = []
    gcm = []
    
    # Coordination number first solvation shell
    nloc = []
    
    for iv, Vmi in enumerate(Vm[islv]):
        print(slvi, Vmi)
        # Radial distribution functions of each sample
        gsmpls = []
        rsmpls = []
        
        # Sample counter
        Nsmpls = 0
        
        for ismpl in ismpls:
            
            # Working directory
            workdir = os.path.join(
                sys_sdir[islv], 
                'T{:d}'.format(int(T[islv])), 
                'V{:d}_{:d}'.format(int(Vmi), ismpl))
            
            # System tag
            tag = '{:d}_{:d}'.format(int(T[islv]), int(Vmi))
            
            # Radial distribution function
            gfile = os.path.join(
                workdir, 'g_slv_{:s}.npy'.format(tag))
            
            # Solute - Solvent distance histogram file
            nfile = os.path.join(
                workdir, 'nhslv_{:s}.npy'.format(tag))
            
            # Time file
            tfile = os.path.join(
                workdir, 'ntslv_{:s}.npy'.format(tag))
            
            # Load data
            g = np.load(gfile)
            #nlst = np.load(nfile)
            #tlst = np.load(tfile)
            
            ## Sum time steps
            #nlst = np.sum(nlst, axis=0)
            
            ## Radial grid
            #dst_rnge = [0.0, len(nlst)*dst_step]
            #dst_bins = np.arange(0.0, dst_rnge[-1] + dst_step/2., dst_step)
            #dst_cntr = dst_bins[:-1] + dst_step/2.
            
            ## Normalize
            #N = np.sum(nlst)/len(tlst)
            #V = 4./3.*np.pi*dst_rnge[1]**3
            
            ## Append result to list
            gsmpls.append(g)
            #rsmpls.append(N/V)
            Nsmpls += 1
        
        # Get sample average
        dst_Nstp = np.min([len(gsmpl) for gsmpl in gsmpls if len(gsmpl)])
        g = np.zeros(dst_Nstp, dtype=float)
        for gsmpl in gsmpls:
            g += gsmpl[:dst_Nstp]/Nsmpls
        #rho_local = np.mean(rsmpls)
        
        # Radial grid
        dst_rnge = [0.0, len(g)*dst_step]
        dst_bins = np.arange(0.0, dst_rnge[-1] + dst_step/2., dst_step)
        dst_cntr = dst_bins[:-1] + dst_step/2.
        
        # Pressure ratio
        p = M[islv]/Vmi/rhocrit[islv]
        
        label = r'{:.2f} [{:.2f}]'.format(CM[islv][iv], p)
        
        overlap = 1.5
        axs1.plot(
            dst_cntr, g + iv, '-', color=color_scheme[iv], label=label)
        
        # Average N in First solvation shell
        #n_first = 4.0*np.pi*rho_local*(
            #np.sum(
                #g[dst_cntr <= slv_slv_dfrst[islv]]
                #*dst_cntr[dst_cntr <= slv_slv_dfrst[islv]]**2)
            #*dst_step)
        
        #gamp.append(np.max(g))
        #nloc.append(n_first)
        #gcm.append(CM[islv][iv])
        
        # Add system state
        state_dict = {
            'style'  : 'normal',
            'weight' : 'bold',
            'size'   : 'small'}
        axs1.text(
            0.20, 
            iv + 0.1, '{:s}'.format(slv_states[islv][iv]),
            va='bottom', ha='left',
            fontdict=state_dict)
        
    
        
    axs1.set_xlim([0.0, dst_mplt])
    axs1.set_ylim([0.0, len(Vm[islv]) - 1 + np.max(g)*1.1])
    
    axs1.set_xticks([ii for ii in range(0, int(dst_mplt), 4)])
    
    intgmax = int(np.ceil(np.max(g)))
    axs1.set_yticks([ii for ii in range(len(Vm[islv]) - 1 + intgmax)])
    axs1.set_yticklabels([])
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = list(range(len(Vm[islv])))[::-1]
    axs1.legend(
        [handles[idx] for idx in order], [labels[idx] for idx in order],
        loc=(1.05, -0.17),  framealpha=1.0,
        title=(
            #r'RDF (N$_2$O - {:s})'.format(sys_tag[slvi]) + '\n'
            r'$c$ (M) of {:s} [$\rho*$]'.format(sys_tag[slvi])
            )
        )
    
    axs1.set_xlabel(r'Distance $r$ ($\mathrm{\AA}$)', fontweight='bold')
    axs1.get_xaxis().set_label_coords(0.5, -0.1)
    if islv == 0:
        axs1.set_ylabel('g(r)', fontweight='bold')
        axs1.get_yaxis().set_label_coords(-0.10, 0.50)
    
    axs1.set_title(r'{0:s}-{0:s}'.format(sys_tag[slvi]), fontweight='bold')
    
    tbox = TextArea(
        figlabels[islv*2], textprops=dict(color='k', fontsize=MEDIUM_SIZE))

    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.02, 0.95),
        bbox_transform=axs1.transAxes, borderpad=0.)
    
    axs1.add_artist(anchored_tbox)
    #print(nloc)
    
    ##axs2.plot(gcm, gamp, 'o-k')
    #axs2.plot(gcm, nloc, 'o-k')
    #CMmax = np.max([np.max(CMi) for CMi in CM])
    
    #intcmax = int(np.ceil(CMmax))
    #axs2.set_xticks([ii for ii in range(0, int(intcmax), 4)])
    ##intgmax = int(np.ceil(np.max(gamp)))
    ##axs2.set_yticks([ii for ii in range(1, intgmax + 1)])
    #intnloc = int(np.ceil(np.max(nloc)))
    #axs2.set_yticks([ii for ii in range(0, intnloc + 1, 2)])
    
    #axs2.set_xlim([-intcmax*0.02, intcmax*1.02])
    ##axs2.set_ylim([1.0, np.ceil(np.max(gamp)*1.0)])
    #axs2.set_ylim([0.0, np.ceil(np.max(nloc)*1.0)])
    
    #axs2.set_xlabel(r'$c$ (M)', fontweight='bold', fontsize=SMALL_SIZE)
    #axs2.get_xaxis().set_label_coords(0.5, -0.25)
    ##axs2.set_ylabel(r'$g_\mathrm{max}$', fontweight='bold', fontsize=SMALL_SIZE)
    #axs2.set_ylabel(
        #r'$\langle N(r_\mathrm{first}) \rangle$',#.format(slv_slv_dfrst[islv]), 
        #fontweight='bold', fontsize=SMALL_SIZE)
    #axs2.get_yaxis().set_label_coords(-0.25, 0.50)
    
    #tbox = TextArea(
        #figlabels[islv*2 + 1], textprops=dict(color='k', fontsize=MEDIUM_SIZE))

    ##anchored_tbox = AnchoredOffsetbox(
        ##loc='upper right', child=tbox, pad=0., frameon=False,
        ##bbox_to_anchor=(0.95, 0.90),
        ##bbox_transform=axs2.transAxes, borderpad=0.)
    #anchored_tbox = AnchoredOffsetbox(
        #loc='upper left', child=tbox, pad=0., frameon=False,
        #bbox_to_anchor=(0.05, 0.90),
        #bbox_transform=axs2.transAxes, borderpad=0.)
    
    #axs2.add_artist(anchored_tbox)
    
    #axs2.plot([slv_change[islv][0]]*2, [0.0, np.ceil(np.max(nloc)*1.0)], ':k')
    #axs2.plot([slv_change[islv][1]]*2, [0.0, np.ceil(np.max(nloc)*1.0)], ':k')
    
    ## Add system state
    #state_dict = {
        #'style'  : 'normal',
        #'weight' : 'bold',
        #'size'   : 'xx-small'}
    #axs2.text(
        #slv_change[islv][0]/2., 
        #6.0, 'Gas',
        #va='bottom', ha='center',
        #fontdict=state_dict)
    #axs2.text(
        #(slv_change[islv][1] + slv_change[islv][0])/2., 
        #2.0, 'SCF',
        #va='bottom', ha='center',
        #fontdict=state_dict)
    
    
    
    
plt.savefig(
    os.path.join(resdir, 'g_solvent_solvent.png'), 
    format='png', dpi=dpi)
plt.close()





