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
TINY_SIZE = 9
XSMALL_SIZE = 11
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
# Correlation and lifetimes
#---------------------------------

recalc_correlation = False

# Figure arrangement
figsize = (12, 7.2)
left = 0.10
bottom = 0.09
row = np.array([0.35, 0.12])
column = np.array([0.24, 0.23])

# Plot FFCF
fig = plt.figure(figsize=figsize)

figlabels = ['A', 'B', 'C', 'D']


corr_nmd = 9

tstart = 0.10
fitt = 2.0
pltt = 2.0
#ftype = "bicosfunc"
ftype = "bifunc"
#ftype = "singlefunc"

def bicosfunc(x, a1, b1, t1, a2, t2, c):
    return a1 * np.cos(b1*x) * np.exp(-x/t1) + a2 * np.exp(-x/t2) + c
def bifunc(x, a1, t1, a2, t2, c):
    return a1 * np.exp(-x/t1) + a2 * np.exp(-x/t2) + c
def singlefunc(x, a1, t1, c):
    return a1 * np.exp(-x/t1) + c

def bicosfunc_cons(x, a1, b1, t1, t2, c):
    return a1 * np.cos(b1*x) * np.exp(-x/t1) + (1.0 - a1) * np.exp(-x/t2) + c
def bifunc_const(x, a1, t1, t2, c):
    return a1 * np.exp(-x/t1) + (1.0 - a1) * np.exp(-x/t2) + c

offset = 0.01


# Fit results
A_list_slv     = []
tau_list_slv   = []
b_list_slv     = []
delta_list_slv = []
CM_list_slv    = []

# Iterate over systems
for islv, slvi in enumerate(sys_solv):
    
    axs1 = fig.add_axes([
        left + islv*np.sum(column), bottom + np.sum(row), column[0], row[0]])
    
    # Fit results
    A_list = []
    tau_list = []
    b_list = []
    delta_list = []
    CM_list = []
    
    for iv, Vmi in enumerate(Vm[islv]):
        print(slvi, Vmi)
        
        for ismpl in ismpls:
            
            # Working directory
            workdir = os.path.join(
                sys_sdir[islv], 
                'T{:d}'.format(int(T[islv])), 
                'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
            # System tag
            tag = '{:d}_{:d}'.format(int(T[islv]), int(Vmi))
            
            # Quenched normal mode frequencies file
            qfile = os.path.join(
                workdir, 'qnmfr_{:s}.npy'.format(tag))
            
            # Time file
            tfile = os.path.join(
                workdir, 'qnmtm_{:s}.npy'.format(tag))
            
            # Load time list
            tlst = np.load(tfile)
            
            if ismpl == 0:
                
                # Correlation arrays
                corr_qnms = np.zeros(tlst.shape[0], dtype=float)
                
            # Quenched normal mode frequency correlation file
            cfile = os.path.join(
                workdir, 'crqnm_{:s}_{:d}.npy'.format(tag, corr_nmd))
            
            if not os.path.exists(cfile) or recalc_correlation:
                
                # Load data
                qnmlst = np.load(qfile)
                mask = np.logical_and(
                    qnmlst[:, -1] > 2500.,
                    qnmlst[:, -1] < 1900.)
                qnmlst[mask, :] = np.nan
                qnmlst = qnmlst[:, corr_nmd - 1]
                
                # Compute correlation
                corr_qnm = acovf(qnmlst, fft=True, missing='conservative')
                
                # Save correlation
                np.save(cfile, corr_qnm)
                
            else:
                
                corr_qnm = np.load(cfile)
            
            # Add correlation
            corr_qnms[:len(corr_qnm)] += corr_qnm
            
        # Average correlation
        corr_qnms /= len(ismpls)
        
        # Normalize correlation
        corr_qnms /= corr_qnms[0]
        
        select_fit = np.logical_and(
                np.logical_not(np.isnan(corr_qnms)),
                np.logical_and(
                    tlst >= tstart,
                    tlst <= fitt)
            )
        select_fit = np.logical_and(
            tlst >= tstart,
            tlst <= fitt)
        
        sigma = 1./(1. + tlst**(1/2))
        
        try:
            
            if ftype == "bicosfunc":
                
                # Initial guess
                popt = [0.10, 1.0/np.pi, 0.5, 0.05, 0.01]
                
                # Fit procedure
                popt, pcov = curve_fit(
                    bicosfunc_cons, 
                    tlst[select_fit], 
                    corr_qnms[select_fit], 
                    p0=popt,
                    sigma=sigma[select_fit],
                    bounds=[
                        [0.0, 0.0, 0.0, 0.0, 0.0], 
                        [np.inf, np.inf, np.inf, np.inf, 0.01]])
                    
                popt = [
                    popt[0], popt[1], popt[2], 
                    1.0 - popt[0], popt[3], 
                    popt[-1]]
                    
            elif ftype == "bifunc":
                
                # Initial guess
                popt = [0.025, 0.18, 0.09, 0.01]
                
                # Fit procedure
                popt, pcov = curve_fit(
                    bifunc_const, 
                    tlst[select_fit], 
                    corr_qnms[select_fit], 
                    p0=popt,
                    sigma=sigma[select_fit],
                    bounds=[
                        [0.0, 0.0, 0.0, 0.0], 
                        [1.0, np.inf, np.inf, 0.01]])
                    
                popt = [popt[0], popt[1], 1.0 - popt[0], popt[2], popt[-1]]
                
            elif ftype == "singlefunc":
                
                # Initial guess
                popt = [1.0, 0.05, 0.001]
                
                # Fit procedure
                popt, pcov = curve_fit(
                    singlefunc, 
                    tlst[select_fit], 
                    corr_qnms[select_fit], 
                    p0=popt,
                    sigma=sigma[select_fit],
                    bounds=[
                        [0.0, 0.0, 0.0], 
                        [np.inf, np.inf, 0.005]])
                    
        except (RuntimeError, ValueError):
            
            fit_complete = False
            
        else:
            
            fit_complete = True
            
        if ftype == "bicosfunc":
            
            label_amplitute = (
                'A = ' + (('{:.1e}, '*2).format(popt[0], popt[3]))[:-2])
            label_b = (
                'b = ' + (('{:.1e}, '*1).format(popt[1]))[:-2]
                + r' ps$^{-1}$')
            label_tau = (
                r'$\tau$ = ' + (('{:.1e}, '*2).format(popt[2], popt[4]))[:-2]
                + r' ps')
            label_delta = (
                r'$\Delta$ = ' + (('{:.1e}, '*1).format(popt[-1]))[:-2]
                + r' ps')
            
            A_list.append((popt[0], popt[3]))
            tau_list.append((popt[2], popt[4]))
            b_list.append(popt[1])
            delta_list.append(popt[-1])
            CM_list.append(CM[islv][iv])
            
            print(label_amplitute)
            print(label_b)
            print(label_tau)
            print(label_delta)
        
        if ftype == "bifunc":
            
            #asort = np.argsort((popt[0], popt[2]))[::-1]
            asort = np.argsort((popt[1], popt[3]))[::-1]
            i1, i2 = np.array([0, 2])[asort]
            
            label_amplitute = (
                'A = ' + (('{:.1e}, '*2).format(
                    popt[i1], popt[i2]))[:-2])
            label_tau = (
                r'$\tau$ = ' + (('{:.1e}, '*2).format(
                    popt[i1 + 1], popt[i2 + 1]))[:-2]
                + r' ps$^{-1}$')
            label_delta = (
                r'$\Delta$ = ' + (('{:.1e}, '*1).format(popt[-1]))[:-2]
                + r' ps')
            
            A_list.append((popt[i1], popt[i2]))
            tau_list.append((popt[i1 + 1], popt[i2 + 1]))
            delta_list.append(popt[-1])
            CM_list.append(CM[islv][iv])
            
            print(label_amplitute)
            print(label_tau)
            print(label_delta)
        
        elif ftype == "singlefunc":
            label_amplitute = (
                'A = ' + (('{:.1e}, '*1).format(popt[0]))[:-2])
            label_tau = (
                r'$\tau$ = ' + (('{:.1e}, '*1).format(popt[1]))[:-2]
                + r' ps$^{-1}$')
            label_delta = (
                r'$\Delta$ = ' + (('{:.1e}, '*1).format(popt[-1]))[:-2]
                + r' ps')
            
            A_list.append(popt[0])
            tau_list.append(popt[1])
            delta_list.append(popt[-1])
            CM_list.append(CM[islv][iv])
            
            print(label_amplitute)
            print(label_tau)
            print(label_delta)
            
        # Pressure ratio
        p = M[islv]/Vmi/rhocrit[islv]
        label = r'{:.2f} [{:.2f}]'.format(CM[islv][iv], p)
    
        # Plot selection
        select_plt = np.logical_and(
            np.logical_not(np.isnan(corr_qnms)),
            np.logical_and(
                tlst <= pltt,
                tlst >= 0.0))
            
        if fit_complete:
            if ftype == "bicosfunc":
                fit_corr = bicosfunc(tlst[select_plt], *popt)
            elif ftype == "bifunc":
                fit_corr = bifunc(tlst[select_plt], *popt)
            elif ftype == "singlefunc":
                fit_corr = singlefunc(tlst[select_plt], *popt)
            axs1.plot(
                tlst[select_plt], fit_corr + offset*iv, '--', 
                color=color_scheme[iv])
        
        axs1.plot(
            tlst[select_plt], corr_qnms[select_plt] + offset*iv, '-', 
            color=color_scheme[iv], label=label)
        
    A_list_slv.append(A_list)
    if ftype == "bicosfunc":
        b_list_slv.append(b_list)
    tau_list_slv.append(tau_list)
    delta_list_slv.append(delta_list)
    CM_list_slv.append(CM_list)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    order = list(range(len(Vm[islv])))[::-1]
    axs1.legend(
        [handles[idx] for idx in order], [labels[idx] for idx in order],
        loc=(1.08, -0.30),  framealpha=1.0,
        title=(
            r'$c$ (M) of {:s} [$\rho*$]'.format(sys_tag[slvi])
            ),
        title_fontsize=MEDIUM_SIZE
        )
    
    axs1.set_title(r'N$_2$O:{:s}'.format(sys_tag[slvi]), fontweight='bold')
    
    axs1.set_xlabel(r'Time (ps)', fontweight='bold')
    axs1.get_xaxis().set_label_coords(0.5, -0.15)
    if islv == 0:
        axs1.set_ylabel(r'FFCF($\nu_\mathrm{as}$)', fontweight='bold')
        axs1.get_yaxis().set_label_coords(-0.20, 0.50)
    
    tbox = TextArea(
        figlabels[islv*2], textprops=dict(color='k', fontsize=MEDIUM_SIZE))

    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.90, 0.95),
        bbox_transform=axs1.transAxes, borderpad=0.)
    
    axs1.add_artist(anchored_tbox)
    
    axs1.set_xlim([0.0, pltt])
    axs1.set_ylim([-0.2*0.01, 0.2*1.1])
    
    axs1.set_yticks(np.array([ii for ii in order])*offset)
    axs1.set_yticklabels([])
    axs1.yaxis.tick_right()


    scale = 1.5
    axs2 = fig.add_axes([
        left + islv*np.sum(column), bottom, column[0]*scale, row[0]])
    
    
    # Plot
    tau_list = np.array(tau_list)
    label = r'$\tau_\mathrm{slow}}$'
    axs2.plot(
        CM_list, tau_list[:, 0], 'o-b', label=label)
    label = r'$\tau_\mathrm{fast}}$'
    axs2.plot(
        CM_list, tau_list[:, 1], 's-r', label=label)
    
    
    axs2.set_xlim([0.0, np.max([np.max(CMi) for CMi in CM])*1.2])
    tau_max = np.max(tau_list)*1.1
    tau_max = 0.5*1.2
    axs2.set_ylim([0.0, tau_max])
    
    axs2.plot(
        [ccrit_sim[islv], ccrit_sim[islv]], [0.0, tau_max], 
        'k', ls='dotted',
        label=r'c$_\mathrm{crit}$ (Sim.)')
    
    concrit = 1000.*rhocrit[islv]/M[islv]
    axs2.plot(
        [concrit, concrit], [0.0, tau_max], 
        'k', ls=(0, (5, 5)),
        label=r'c$_\mathrm{crit}$ (Exp.)')
    
    axs2.set_xlabel(
        '{:s} Concentration c (M)'.format(sys_tag[slvi]), fontweight='bold')
    axs2.get_xaxis().set_label_coords(0.50, -0.15)
    if islv == 0:
        axs2.set_ylabel(r'$\tau$ (ps)', fontweight='bold')
        axs2.get_yaxis().set_label_coords(-0.20/scale, 0.50)
        
    if islv == 0:
        axs2.legend(
            loc='upper right')
    else:
        axs2.legend(
            loc='center right')
        
    tbox = TextArea(
        figlabels[islv*2 + 1], textprops=dict(color='k', fontsize=BIGGER_SIZE))

    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.05, 0.95),
        bbox_transform=axs2.transAxes, borderpad=0.)
    
    axs2.add_artist(anchored_tbox)
        
plt.savefig(
    os.path.join(resdir, 'ffcf.png'), 
    format='png', dpi=dpi)
plt.close()   
#plt.show()



















# Figure arrangement
figsize = (12, 10)
left = 0.10
bottom = 0.08
row = np.array([0.25, 0.05])
column = np.array([0.36, 0.10])

# Plot FFCF
fig = plt.figure(figsize=figsize)

figlabels = ['A', 'B', 'C', 'D', 'E', 'F']

A_max = 1.0
tau_max = np.max([np.max(tau_list) for tau_list in tau_list_slv])*1.1
delta_max = np.max([np.max(delta_list) for delta_list in delta_list_slv])*1.1

# Iterate over systems
for islv, slvi in enumerate(sys_solv):
    
    axs1 = fig.add_axes([
        left + islv*np.sum(column), bottom + 2*np.sum(row), column[0], row[0]])
    axs2 = fig.add_axes([
        left + islv*np.sum(column), bottom + 1*np.sum(row), column[0], row[0]])
    axs3 = fig.add_axes([
        left + islv*np.sum(column), bottom + 0*np.sum(row), column[0], row[0]])
    
    xlim = [0.0, np.max([np.max(CM_list) for CM_list in CM_list_slv])*1.1]
    concrit = 1000.*rhocrit[islv]/M[islv]
    
    # Amplitute
    label = r'$A_\mathrm{slow}}$'
    axs1.plot(
        CM_list_slv[islv], np.array(A_list_slv[islv])[:, 0], 'o-b', label=label)
    label = r'$A_\mathrm{fast}}$'
    axs1.plot(
        CM_list_slv[islv], np.array(A_list_slv[islv])[:, 1], 's-r', label=label)
    
    axs1.set_xlim(xlim)
    axs1.set_ylim([0.0, A_max])
    
    label = r'c$_\mathrm{crit}$ (Sim.)'
    axs1.plot(
        [ccrit_sim[islv], ccrit_sim[islv]], [0.0, A_max], 
        'k', ls='dotted',
        label=label)
    
    label = r'c$_\mathrm{crit}$ (Exp.)'
    axs1.plot(
        [concrit, concrit], [0.0, A_max],
        'k', ls=(0, (5, 5)),
        label=label)
    
    axs1.set_ylabel(r'$A$', fontweight='bold')
    axs1.get_yaxis().set_label_coords(-0.15, 0.50)
    
    if islv:
        axs1.legend(loc='center right')
    
    axs1.set_title(r'N$_2$O:{:s}'.format(sys_tag[slvi]), fontweight='bold')
    
    # Lifetime
    label = r'$\tau_\mathrm{slow}}$'
    axs2.plot(
        CM_list_slv[islv], np.array(tau_list_slv[islv])[:, 0], 'o-b', label=label)
    label = r'$\tau_\mathrm{fast}}$'
    axs2.plot(
        CM_list_slv[islv], np.array(tau_list_slv[islv])[:, 1], 's-r', label=label)
    
    axs2.set_xlim(xlim)
    axs2.set_ylim([0.0, tau_max])
    
    label = r'c$_\mathrm{crit}$ (Sim.)'
    axs2.plot(
        [ccrit_sim[islv], ccrit_sim[islv]], [0.0, tau_max], 
        'k', ls='dotted',
        label=label)
    
    label = r'c$_\mathrm{crit}$ (Exp.)'
    axs2.plot(
        [concrit, concrit], [0.0, tau_max],
        'k', ls=(0, (5, 5)),
        label=label)
    
    axs2.set_ylabel(r'$\tau$ (ps)', fontweight='bold')
    axs2.get_yaxis().set_label_coords(-0.15, 0.50)
    
    if islv:
        axs2.legend(
            loc='center right')
    #if islv == 0:
        #axs2.legend(
            #loc='upper right')
    #else:
        #axs2.legend(
            #loc='center right')
    
    # Delta
    label = r'$\Delta$'
    axs3.plot(
        CM_list_slv[islv], delta_list_slv[islv], 'd-k', label=label)
    
    axs3.set_xlim(xlim)
    axs3.set_ylim([0.0, delta_max])
    
    label = r'c$_\mathrm{crit}$ (Sim.)'
    axs3.plot(
        [ccrit_sim[islv], ccrit_sim[islv]], [0.0, delta_max], 
        'k', ls='dotted',
        label=label)
    
    label = r'c$_\mathrm{crit}$ (Exp.)'
    axs3.plot(
        [concrit, concrit], [0.0, delta_max],
        'k', ls=(0, (5, 5)),
        label=label)
    
    axs3.set_xlabel(
        '{:s} Concentration c (M)'.format(sys_tag[slvi]), fontweight='bold')
    axs3.get_xaxis().set_label_coords(0.50, -0.15)
    
    axs3.set_ylabel(r'$\Delta$', fontweight='bold')
    axs3.get_yaxis().set_label_coords(-0.15, 0.50)
    
    if islv:
        axs3.legend(loc='upper right')
    
    
    
    tbox = TextArea(
        figlabels[islv*1 + 0], textprops=dict(color='k', fontsize=BIGGER_SIZE))
    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.05, 0.90),
        bbox_transform=axs1.transAxes, borderpad=0.)
    axs1.add_artist(anchored_tbox)
    
    tbox = TextArea(
        figlabels[islv*1 + 2], textprops=dict(color='k', fontsize=BIGGER_SIZE))
    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.05, 0.95),
        bbox_transform=axs2.transAxes, borderpad=0.)
    axs2.add_artist(anchored_tbox)
    
    tbox = TextArea(
        figlabels[islv*1 + 4], textprops=dict(color='k', fontsize=BIGGER_SIZE))
    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.05, 0.95),
        bbox_transform=axs3.transAxes, borderpad=0.)
    axs3.add_artist(anchored_tbox)
    
plt.savefig(
    os.path.join(resdir, 'si_ffcf_fit.png'), 
    format='png', dpi=dpi)
plt.close()   

        
