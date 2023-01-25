# Test Script to import PhysNet as energy function in CHARMM via PyCHARMM

# Basics
import os
import sys
import numpy as np

# Matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

# Result directory
resdir = "paper_figures"
if not os.path.exists(resdir):
    os.mkdir(resdir)

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

sys_fdir = [
    'ReferenceCalcs/Fit_n2o_xe',
    'ReferenceCalcs/Fit_n2o_sf6']

# Get potential boundaries
emin = 0.0
emax = 0.0
for idir, fdir in enumerate(sys_fdir):

    fdata = np.load(os.path.join(fdir, "save_results.npz"))
    
    fit_Vnbond = fdata['fit_Vnbond']
    fit_potential = fdata['fit_potential']
    fit_rmse  = fdata['fit_rmse']
    threshold = fdata['threshold']
    
    # Range
    emin = np.min([np.min(fit_Vnbond), np.min(fit_potential), emin])
    emax = np.max([np.max(fit_Vnbond), np.max(fit_potential), emax])
    de = emax - emin

# Figure size
figsize = (12, 6)
sfig = float(figsize[0])/float(figsize[1])

# Figure
fig = plt.figure(figsize=figsize)
# Axes arrangement
left = 0.10
bottom = 0.15
column = [0.35, 0.10]
row = column[0]*sfig

color = ['b', 'r']
figlabels = ['A', 'B']

for idir, fdir in enumerate(sys_fdir):

    fdata = np.load(os.path.join(fdir, "save_results.npz"))
    
    fit_Vnbond = fdata['fit_Vnbond']
    fit_potential = fdata['fit_potential']
    fit_rmse  = fdata['fit_rmse']
    threshold = fdata['threshold']
    
    # Axes
    axs1 = fig.add_axes(
        [left + idir*np.sum(column), bottom, column[0], row])
    
    # Range
    axs1.set_xlim(emin - 0.1*de, emax + 0.1*de)
    axs1.set_ylim(emin - 0.1*de, emax + 0.1*de)
    
    # Plot
    axs1.plot(
        [emin - 0.1*de, emax + 0.1*de], [emin - 0.1*de, emax + 0.1*de], '-k')
    label = "RMSE = {:.2f} kcal/mol".format(fit_rmse)
    axs1.plot(
        fit_potential, fit_Vnbond, 'o{:s}'.format(color[idir]), 
        mfc='None', label=label)
    
    axs1.set_xlabel(r'$\Delta$E$_\mathrm{CCSD}$ (kcal/mol)', fontweight='bold')
    axs1.get_xaxis().set_label_coords(0.5, -0.12)
    axs1.set_ylabel(r'$\Delta$E$_\mathrm{FF}$ (kcal/mol)', fontweight='bold')
    axs1.get_yaxis().set_label_coords(-0.12, 0.5)
    
    axs1.set_xticks(np.arange(int(np.floor(emin)), int(np.ceil(emax)), 1))
    axs1.set_yticks(np.arange(int(np.floor(emin)), int(np.ceil(emax)), 1))
    
    if idir==0:
        title = r"N$_2$O-Xe interaction potential $\Delta$E"
    else:
        title = r"N$_2$O-SF$_6$ interaction potential $\Delta$E"
    axs1.set_title(title, fontweight='bold')
    
    axs1.legend(loc='lower right')
    
    tbox = TextArea(
        figlabels[idir], textprops=dict(color='k', fontsize=MEDIUM_SIZE))

    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.02, 0.95),
        bbox_transform=axs1.transAxes, borderpad=0.)
    
    axs1.add_artist(anchored_tbox)
    

plt.savefig(
    os.path.join(resdir, "fit_Ecorr.png"),
    format='png', dpi=dpi)
plt.close()
