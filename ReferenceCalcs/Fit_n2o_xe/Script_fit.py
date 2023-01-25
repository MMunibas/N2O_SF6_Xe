# Test Script to import PhysNet as energy function in CHARMM via PyCHARMM

# Basics
import os
import sys
import ctypes
import pandas
import numpy as np

# ASE basics (v 3.20.1 modified)
from ase import Atoms
from ase import io

# Optimization algorithms
from scipy.optimize import differential_evolution, minimize, curve_fit

# Miscellaneous
from ase.visualize import view
import ase.units as units
import time

# PyCHARMM
import pycharmm
import pycharmm.generate as gen
import pycharmm.ic as ic
import pycharmm.coor as coor
import pycharmm.energy as energy
import pycharmm.dynamics as dyn
import pycharmm.nbonds as nbonds
#import pycharmm.minimize as minimize
import pycharmm.crystal as crystal
import pycharmm.image as image
import pycharmm.psf as psf
import pycharmm.read as read
import pycharmm.write as write
import pycharmm.settings as settings
import pycharmm.lingo as stream
import pycharmm.select as select
import pycharmm.shake as shake
import pycharmm.cons_fix as cons_fix
import pycharmm.cons_harm as cons_harm
from pycharmm.lib import charmm as libcharmm

# Matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri

# Step 0: Parameter definition
#-----------------------------------------------------------

Ha2kcalmol = units.mol*units.Hartree/units.kcal

# Step 1: Load results
#-----------------------------------------------------------

positions = np.load("n2o_xe_positions.npy")
potential = np.load("n2o_xe_intactpot.npy")

# Add some slight angle to N2O
positions[:, 2, 0] = positions[:, 2, 0] + 0.00001

# Limit fit data to potential threshold
threshold = 5.0
fit_positions = positions[potential < threshold]
fit_potential = potential[potential < threshold]

# Step 2: Load CHARMM parameter files
#-----------------------------------------------------------

source_dir = 'source'

# Load topology files
n2o_top_file = os.path.join(source_dir, 'n2o.top')
read.rtf(n2o_top_file)
xe_top_file = os.path.join(source_dir, 'xe.top')
read.rtf(xe_top_file, append=True)

n2o_par_file = os.path.join(source_dir, 'original_n2o.par')
read.prm(n2o_par_file, flex=True)
xe_par_file = os.path.join(source_dir, 'xe.par')
read.prm(xe_par_file, flex=True, append=True)

# Step 3: Generate system
#-----------------------------------------------------------

# Generate N2O - Xe pair
read.sequence_string('N2O XE')
gen.new_segment(
    seg_name='SYS',
    setup_ic=True)

# Set initial positions
pandas_pos = pandas.DataFrame({
    'x': positions[0, :, 0], 'y': positions[0, :, 1], 'z': positions[0, :, 2]})
coor.set_positions(pandas_pos)
coor.show()

# Step 4: Finalize CHARMM Properties
#-----------------------------------------------------------

mdcmlns= (
    "OPEN WRITE UNIT 31 CARD NAME dcm_charges.xyz\n"
    + "OPEN UNIT 40 CARD READ NAME source/n2o_xe.dcm\n"
    + "DCM IUDCM 40 XYZ 31 TSHIFT\n"
    + "CLOSE UNIT 40\n")

energy.show()
stream.charmm_script(mdcmlns)
energy.show()

# Non-bonding parameter
nbonds = {
    'atom': True,
    'fshift': True,
    'cdie': True,
    'eps': 1.0,
    'wmin': 1.5,
    'vdw': True,
    'cutnb': 14,
    'ctofnb': 12,
    'ctonnb': 10,
    }

nbond = pycharmm.NonBondedScript(**nbonds)
nbond.run()

# Write pdb and psf files
write.coor_pdb("n2o_xe.pdb", title="N2O - Xe pair")
write.psf_card("n2o_xe.psf", title="N2O - Xe pair")

# Test energy
energy.show()

# Step 5: Prepare van-der-Waals fit
#-----------------------------------------------------------

# Compute interatomic energy of certain conformation
def get_nbond_energy(positions):
    
    # Iterate over positions
    Vnbond = np.zeros(positions.shape[0])
    for ip, pos in enumerate(positions):
        
        # Set position
        pandas_pos = pandas.DataFrame({
            'x': pos[:, 0], 'y': pos[:, 1], 'z': pos[:, 2]})
        coor.set_positions(pandas_pos)
        
        # Get potential contributions
        Vcntrb = energy.get_energy()
        
        # Sum up interatomic potential
        if np.any(Vcntrb.name=="vdwaals"):
            Vnbond[ip] += (
                Vcntrb.loc[Vcntrb.name=="vdwaals", "value"].tolist()[0])
        if np.any(Vcntrb.name=="elec"):
            Vnbond[ip] += (
                Vcntrb.loc[Vcntrb.name=="elec", "value"].tolist()[0])
        if np.any(Vcntrb.name=="hbonds"):
            Vnbond[ip] += (
                Vcntrb.loc[Vcntrb.name=="hbonds", "value"].tolist()[0])
        
    return Vnbond

# Get interatomic energy for certain vdW parameter
def clc_vdw(positions, *pars):
    
    # Read parameter file template
    with open(os.path.join(source_dir, 'template_n2o.par'), 'r') as f:
        ltmp = f.read()
        
    # Write parameter
    ltmp = ltmp.replace("%n1epsl%", "{:.5f}".format(pars[0]))
    ltmp = ltmp.replace("%n2epsl%", "{:.5f}".format(pars[1]))
    ltmp = ltmp.replace("%o1epsl%", "{:.5f}".format(pars[2]))
    ltmp = ltmp.replace("%n1rm%", "{:.4f}".format(pars[3]))
    ltmp = ltmp.replace("%n2rm%", "{:.4f}".format(pars[4]))
    ltmp = ltmp.replace("%o1rm%", "{:.4f}".format(pars[5]))
    
    
    # Save parameter file
    with open(os.path.join(source_dir, 'step_n2o.par'), 'w') as f:
        f.write(ltmp)
    
    # Reload parameter files
    n2o_par_file = os.path.join(source_dir, 'step_n2o.par')
    read.prm(n2o_par_file, flex=True)
    xe_par_file = os.path.join(source_dir, 'xe.par')
    read.prm(xe_par_file, flex=True, append=True)

    # Get interatomic energy
    Vnbond = get_nbond_energy(positions)
    
    # Bonding potential
    Vbond = pars[6]
    
    return Vnbond + Vbond

def fit_vdw(pars):
    
    # Read parameter file template
    with open(os.path.join(source_dir, 'template_n2o.par'), 'r') as f:
        ltmp = f.read()
        
    # Write parameter
    ltmp = ltmp.replace("%n1epsl%", "{:.5f}".format(pars[0]))
    ltmp = ltmp.replace("%n2epsl%", "{:.5f}".format(pars[1]))
    ltmp = ltmp.replace("%o1epsl%", "{:.5f}".format(pars[2]))
    ltmp = ltmp.replace("%n1rm%", "{:.4f}".format(pars[3]))
    ltmp = ltmp.replace("%n2rm%", "{:.4f}".format(pars[4]))
    ltmp = ltmp.replace("%o1rm%", "{:.4f}".format(pars[5]))
    
    # Save parameter file
    with open(os.path.join(source_dir, 'step_n2o.par'), 'w') as f:
        f.write(ltmp)
    #time.sleep(0.1)
    # Reload parameter files
    n2o_par_file = os.path.join(source_dir, 'step_n2o.par')
    read.prm(n2o_par_file, flex=True)
    xe_par_file = os.path.join(source_dir, 'xe.par')
    read.prm(xe_par_file, flex=True, append=True)

    # Get interatomic energy
    Vnbond = get_nbond_energy(fit_positions)
    
    # Bonding potential
    Vbond = pars[6]
    
    # Weighting
    sigma = 1./((fit_potential - np.min(fit_potential)) + 0.0001)
    
    print(np.mean((fit_potential - Vnbond)**2))
    print(pars)
    return np.sqrt(np.mean(sigma*(fit_potential - Vnbond)**2))

def cfit_vdw(x, *cpars):
    
    #indc = np.array(x, dtype=int)
    #cfit_positions = fit_positions[indc]
    
    # Read parameter file template
    with open(os.path.join(source_dir, 'template_n2o.par'), 'r') as f:
        ltmp = f.read()
        
    # Write parameter
    ltmp = ltmp.replace("%n1epsl%", "{:.5f}".format(cpars[0]))
    ltmp = ltmp.replace("%n2epsl%", "{:.5f}".format(cpars[1]))
    ltmp = ltmp.replace("%o1epsl%", "{:.5f}".format(cpars[2]))
    ltmp = ltmp.replace("%n1rm%", "{:.4f}".format(cpars[3]))
    ltmp = ltmp.replace("%n2rm%", "{:.4f}".format(cpars[4]))
    ltmp = ltmp.replace("%o1rm%", "{:.4f}".format(cpars[5]))
    
    # Save parameter file
    os.remove(os.path.join(source_dir, 'step_n2o.par'))
    with open(os.path.join(source_dir, 'step_n2o.par'), 'w') as f:
        f.write(ltmp)
    
    # Reload parameter files
    n2o_par_file = os.path.join(source_dir, 'step_n2o.par')
    read.prm(n2o_par_file, flex=True)
    xe_par_file = os.path.join(source_dir, 'xe.par')
    read.prm(xe_par_file, flex=True, append=True)
    
    # Get interatomic energy
    #Vnbond = get_nbond_energy(cfit_positions)
    Vnbond = get_nbond_energy(x)
    
    # Bonding potential
    Vbond = cpars[6]
    print("Current pars:", cpars)
    print("Current RMSE:", np.sqrt(np.mean((Vnbond + Vbond - fit_potential)**2)))
    return Vnbond #+ Vbond


# Define initial vdW and Vbond guess: N1, N2, O
epsilon = np.array([-0.15521, -0.06885, -0.12094])
#epsilon = np.array([-2.63e-03, -1.12e-03, -1.01e-03])
#epsilon = np.array([-0.36546, -0.00046, -0.34473])
rminhlf = np.array([1.7488, 1.6427, 1.7084])
#rminhlf = np.array([2.25, 2.49, 2.25])
#rminhlf = np.array([1.6306, 2.4395, 1.4953])
vbond = 0.0

# Pack to parameter array
pars = np.stack([epsilon, rminhlf]).reshape(-1)
pars = np.append(pars, vbond)

# Step 6: Start van-der-Waals fit
#-----------------------------------------------------------

if False:
        
    ## Start optimization
    #bounds = np.array([
        #(-0.5, -0.001),
        #(-0.5, -0.001),
        #(-0.5, -0.001),
        #(1.5, 3.0),
        #(1.5, 3.0),
        #(1.5, 3.0),
        #(-0.001, 0.001)])

    ##res = differential_evolution(fit_vdw, bounds)
    ##res = res.x

    ##res = minimize(fit_vdw, pars, bounds=bounds)
    ##res = res.x

    bounds = np.array([
        (-1.0, -0.0001),
        (-1.0, -0.0001),
        (-1.0, -0.0001),
        (1.0, 3.0),
        (1.0, 3.0),
        (1.0, 3.0),
        (-0.1, 0.1)])

    # Weighting
    sigma = ((fit_potential - np.min(fit_potential)) + 1)**2

    # Curve fit
    res, _ = curve_fit(
        cfit_vdw, 
        fit_positions, #np.arange(len(fit_positions)),
        fit_potential,
        p0=pars, 
        bounds=bounds.T,
        sigma=sigma,
        diff_step=1e-4)

    ## Assign result

    pars = res

    epsilon = np.array([pars[0], pars[1], pars[2]])
    rminhlf = np.array([pars[3], pars[4], pars[5]])
    vbond = pars[-1]

else:
    
    with open(os.path.join(source_dir, 'step_n2o.par'), 'r') as f:
        parlines = f.readlines()
    
    epsilon = np.zeros(3, dtype=float)
    rminhlf = np.zeros(3, dtype=float)
    for parl in parlines:
        if "N=       0.0" in parl:
            epsilon[0], rminhlf[0] = np.array(
                parl.split()[2:4], dtype=float)
        elif "=N=      0.0" in parl:
            epsilon[1], rminhlf[1] = np.array(
                parl.split()[2:4], dtype=float)
        elif "O=       0.0" in parl:
            epsilon[2], rminhlf[2] = np.array(
                parl.split()[2:4], dtype=float)
    
    vbond = 0.0
    pars = np.stack([epsilon, rminhlf]).reshape(-1)
    pars = np.append(pars, vbond)
    print(pars)
    
# Get interaction potential
fit_Vnbond = clc_vdw(fit_positions, *pars)
Vnbond = clc_vdw(positions, *pars)

#print(res)

fit_Vdiff = fit_Vnbond - fit_potential
fit_rmse = np.sqrt(np.mean(fit_Vdiff**2))
print(fit_rmse)

Vdiff = Vnbond - potential
rmse = np.sqrt(np.mean(Vdiff**2))
print(rmse)
print(pars)
Vdiff[potential > threshold] = np.nan

# Step 7: Plot nonbonding potential
#-----------------------------------------------------------

# Mirror results at yz plane
mpositions = positions.copy()
mpotential = potential.copy()
mVnbond = Vnbond.copy()
mVdiff = Vdiff.copy()
for ip, posi in enumerate(positions):
    
    if np.any(posi[:, 0] != 0.0):
        mpos = posi.reshape(1, -1, 3)
        mpos[0, :, 0] = -mpos[0, :, 0]
        mpositions = np.append(mpositions, mpos, axis=0)
        mpotential = np.append(
            mpotential, [mpotential[ip]], axis=0)
        mVnbond = np.append(
            mVnbond, [mVnbond[ip]], axis=0)
        mVdiff = np.append(
            mVdiff, [mVdiff[ip]], axis=0)
        
# Irregular grid values
x = mpositions[:, -1, 0]
y = mpositions[:, -1, 2]

# Create grid values
r1range = [-8., 8.]
r2range = [-10., 10.]
srange = (r1range[1] - r1range[0])/(r2range[1] - r2range[0])
xi = np.linspace(r1range[0], r1range[1], 101)
yi = np.linspace(r2range[0], r2range[1], 101)
Xi, Yi = np.meshgrid(xi, yi)

# Interpolation on a grid - Reference
triang = tri.Triangulation(x, y)
interpolator = tri.LinearTriInterpolator(triang, mpotential)
potref_grid = interpolator(Xi, Yi)

# Interpolation on a grid - Fit
triang = tri.Triangulation(x, y)
interpolator = tri.LinearTriInterpolator(triang, mVnbond)
potfit_grid = interpolator(Xi, Yi)

# Interpolation on a grid - Diff
triang = tri.Triangulation(x, y)
interpolator = tri.LinearTriInterpolator(triang, mVdiff)
potdff_grid = interpolator(Xi, Yi)


# Plot properties

# Fontsize
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16

#plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


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

# Graphical output format type
gout_type = 'png'
dpi = 200

# Figure size
figsize = (12, 6)
sfig = float(figsize[0])/float(figsize[1])

# Axes arrangement
left = 0.08
bottom = 0.15
column = [0.21, 0.02]
row = [column[0]*sfig/srange]
cbarcolumn = 0.02
cbarspace = 0.1


# Figure
fig = plt.figure(figsize=figsize)

# Axes
axs1 = fig.add_axes(
        [left + 0.*np.sum(column), bottom, column[0], row[0]])
    
axs2 = fig.add_axes(
        [left + 1.*np.sum(column), bottom, column[0], row[0]])

axcb = fig.add_axes([
    left + 2.*np.sum(column), bottom, 
    cbarcolumn, row[0]])

axs3 = fig.add_axes([
    left + 2.*np.sum(column) + cbarspace + cbarcolumn, 
    bottom, column[0], row[0]])

axcb3 = fig.add_axes([
    left + 3.*np.sum(column) + cbarspace + cbarcolumn, bottom, 
    cbarcolumn, row[0]])

# Get potential range
Vmin = np.nanmin([potref_grid, potfit_grid])
Vdiffm = np.nanmax(np.abs(potdff_grid))
Vrange = [Vmin, 3*np.abs(Vmin)]
levels = 100


# Cut largest potential values
potref_grid[potref_grid > Vrange[1]] = Vrange[1]
potfit_grid[potfit_grid > Vrange[1]] = Vrange[1]

# Get potential minima positions
rndpotref_grid = np.around(potref_grid, decimals=3)
posminref = np.where(rndpotref_grid==np.nanmin(rndpotref_grid))
rndpotfit_grid = np.around(potfit_grid, decimals=3)
posminfit = np.where(rndpotfit_grid==np.nanmin(rndpotfit_grid))

# Plot 
if np.nanmin(potref_grid) < np.nanmin(potfit_grid):
    
    cs = axs1.contourf(
        Xi, Yi, potref_grid, levels, 
        vmin=-Vrange[1], vmax=Vrange[1], cmap='seismic')

    axs2.contourf(
        Xi, Yi, potfit_grid, levels, 
        vmin=-Vrange[1], vmax=Vrange[1], cmap='seismic')

else:
    
    axs1.contourf(
        Xi, Yi, potref_grid, levels, 
        vmin=-Vrange[1], vmax=Vrange[1], cmap='seismic')

    cs = axs2.contourf(
        Xi, Yi, potfit_grid, levels, 
        vmin=-Vrange[1], vmax=Vrange[1], cmap='seismic')

cd = axs3.contourf(
        Xi, Yi, potdff_grid, levels, 
        vmin=-Vdiffm, vmax=Vdiffm, cmap='seismic')


# Minima
for (ix, iz) in np.stack(posminref).T:
    axs1.plot(Xi[ix, iz], Yi[ix, iz], 'ok', mfc='None')
for (ix, iz) in np.stack(posminfit).T:
    axs2.plot(Xi[ix, iz], Yi[ix, iz], 'ok', mfc='None')

# Colorbar
cb_ticks = np.arange(np.floor(Vrange[0]), np.ceil(Vrange[1]), 0.5)

cbar = fig.colorbar(
    cs, cax=axcb, ticks=cb_ticks)

#cb_ticks = np.arange(np.floor(Vrange[0]), np.ceil(Vrange[1]), 0.5)

cbar = fig.colorbar(
    cd, cax=axcb3)

# N2O atom positions
axs1.text(
    mpositions[0, 0, 0], mpositions[0, 0, 2], 'N', ha='center', va='center')
axs1.text(
    mpositions[0, 1, 0], mpositions[0, 1, 2], 'N', ha='center', va='center')
axs1.text(
    mpositions[0, 2, 0], mpositions[0, 2, 2], 'O', ha='center', va='center')

axs2.text(
    mpositions[0, 0, 0], mpositions[0, 0, 2], 'N', ha='center', va='center')
axs2.text(
    mpositions[0, 1, 0], mpositions[0, 1, 2], 'N', ha='center', va='center')
axs2.text(
    mpositions[0, 2, 0], mpositions[0, 2, 2], 'O', ha='center', va='center')

axs3.text(
    mpositions[0, 0, 0], mpositions[0, 0, 2], 'N', ha='center', va='center')
axs3.text(
    mpositions[0, 1, 0], mpositions[0, 1, 2], 'N', ha='center', va='center')
axs3.text(
    mpositions[0, 2, 0], mpositions[0, 2, 2], 'O', ha='center', va='center')

# Titles
fig.suptitle(
    r"N$_2$O-Xe Interaction potential $(E_\mathrm{int})$"
    + "\n" + r"RMSE( < {:.0f} kcal/mol) = {:.2f} kcal/mol ".format(
        threshold, fit_rmse),
    fontweight='bold')
axs1.set_title(
    r"CCSD(T)" + "\n" + r"Min$(E_\mathrm{int})$ = " 
    + "{:0.2f} kcal/mol ".format(np.nanmin(potref_grid))
    + r"($\circ$)" + "\n",
    fontweight='bold')
axs2.set_title(
    r"CHARMM" + "\n" + r"Min$(E_\mathrm{int})$ = " 
    + "{:0.2f} kcal/mol ".format(np.nanmin(potfit_grid))
    + r"($\circ$)" + "\n",
    fontweight='bold')
axs3.set_title(
    r"Potential Deviation" + "\n" 
    + r"$E_\mathrm{int}$(CHARMM) - $E_\mathrm{int}$(CCSD(T))",
    fontweight='bold')



# Labels
axs1.set_xlabel(r'Xe position $x$ ($\mathrm{\AA}$)', fontweight='bold')
axs1.get_xaxis().set_label_coords(0.5, -0.12)
axs2.set_xlabel(r'Xe position $x$ ($\mathrm{\AA}$)', fontweight='bold')
axs2.get_xaxis().set_label_coords(0.5, -0.12)
axs3.set_xlabel(r'Xe position $x$ ($\mathrm{\AA}$)', fontweight='bold')
axs3.get_xaxis().set_label_coords(0.5, -0.12)
axs1.set_ylabel(r'Xe position $z$ ($\mathrm{\AA}$)', fontweight='bold')
axs1.get_yaxis().set_label_coords(-0.16, 0.50)

axcb.set_ylabel('Potential (kcal/mol)', fontweight='bold')
axcb3.set_ylabel('Deviation (kcal/mol)', fontweight='bold')

axs1.set_xticks([-6, -3.0, 0.0, 3.0, 6.0])
axs2.set_xticks([-6, -3.0, 0.0, 3.0, 6.0])
axs3.set_xticks([-6, -3.0, 0.0, 3.0, 6.0])
axs1.set_yticks([-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0])
axs2.set_yticks([-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0])
axs2.set_yticklabels([])
axs3.set_yticks([-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0])
axs3.set_yticklabels([])

plt.savefig(
    "fit_interpot.png",
    format='png', dpi=dpi)
#plt.show()

#[-2.63453694e-03  2.25377827e+00 -1.11763312e-03  2.48710397e+00
 #-1.00646381e-03  2.25427950e+00  1.00000000e-02]


# Step 7.1: Plot energy correlation
#-----------------------------------------------------------

np.savez(
    "save_results.npz",
    fit_Vnbond=fit_Vnbond,
    fit_potential=fit_potential,
    fit_rmse=fit_rmse,
    threshold=threshold)

# Figure size
figsize = (6, 6)
sfig = float(figsize[0])/float(figsize[1])

# Axes arrangement
left = 0.15
bottom = 0.15
column = [0.7, 0.00]
row = [column[0]*sfig]

# Figure
fig = plt.figure(figsize=figsize)

# Axes
axs1 = fig.add_axes(
        [left + 0.*np.sum(column), bottom, column[0], row[0]])

# Range
emin = np.min([np.min(fit_Vnbond), np.min(fit_potential)])
emax = np.max([np.max(fit_Vnbond), np.max(fit_potential)])
de = emax - emin
axs1.set_xlim(emin - 0.1*de, emax + 0.1*de)
axs1.set_ylim(emin - 0.1*de, emax + 0.1*de)

# Plot
axs1.plot(
    [emin - 0.1*de, emax + 0.1*de], [emin - 0.1*de, emax + 0.1*de], '-k')
label = "RMSE = {:.2f} kcal/mol".format(fit_rmse)
axs1.plot(fit_potential, fit_Vnbond, 'ob', mfc='None', label=label)

# Labels
axs1.set_xlabel(r'$\Delta$E$_\mathrm{CCSD}$ (kcal/mol)', fontweight='bold')
axs1.get_xaxis().set_label_coords(0.5, -0.12)
axs1.set_ylabel(r'$\Delta$E$_\mathrm{FF}$ (kcal/mol)', fontweight='bold')
axs1.get_yaxis().set_label_coords(-0.12, 0.5)

axs1.set_xticks(np.arange(int(np.floor(emin)), int(np.ceil(emax)), 1))
axs1.set_yticks(np.arange(int(np.floor(emin)), int(np.ceil(emax)), 1))

axs1.set_title(
    r"N$_2$O-Xe interaction potential $\Delta$E"
    + "\n" + r"($\Delta$E < {:.0f} kcal/mol)".format(
        threshold),
    fontweight='bold')

axs1.legend(loc='upper left')

plt.savefig(
    "fit_Ecorr.png",
    format='png', dpi=dpi)
#plt.show()
plt.close()
print(len(fit_potential))
