# Basics
import os
import sys
import numpy as np
from glob import glob
import scipy

# ASE - Basics
from ase import Atoms
import ase.units as units

# MDAnalysis
import MDAnalysis

# Multiprocessing
from multiprocessing import Pool

# Statistics
from statsmodels.tsa.stattools import acovf

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Miscellaneous
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar

#------------------------
# Matplotlib Setup
#------------------------

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
plt.rc('xtick', labelsize=XSMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=XSMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
#plt.rc('legend.title_fontsize', titlesize=MEDIUM_SIZE)  # legend title fontsize
dpi = 200

color_scheme = [
    'b', 'r', 'g', 'magenta', 'orange', 'purple', 'brown', 'darkblue',
    'darkred', 'darkgreen', 'darkgrey', 'olive']

# Figure arrangement
figsize = (12, 5)
left = 0.18
bottom = 0.15
row = np.array([0.80, 0.00])
column = np.array([0.15, 0.02])

# Frequency ranges
rngen = 200
rnge1 = [580, 650]
rnge2 = [1180, 1350]
rnge3 = [2120, 2310]

# Plot absorption spectra
fig = plt.figure(figsize=figsize)

# Initialize axes
cshift = .0
axs1 = fig.add_axes([
    left + cshift, bottom, 
    column[0]*((rnge1[1] - rnge1[0])/rngen), row[0]])
cshift += column[0]*((rnge1[1] - rnge1[0])/rngen) + column[1]
axs2 = fig.add_axes([
    left + cshift, bottom, 
    column[0]*((rnge2[1] - rnge2[0])/rngen), row[0]])
cshift += column[0]*((rnge2[1] - rnge2[0])/rngen) + column[1]
axs3 = fig.add_axes([
    left + cshift, bottom, 
    column[0]*((rnge3[1] - rnge3[0])/rngen), row[0]])





# Result directory
resdir = "paper_figures"
if not os.path.exists(resdir):
    os.mkdir(resdir)

#------------------------
# Setup Parameters 1
#------------------------

# Damping rate in ps
tau_damping_list = [0.5, 0.2, 0.1, 0.05]
tau_damping = 0.1

# Number of parallel tasks
tasks = 20

# Source directory
sys_srce = 'N2O'

# Case information
sys_cdir = [
    '1_rot_0',
    '2_rot_y',
    '3_rot_x',
    '4_boltzman']
sys_trgt = 'N2O_model'

# Temperatures [K]
T = 321.93

# Time step size in fs
dt = 1.0

# Evaluation result files
axsfile = "axis.npy"
acvfile = "acv_axis.npy"
dipfile = "dipole.npy"
dcvfile = "acv_dipole_damping.npy"
spcfile = "ir_spectra_damping.npy"
frqfile = "frequencies_damping.npy"

# Working directories
Jdirs = "J{:s}"
smpldirs = "J{:s}_S{:s}"

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

# IR Weighting constant
beta = 1.0/3.1668114e-6/float(T)
hbar = 1.0
cminvtoau = 1.0/2.1947e5
const = beta*cminvtoau*hbar

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
n2o_J = np.arange(1000, dtype=float)
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

P_J = boltzmann(n2o_J, T)

Jmax = int(np.argmax(P_J))

#-----------------------------
# Plot damping IR
#-----------------------------

avgfreq = 1.0
def moving_average(data_set, periods=9):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, 'same')


for ic, tdamp in enumerate(tau_damping_list):
    
    cdir = sys_cdir[2]
    
    # J state samples
    J10 = Jmax
    
    # Sample dirs - three samples
    sdir10 = os.path.join(
        sys_srce, cdir, Jdirs.format(str(J10)), smpldirs.format(str(J10), "0"))
    
    # Load results
    urot = np.load(os.path.join(sdir10, axsfile))
    acv_urot = np.load(os.path.join(sdir10, acvfile))
    dples = np.load(os.path.join(sdir10, dipfile))
    
    # Frequency array
    Nframes = dples.shape[0]
    Nfreq = int(Nframes/2) + 1
    freq = np.arange(Nfreq)/float(Nframes)/dt*jiffy
    
    # Align dipole with rotational axis and get 
    du = np.zeros([Nframes, 3], dtype=float)
    acv_du = np.zeros([Nframes, 3], dtype=float)
    for iax in range(3):
        
        du[:, iax] = np.sum(urot[:, iax]*dples, axis=1)
        acv_du[:, iax] = acovf(du[:, iax], fft=True)
    
    # Compute rotational lifetime array
    rot_damp = np.exp(-np.arange(0.0, Nframes*dt, dt)/(tdamp*1.e3))

    # Compute IR spectra
    acv = (
        acv_du[:, 0]*acv_urot[:, 0]*rot_damp
        + acv_du[:, 1]*acv_urot[:, 1]*rot_damp
        + acv_du[:, 2]*acv_urot[:, 2]*rot_damp
        )
    blackman_acv = acv*np.blackman(Nframes)
    spec = np.abs(np.fft.rfftn(blackman_acv))*np.tanh(const*freq/2.)

    # Apply moving average
    Nave = int(avgfreq/(freq[1] - freq[0]))
    avgspec = moving_average(spec, Nave)

    # Scale avgspec
    select = np.logical_and(
        freq > rnge1[0], freq < rnge3[1])
    avgspec /= np.max(avgspec[select])
    
    # Select ranges
    select1 = np.logical_and(
        freq > rnge1[0], freq < rnge1[1])
    select2 = np.logical_and(
        freq > rnge2[0], freq < rnge2[1])
    select3 = np.logical_and(
        freq > rnge3[0], freq < rnge3[1])
    
    # Plot
    label = (
        r'$j(\vec{e}_y)=$' + '{:d}'.format(J10) + '\n'
        + r'$\tau = {:.2f}$ ps'.format(tdamp))
    
    # Bending mode
    overlap = 1.0
    shift = len(sys_cdir) - 1 - ic
    shift = ic
    scale1 = 1./np.max(avgspec[select1])
    axs1.plot(
        freq[select1], overlap*avgspec[select1]*scale1 + shift, '-', 
        color=color_scheme[ic], label=label)

    # Symmetric stretch
    scale2 = 1./np.max(avgspec[select2])
    axs2.plot(
        freq[select2], overlap*avgspec[select2]*scale2 + shift, '-', 
        color=color_scheme[ic], label=label)

    # Asymmetric stretch
    scale3 = 1./np.max(avgspec[select3])
    axs3.plot(
        freq[select3], 
        overlap*avgspec[select3]*scale3 + shift, '-', 
        color=color_scheme[ic], label=label)
    
    # Add scaling factors
    scale_dict = {
        'family' : 'monospace',
        'weight' : 'light',
        'size'   : 'x-small'}
    if ic in [2]:
        axs1.text(
            rnge1[0] + (rnge1[1] - rnge1[0])*0.01, 
            shift + 0.2, 'x{:.0f}'.format(scale1),
            fontdict=scale_dict)
        axs2.text(
            rnge2[0] + (rnge2[1] - rnge2[0])*0.01, 
            shift + 0.2, 'x{:.0f}'.format(scale2),
            fontdict=scale_dict)
        axs3.text(
            rnge3[0] + (rnge3[1] - rnge3[0])*0.01, 
            shift + 0.2, 'x{:.0f}'.format(scale3),
            fontdict=scale_dict)
    elif ic in [1]:
        axs1.text(
            rnge1[0] + (rnge1[1] - rnge1[0])*0.01, 
            shift + 0.15, 'x{:.0f}'.format(scale1),
            fontdict=scale_dict)
        axs2.text(
            rnge2[0] + (rnge2[1] - rnge2[0])*0.01, 
            shift + 0.15, 'x{:.0f}'.format(scale2),
            fontdict=scale_dict)
        axs3.text(
            rnge3[0] + (rnge3[1] - rnge3[0])*0.01, 
            shift + 0.15, 'x{:.0f}'.format(scale3),
            fontdict=scale_dict)
    else:    
        axs1.text(
            rnge1[0] + (rnge1[1] - rnge1[0])*0.01, 
            shift + 0.1, 'x{:.0f}'.format(scale1),
            fontdict=scale_dict)
        axs2.text(
            rnge2[0] + (rnge2[1] - rnge2[0])*0.01, 
            shift + 0.1, 'x{:.0f}'.format(scale2),
            fontdict=scale_dict)
        axs3.text(
            rnge3[0] + (rnge3[1] - rnge3[0])*0.01, 
            shift + 0.1, 'x{:.0f}'.format(scale3),
            fontdict=scale_dict)
        

# Axis range
axs1.set_xlim(rnge1)
axs2.set_xlim(rnge2)
axs3.set_xlim(rnge3)
axs1.set_ylim([0, len(sys_cdir) - 1 + overlap])
axs2.set_ylim([0, len(sys_cdir) - 1 + overlap])
axs3.set_ylim([0, len(sys_cdir) - 1 + overlap])

# Axis labels
#axs1.set_xticks([])
axs2.set_xticks([1200, 1250, 1300, 1350])
axs3.set_xticks([2150, 2200, 2250, 2300])
axs1.set_yticks([ii for ii in range(len(sys_cdir))])
axs2.set_yticks([ii for ii in range(len(sys_cdir))])
axs3.set_yticks([ii for ii in range(len(sys_cdir))])
axs1.set_yticklabels([])
axs2.set_yticklabels([])
axs3.set_yticklabels([])

axs2.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)', fontweight='bold')
axs2.get_xaxis().set_label_coords(1.1, -0.1)

handles, labels = plt.gca().get_legend_handles_labels()
order = list(range(len(sys_cdir)))
axs1.legend(
    [handles[idx] for idx in order], [labels[idx] for idx in order],
    loc=(-3.30, 0.05),  framealpha=1.0,
    title=r"N$_2$O model")

tbox = TextArea(
    "A", textprops=dict(color='k', fontsize=BIGGER_SIZE))

anchored_tbox = AnchoredOffsetbox(
    loc='upper right', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(-0.15, 0.95),
    bbox_transform=axs1.transAxes, borderpad=0.)

axs1.add_artist(anchored_tbox)
    



#------------------------
# Setup Parameters 2
#------------------------


moving_freq = 5.0
def moving_average(data_set, periods=9):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, 'same')


# Maximum range of J states
maxJ = None

# Langevin damping factors
langevin_damp = [
    10.0, 1.0, 0.1, 
    0.01, 0.0001]   #, 0.000001

sys_srce = "N2O_model/5_langevin"

# Rotational temperature
rot_temp = 321.9

# Vibrational temperature                                                                                           
vib_temp = 321.9  

# Number of samples per J state
Nsmpl = 1

# Time step size in fs
dt = 1.0

# Working directories
Jdirs = "J{:d}"
smpldirs = "J{:d}_D{:.1E}_S{:d}"

# Simulation result files
dcdfile = "dyna_crd.0.dcd"
psffile = "n2o.psf"

# Evaluation result files
axsfile = "axis.npy"
acvfile = "acv_axis.npy"
dipfile = "dipole.npy"
dcvfile = "acv_dipole.npy"
spcfile = "ir_spectra.npy"
frqfile = "frequencies.npy"

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
   
# IR Weighting constant
beta = 1.0/3.1668114e-6/float(rot_temp)
hbar = 1.0
cminvtoau = 1.0/2.1947e5
const = beta*cminvtoau*hbar



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
# Plot Thermostat Model
#------------------------

# Initialize axes
cshift += column[0]*1.4
axs4 = fig.add_axes([
    left + cshift, bottom, 
    column[0]*((rnge1[1] - rnge1[0])/rngen), row[0]])
cshift += column[0]*((rnge1[1] - rnge1[0])/rngen) + column[1]
axs5 = fig.add_axes([
    left + cshift, bottom, 
    column[0]*((rnge2[1] - rnge2[0])/rngen), row[0]])
cshift += column[0]*((rnge2[1] - rnge2[0])/rngen) + column[1]
axs6 = fig.add_axes([
    left + cshift, bottom, 
    column[0]*((rnge3[1] - rnge3[0])/rngen), row[0]])


# Panel labels

if maxJ is None:
    
    if rot_temp == 0.0:
        Ji = 0
    else:
        Ji = np.argmax(P_J)
    
    for ic, ldamp in enumerate(langevin_damp[::-1]):
            
        # Intialize average spectra
        workdir = os.path.join(
            sys_srce, Jdirs.format(Ji), smpldirs.format(Ji, ldamp, 0))
        spec = np.load(os.path.join(workdir, spcfile))
        freq = np.load(os.path.join(workdir, frqfile))
        avgfreq = freq
        avgspec = np.zeros_like(spec)
        
        # Iterate over samples
        for ismpl in range(Nsmpl):
            
            # Working directory
            workdir = os.path.join(
                sys_srce, Jdirs.format(Ji), 
                smpldirs.format(Ji, ldamp, ismpl))
            
            # Load spectra
            spec = np.load(os.path.join(workdir, spcfile))
            
            # Add spectra
            avgspec += spec/float(Nsmpl)
        
        # Apply moving average
        if moving_freq is None:
            Nave = 0.0
            avgspec = spec
        else:
            Nave = int(moving_freq/(freq[1] - freq[0]))
            avgspec = moving_average(spec, Nave)
            
        # Scale avgspec
        select = np.logical_and(
            freq > 300, freq < 2500)
        avgspec /= np.max(avgspec[select])
        
        # Select ranges
        select1 = np.logical_and(
            freq > rnge1[0], freq < rnge1[1])
        select2 = np.logical_and(
            freq > rnge2[0], freq < rnge2[1])
        select3 = np.logical_and(
            freq > rnge3[0], freq < rnge3[1])
        
        # Plot
        label = (
            r'$j(\vec{e}_y)=$' + '{:d}'.format(Ji) + '\n'
            + r'$\tau = {:.3f}$ ps'.format(ldamp))
        label = r'$\gamma_i$ = {:.0E} ps'.format(ldamp)
            
        # Bending mode
        overlap = 1.5
        shift = ic
        scale1 = 1./np.max(avgspec[select1])
        axs4.plot(
            freq[select1][::20], overlap*avgspec[select1][::20]*scale1 + shift, 
            '--', color=color_scheme[ic], label=label)

        # Symmetric stretch
        scale2 = 1./np.max(avgspec[select2])
        axs5.plot(
            freq[select2][::20], overlap*avgspec[select2][::20]*scale2 + shift, 
            '--', color=color_scheme[ic], label=label)

        # Asymmetric stretch
        scale3 = 1./np.max(avgspec[select3])
        axs6.plot(
            freq[select3][::20], 
            overlap*avgspec[select3][::20]*scale3 + shift, '--', 
            color=color_scheme[ic], label=label)
        
        # Add scaling factors
        scale_dict = {
            'family' : 'monospace',
            'weight' : 'light',
            'size'   : 'x-small'}
        if ic in [1,2,3]:
            axs4.text(
                rnge1[0] + (rnge1[1] - rnge1[0])*0.01, 
                shift + 0.2, 'x{:.0f}'.format(scale1),
                fontdict=scale_dict)
            axs5.text(
                rnge2[0] + (rnge2[1] - rnge2[0])*0.01, 
                shift + 0.2, 'x{:.0f}'.format(scale2),
                fontdict=scale_dict)
            axs6.text(
                rnge3[0] + (rnge3[1] - rnge3[0])*0.01, 
                shift + 0.2, 'x{:.0f}'.format(scale3),
                fontdict=scale_dict)
        else:    
            axs4.text(
                rnge1[0] + (rnge1[1] - rnge1[0])*0.01, 
                shift + 0.1, 'x{:.0f}'.format(scale1),
                fontdict=scale_dict)
            axs5.text(
                rnge2[0] + (rnge2[1] - rnge2[0])*0.01, 
                shift + 0.1, 'x{:.0f}'.format(scale2),
                fontdict=scale_dict)
            axs6.text(
                rnge3[0] + (rnge3[1] - rnge3[0])*0.01, 
                shift + 0.1, 'x{:.0f}'.format(scale3),
                fontdict=scale_dict)
            
        
    
    # Axis range
    axs4.set_xlim(rnge1)
    axs5.set_xlim(rnge2)
    axs6.set_xlim(rnge3)
    axs4.set_ylim([0, len(langevin_damp) - 1 + overlap])
    axs5.set_ylim([0, len(langevin_damp) - 1 + overlap])
    axs6.set_ylim([0, len(langevin_damp) - 1 + overlap])

    # Axis labels
    #axs4.set_xticks([])
    axs5.set_xticks([1200, 1250, 1300, 1350])
    axs6.set_xticks([2150, 2200, 2250, 2300])
    axs4.set_yticks([ii for ii in range(len(langevin_damp))])
    axs5.set_yticks([ii for ii in range(len(langevin_damp))])
    axs6.set_yticks([ii for ii in range(len(langevin_damp))])
    axs4.set_yticklabels([])
    axs5.set_yticklabels([])
    axs6.set_yticklabels([])

    axs5.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)', fontweight='bold')
    axs5.get_xaxis().set_label_coords(1.1, -0.1)

    #handles, labels = plt.gca().get_legend_handles_labels()
    #order = list(range(len(langevin_damp)))
    #axs6.legend(
        #[handles[idx] for idx in order], [labels[idx] for idx in order],
        #loc=(1.10, 0.05),  framealpha=1.0,
        #title="Langevin friction\ncoefficient " + r"$\gamma_i$")

    tbox = TextArea(
        "B", textprops=dict(color='k', fontsize=BIGGER_SIZE))

    anchored_tbox = AnchoredOffsetbox(
        loc='upper right', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(-0.15, 0.95),
        bbox_transform=axs4.transAxes, borderpad=0.)
    
    axs4.add_artist(anchored_tbox)
        


# Save figure
plt.savefig(
    os.path.join(resdir, 'Paper_IR_model_damp.png'),
    format='png', dpi=dpi)
#plt.show()
plt.close()




