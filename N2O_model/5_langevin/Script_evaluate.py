# Basics
import os
import sys
import numpy as np
from glob import glob
import scipy
from scipy.optimize import curve_fit
import string

# ASE - Basics
from ase import Atoms
import ase.units as units

# MDAnalysis
import MDAnalysis

# Statistics
from statsmodels.tsa.stattools import acovf

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#------------------------
# Setup Parameters
#------------------------

# Maximum range of J states
maxJ = None

# Langevin damping factors
#langevin_damp = [1.0, 0.1, 0.01, 0.001]
langevin_damp = [
    1000.0, 100.0, 10.0,
    1.0, 0.1, 0.01, 0.001,
    0.0001, 0.00001, 0.000001]
langevin_damp = [
    10.0, 1.0, 0.1, 
    0.01, 0.0001, 0.000001]

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


# Fontsize
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18

plt.rc('font', size=SMALL_SIZE, weight='bold')  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Figure resolution
dpi = 200

# Line colors
color_scheme = [
    'b', 'r', 'g', 'purple', 'orange', 'magenta', 'brown', 'darkblue',
    'darkred', 'darkgreen', 'darkgrey', 'olive']



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
# Evaluate Sample Runs
#------------------------

if maxJ is None:
    
    if rot_temp == 0.0:
        Ji = 0
    else:
        Ji = np.argmax(P_J)
    
    # Iterate over damping factors
    for ldamp in langevin_damp:

        # Iterate over samples
        for ismpl in range(Nsmpl):
        
            # Working directory
            workdir = os.path.join(
                Jdirs.format(Ji), 
                smpldirs.format(Ji, ldamp, ismpl))
            
            if not os.path.exists(os.path.join(workdir, axsfile)):
                
                # Open dcd file
                dcd = MDAnalysis.Universe(
                    os.path.join(workdir, psffile), 
                    os.path.join(workdir, dcdfile))
                
                # Get trajectory parameter
                Nframes = len(dcd.trajectory)
                
                # Atom number and indices
                Natms = 3
                indcs = np.array([0, 1, 2])
                
                # Get masses
                masses = dcd._topology.masses.values
                m = masses[indcs]
                
                # Get atom charges
                charges = dcd._topology.charges.values
                c = charges[indcs]
                
                # Get atom positions and velocities
                pstns = np.zeros([Nframes, Natms, 3], dtype=np.float32)
                
                for ii, frame in enumerate(dcd.trajectory):
                    
                    # Atom positions (atom, cart)
                    pstns[ii] = frame._pos[indcs]
                    
                # Result arrays
                urot = np.zeros([Nframes, 3, 3], dtype=np.float32)
                acv_urot = np.zeros([Nframes, 3], dtype=np.float32)
        
                # Get COM
                coms = (
                    np.sum(m.reshape(1, -1, 1)*pstns, axis=1)
                    /np.sum(m))
                
                # Center positions
                poscm = pstns - coms.reshape(Nframes, 1, 3)
                
                # Moments and principle axis of inertia
                Icart = np.zeros((3, 3), dtype=np.float32)
                for ii, posi in enumerate(poscm):
                    
                    Icart[:, :] = 0.0
                    Icart[0, 0] = np.sum(m*(posi[:, 1]**2 + posi[:, 2]**2))
                    Icart[1, 1] = np.sum(m*(posi[:, 0]**2 + posi[:, 2]**2))
                    Icart[2, 2] = np.sum(m*(posi[:, 0]**2 + posi[:, 1]**2))
                    Icart[0, 1] = Icart[1, 0] = np.sum(
                        -m*posi[:, 0]*posi[:, 1])
                    Icart[0, 2] = Icart[2, 0] = np.sum(
                        -m*posi[:, 0]*posi[:, 2])
                    Icart[1, 2] = Icart[2, 1] = np.sum(
                        -m*posi[:, 1]*posi[:, 2])
                    
                    Ip, Ib = np.linalg.eigh(Icart)
                    urot[ii, :, :] = Ib.T
                
                iax_choice = [0]
                for iax in iax_choice:
                    abdot = np.sum(urot[1:, iax]*urot[:-1, iax], axis=1)
                    sabdot = np.sign(abdot)
                    current_correction = 1.0
                    for ib, si in enumerate(sabdot):
                        current_correction *= si
                        urot[ib + 1, iax] = (
                            urot[ib + 1, iax]*current_correction)
                
                # Dipoles
                dples = np.sum(poscm*c.reshape(1, Natms, 1), axis=1)
                
                # Axial and perpendicular dipole vector
                adple = (
                    urot[:, 0]
                    *np.sum(dples*urot[:, 0], axis=1).reshape(
                        Nframes, 1)
                    )
                pdple = dples - adple

                refax = pdple[0]
                ortax = np.cross(refax, urot[:, 0])
                ortax = ortax/np.sqrt(np.sum(ortax**2, axis=1)).reshape(-1, 1)
                plnax = np.cross(ortax, urot[:, 0])
                
                # Assign new vectors
                urot[:, 1] = plnax
                urot[:, 2] = ortax
                
                iax_choice = [1, 2]
                for iax in iax_choice:
                    abdot = np.sum(urot[1:, iax]*urot[:-1, iax], axis=1)
                    sabdot = np.sign(abdot)
                    current_correction = 1.0
                    for ib, si in enumerate(sabdot):
                        current_correction *= si
                        urot[ib + 1, iax] = (
                            urot[ib + 1, iax]*current_correction)
                
                # Axis dot product
                for iax in range(3):
                    acv_urot[:, iax] = np.sum(
                        urot[0, iax].reshape(1, -1)
                        *urot[:, iax], axis=1)
                
                # Save results
                np.save(os.path.join(workdir, axsfile), urot)
                np.save(os.path.join(workdir, acvfile), acv_urot)
                np.save(os.path.join(workdir, dipfile), dples)
            
            else:
                
                # Load results
                urot = np.load(os.path.join(workdir, axsfile))
                acv_urot = np.load(os.path.join(workdir, acvfile))
                dples = np.load(os.path.join(workdir, dipfile))
                
            # Compute IR spectra
            Nframes = dples.shape[0]
            Nfreq = int(Nframes/2) + 1
            
            if not os.path.exists(os.path.join(workdir, spcfile)) or False:
                
                # Frequency array
                freq = np.arange(Nfreq)/float(Nframes)/dt*jiffy
                
                # Align dipole with rotational axis and get 
                du = np.zeros([Nframes, 3], dtype=float)
                acv_du = np.zeros([Nframes, 3], dtype=float)
                acv_dples = np.zeros([Nframes, 3], dtype=float)
                for iax in range(3):
                    
                    
                    du[:, iax] = np.sum(urot[:, iax]*dples, axis=1)
                    acv_du[:, iax] = acovf(du[:, iax], fft=True)
                    acv_dples[:, iax] = acovf(dples[:, iax], fft=True)
                
                # Compute IR spectra
                #acv = (
                    #acv_du[:, 0]*acv_urot[:, 0]
                    #+ acv_du[:, 1]*acv_urot[:, 1]
                    #+ acv_du[:, 2]*acv_urot[:, 2]
                    #)
                acv = (
                    acv_dples[:, 0]
                    + acv_dples[:, 1]
                    + acv_dples[:, 2]
                    )
                blackman_acv = acv*np.blackman(Nframes)
                spec = np.abs(np.fft.rfftn(blackman_acv))*np.tanh(const*freq/2.)
                
                # Save spectra
                np.save(os.path.join(workdir, spcfile), spec)
                np.save(os.path.join(workdir, dcvfile), acv)
                np.save(os.path.join(workdir, frqfile), freq)
                
            else:
                
                # Load spectra
                spec = np.load(os.path.join(workdir, spcfile))
                acv = np.load(os.path.join(workdir, dcvfile))
                freq = np.load(os.path.join(workdir, frqfile))
        
        
#-------------------------
# Plot Dipole Correlation
#-------------------------


## Figure arrangement
#figsize = (12, 8)
#left = 0.12
#bottom = 0.15
#row = np.array([0.70, 0.00])
#column = np.array([0.50, 0.10])

## Plot Correlation
#fig1 = plt.figure(figsize=figsize)

## Initialize axes
#axs1 = fig1.add_axes([left, bottom, column[0], row[0]])

#plt_log = False
#pltt = 1.0
#tstart = 0.00
#cmax = 0.0
#cmin = 0.0

#if maxJ is None:
    
    #if rot_temp == 0.0:
        #Ji = 0
    #else:
        #Ji = np.argmax(P_J)
    
    #for ic, ldamp in enumerate(langevin_damp):
            
        ## Intialize average spectra
        #workdir = os.path.join(Jdirs.format(Ji), smpldirs.format(Ji, ldamp, 0))
        #urot = np.load(os.path.join(workdir, axsfile))
        #acv_urot = np.load(os.path.join(workdir, acvfile))
        #dples = np.load(os.path.join(workdir, dipfile))
        #Nframes = dples.shape[0]
        #time = np.arange(0.0, Nframes*dt, dt)/1000.0
        
        ## Align dipole with rotational axis and get 
        #du = np.zeros([Nframes, 3], dtype=float)
        #acv_du = np.zeros([Nframes, 3], dtype=float)
        #acv_dples = np.zeros([Nframes, 3], dtype=float)
        #for iax in range(3):
            
            #du[:, iax] = np.sum(urot[:, iax]*dples, axis=1)
            #acv_du[:, iax] = acovf(du[:, iax], fft=True)
            #acv_dples[:, iax] = acovf(dples[:, iax], fft=True)
        
        ## Plot
        #label = r'$\gamma_i$ = {:.1E}'.format(ldamp)
        #iax = 2
        #if plt_log:
            #log_acv_dples = np.zeros_like(acv_dples[:, iax])
            #log_acv_dples[acv_dples[:, iax] > 0.0] = (
                #np.log(acv_dples[:, iax][acv_dples[:, iax] > 0.0]))
            #log_acv_dples[acv_dples[:, iax] <= 0.0] = np.nan
            #axs1.plot(
                #time, log_acv_dples, '-', 
                #color=color_scheme[ic], label=label)
        #else:
            #axs1.plot(
                #time, acv_dples[:, iax], '-', 
                #color=color_scheme[ic], label=label)
        
        #if np.max(acv_dples[:, iax]) > cmax:
            #cmax = np.max(acv_dples[:, iax])
        #if np.min(acv_dples[:, iax]) < cmin:
            #cmin = np.min(acv_dples[:, iax])

    #axs1.set_xlim([-pltt/20.0, pltt])
    
    #fig1.suptitle(
        #r'Dipole-Dipole (z-component) correlation function',
        #fontweight='bold')

    #axs1.legend(
        #loc=[1.05, -0.20], 
        #title="Langevin friction\ncoefficient " + r"$\gamma_i$")
    
    #axs1.set_xlabel(r'Time (ps)', fontweight='bold')
    #axs1.get_xaxis().set_label_coords(0.5, -0.1)
    #if plt_log:
        #ylabel = (
            #r'ln($\left< \left| \mu_z(0) \right|~\left| \mu_z(t) \right| \right>$) '
            #+ r'(ln(ps$^{-1}$))')
        #axs1.set_ylabel(r'{:s}'.format(ylabel), fontweight='bold')
    #else:
        #ylabel = (
            #r'$\left< \left| \mu_z(0) \right|~\left| \mu_z(t) \right| \right>$ ' 
            #+ r'(ps$^{-1}$)')
        #axs1.set_ylabel(ylabel, fontweight='bold')
    #axs1.get_yaxis().set_label_coords(-0.08, 0.50)
    
    #if plt_log:
        #figtitle = 'DipCorr_z_damping_log.png'
    #else:
        #figtitle = 'DipCorr_z_damping.png'
        
    #fig1.savefig(figtitle, format='png', dpi=dpi)
    #plt.close(fig1)



#------------------------
# Plot Sample Runs
#------------------------

moving_freq = 5.0
def moving_average(data_set, periods=9):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, 'same')


# Frequency ranges
rngen = 200
rnge1 = [580, 650]
rnge2 = [1180, 1350]
rnge3 = [2120, 2310]

# Figure arrangement
figsize = (8, 6)
left = 0.04
bottom = 0.15
row = np.array([0.80, 0.00])
column = np.array([0.29, 0.03])

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

# Panel labels
clabels = list(string.ascii_uppercase)

if maxJ is None:
    
    if rot_temp == 0.0:
        Ji = 0
    else:
        Ji = np.argmax(P_J)
    
    for ic, ldamp in enumerate(langevin_damp):
            
        # Intialize average spectra
        workdir = os.path.join(Jdirs.format(Ji), smpldirs.format(Ji, ldamp, 0))
        spec = np.load(os.path.join(workdir, spcfile))
        freq = np.load(os.path.join(workdir, frqfile))
        avgfreq = freq
        avgspec = np.zeros_like(spec)
        
        # Iterate over samples
        for ismpl in range(Nsmpl):
            
            # Working directory
            workdir = os.path.join(
                Jdirs.format(Ji), 
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
        shift = len(langevin_damp) - 1 - ic
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
            
        #tbox = TextArea(
            #clabels[ic], textprops=dict(color='k', fontsize=BIGGER_SIZE))

        #shift = len(langevin_damp) - 1 - ic
        #anchored_tbox = AnchoredOffsetbox(
            #loc='upper left', child=tbox, pad=0., frameon=False,
            #bbox_to_anchor=(0.05, (shift + 1.)/len(langevin_damp) - 0.02),
            #bbox_transform=axs1.transAxes, borderpad=0.)
        
        #axs1.add_artist(anchored_tbox)
        
    # Axis range
    axs1.set_xlim(rnge1)
    axs2.set_xlim(rnge2)
    axs3.set_xlim(rnge3)
    axs1.set_ylim([0, len(langevin_damp) - 1 + overlap])
    axs2.set_ylim([0, len(langevin_damp) - 1 + overlap])
    axs3.set_ylim([0, len(langevin_damp) - 1 + overlap])

    # Axis labels
    #axs1.set_xticks([])
    axs2.set_xticks([1200, 1250, 1300, 1350])
    axs3.set_xticks([2150, 2200, 2250, 2300])
    axs1.set_yticks([ii for ii in range(len(langevin_damp))])
    axs2.set_yticks([ii for ii in range(len(langevin_damp))])
    axs3.set_yticks([ii for ii in range(len(langevin_damp))])
    axs1.set_yticklabels([])
    axs2.set_yticklabels([])
    axs3.set_yticklabels([])

    axs2.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)', fontweight='bold')
    axs2.get_xaxis().set_label_coords(1.1, -0.1)

    handles, labels = plt.gca().get_legend_handles_labels()
    order = list(range(len(langevin_damp)))
    axs3.legend(
        [handles[idx] for idx in order], [labels[idx] for idx in order],
        loc=(1.10, 0.05),  framealpha=1.0,
        title="Langevin friction\ncoefficient " + r"$\gamma_i$")

    plt.savefig('Paper_IR_damping.png', format='png', dpi=dpi)
    #plt.show()
    plt.close()

