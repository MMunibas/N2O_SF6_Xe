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
# Setup Parameters
#------------------------

# Number of parallel tasks
tasks = 20

# Requested jobs
request_evaluation = True

# Case information
sys_cdir = [
    '1_rot_0',
    '2_rot_y',
    '3_rot_x',
    '4_boltzman']
sys_trgt = 'N2O'

# Temperatures [K]
T = 321.93

# Time step size in fs
dt = 1.0

# Working directories
Jdirs = "J{:s}"
smpldirs = "J{:s}_S{:s}"

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
beta = 1.0/3.1668114e-6/float(T)
hbar = 1.0
cminvtoau = 1.0/2.1947e5
const = beta*cminvtoau*hbar

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
#plt.rc('legend.title_fontsize', titlesize=MEDIUM_SIZE)  # legend title fontsize
dpi = 200

color_scheme = [
    'b', 'r', 'g', 'magenta', 'orange', 'purple', 'brown', 'darkblue',
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

#------------------------
# Evaluate Sample Runs
#------------------------

# Get cases and sample runs
rdirs = []
for cdir in sys_cdir:
    
    # Detect runs
    sdirs = glob(os.path.join(
        cdir, Jdirs.format("*"), smpldirs.format("*", "*")))
    
    # Iterate over sample runs
    for sdir in sdirs:
        rdirs.append(sdir)
        
def evaluate(i):
    
    # Working directory
    workdir = rdirs[i]
    
    if not os.path.exists(os.path.join(workdir, axsfile)) or False:
        
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
    
    #else:
        
        ## Load results
        #urot = np.load(os.path.join(workdir, axsfile))
        #acv_urot = np.load(os.path.join(workdir, acvfile))
        #dples = np.load(os.path.join(workdir, dipfile))
        
    # Compute IR spectra
    if not os.path.exists(os.path.join(workdir, spcfile)) or False:
        
        # Load results
        urot = np.load(os.path.join(workdir, axsfile))
        acv_urot = np.load(os.path.join(workdir, acvfile))
        dples = np.load(os.path.join(workdir, dipfile))
        
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
        
        # Compute IR spectra
        acv = (
            acv_du[:, 0]*acv_urot[:, 0]
            + acv_du[:, 1]*acv_urot[:, 1]
            + acv_du[:, 2]*acv_urot[:, 2]
            )
        blackman_acv = acv*np.blackman(Nframes)
        spec = np.abs(np.fft.rfftn(blackman_acv))*np.tanh(const*freq/2.)
        
        # Save spectra
        np.save(os.path.join(workdir, spcfile), spec)
        np.save(os.path.join(workdir, dcvfile), acv)
        np.save(os.path.join(workdir, frqfile), freq)
        
    #else:
        
        ## Load spectra
        #spec = np.load(os.path.join(workdir, spcfile))
        #acv = np.load(os.path.join(workdir, dcvfile))
        #freq = np.load(os.path.join(workdir, frqfile))
    
if request_evaluation and tasks==1:
    for i in range(0, len(rdirs)):
        evaluate(i)
elif request_evaluation:    
    if __name__ == '__main__':
        pool = Pool(tasks)
        pool.imap(evaluate, range(0, len(rdirs)))
        pool.close()
        pool.join()
        
        
#-----------------------------
# Plot Case IR
#-----------------------------

avgfreq = 1.0
def moving_average(data_set, periods=9):
    weights = np.ones(periods) / periods
    return np.convolve(data_set, weights, 'same')

# Figure arrangement
figsize = (8, 6)
left = 0.04
bottom = 0.15
row = np.array([0.80, 0.00])
column = np.array([0.29, 0.03])

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

# Panel labels
clabels = ["A", "B", "C", "D"]

# Iterate over cases
for ic, cdir in enumerate(sys_cdir):
    
    if ic==0:
        
        # Sample dirs - just one
        sdirs = glob(os.path.join(
            cdir, Jdirs.format("*"), smpldirs.format("*", "*")))
        
        for sdir in sdirs:
            
            # Get J state
            Ji = int(sdir.split("/")[-1].split("_")[0][1:])
            
            # Load results
            urot = np.load(os.path.join(sdir, axsfile))
            acv_urot = np.load(os.path.join(sdir, acvfile))
            dples = np.load(os.path.join(sdir, dipfile))
            spec = np.load(os.path.join(sdir, spcfile))
            acv = np.load(os.path.join(sdir, dcvfile))
            freq = np.load(os.path.join(sdir, frqfile))
            
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
            label = r'$j={:d}$'.format(Ji)
            
            # Bending mode
            
            overlap = 1.0
            shift = len(sys_cdir) - 1 - ic
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
            
            print(ic)
            ilocmax = scipy.signal.argrelmax(overlap*avgspec[select1], order=10)
            print(freq[select1][ilocmax])
            ilocmax = scipy.signal.argrelmax(overlap*avgspec[select2], order=10)
            print(freq[select2][ilocmax])
            ilocmax = scipy.signal.argrelmax(overlap*avgspec[select3], order=10)
            print(freq[select3][ilocmax])
            
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
            
    elif ic == 1 or ic == 2:
        
        # J state samples
        J10 = Jmax
        J05 = int(Jmax*0.5)
        J15 = int(Jmax*1.5)
        
        # Sample dirs - three samples
        sdir10 = os.path.join(
            cdir, Jdirs.format(str(J10)), smpldirs.format(str(J10), "0"))
        sdir05 = os.path.join(
            cdir, Jdirs.format(str(J05)), smpldirs.format(str(J05), "0"))
        sdir15 = os.path.join(
            cdir, Jdirs.format(str(J15)), smpldirs.format(str(J15), "0"))
        
        
        # Load results - J10
        spec = np.load(os.path.join(sdir10, spcfile))
        freq = np.load(os.path.join(sdir10, frqfile))
        
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
        if ic == 1:
            label = r'$j(\vec{e}_y)=$' + '{:d}'.format(J10)
        else:
            label = r'$j(\vec{e}_x)=$' + '{:d}'.format(J10)
        
        # Bending mode
        overlap = 1.0
        shift = len(sys_cdir) - 1 - ic
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
        
        #print(ic)
        #ilocmax = scipy.signal.argrelmax(overlap*avgspec[select1], order=10)
        #print(freq[select1][ilocmax])
        #ilocmax = scipy.signal.argrelmax(overlap*avgspec[select2], order=10)
        #print(freq[select2][ilocmax])
        #ilocmax = scipy.signal.argrelmax(overlap*avgspec[select3], order=10)
        #print(freq[select3][ilocmax])
        
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
        
    elif ic == 3:
        
        # Intialize average spectra
        sdir = os.path.join(
            cdir, Jdirs.format("0"), smpldirs.format("0", "0"))
        spec = np.load(os.path.join(sdir, spcfile))
        freq = np.load(os.path.join(sdir, frqfile))
        avgspec = np.zeros_like(spec)
        
        # Select ranges
        select1 = np.logical_and(
            freq > rnge1[0], freq < rnge1[1])
        select2 = np.logical_and(
            freq > rnge2[0], freq < rnge2[1])
        select3 = np.logical_and(
            freq > rnge3[0], freq < rnge3[1])
        
        # Sample dirs
        sdirs = glob(os.path.join(
            cdir, Jdirs.format("*"), smpldirs.format("*", "*")))
        snum = 0.0
        Bavg = np.zeros(50, dtype=float)
        snum = np.zeros(50, dtype=float)
        for sdir in sdirs:
            
            # Get J state
            Ji = int(sdir.split("/")[-1].split("_")[0][1:])
            
            # Load results
            urot = np.load(os.path.join(sdir, axsfile))
            acv_urot = np.load(os.path.join(sdir, acvfile))
            dples = np.load(os.path.join(sdir, dipfile))
            spec = np.load(os.path.join(sdir, spcfile))
            acv = np.load(os.path.join(sdir, dcvfile))
            freq = np.load(os.path.join(sdir, frqfile))
        
            # Rotational damping
            Nframes = acv_urot.shape[0]
            times = dt*np.arange(Nframes)*1e-3
            taui = 9.5
            rot_damp = np.exp(-times/taui)
            
            # Align dipole with rotational axis and get
            du = np.zeros([Nframes, 3], dtype=float)
            acv_du = np.zeros([Nframes, 3], dtype=float)
            for iax in range(3):
                
                du[:, iax] = np.sum(urot[:, iax]*dples, axis=1)
                acv_du[:, iax] = acovf(du[:, iax], fft=True)
            
            # Compute IR spectra
            acv = (
                acv_du[:, 0]*acv_urot[:, 0]*rot_damp
                + acv_du[:, 1]*acv_urot[:, 1]*rot_damp
                + acv_du[:, 2]*acv_urot[:, 2]*rot_damp
                )
            blackman_acv = acv*np.blackman(Nframes)
            spec = np.abs(np.fft.rfftn(blackman_acv))*np.tanh(const*freq/2.)
            
            avgfreq_P = 2.0
            Nave = int(avgfreq_P/(freq[1] - freq[0]))
            specma = moving_average(spec, Nave)
            
            # Get maxima
            #print(Ji)
            #ilocmax = scipy.signal.argrelmax(specma[select1], order=10)
            #print(freq[select1][ilocmax])
            #ilocmax = scipy.signal.argrelmax(specma[select2], order=10)
            #print(freq[select2][ilocmax])
            #ilocmax = scipy.signal.argrelmax(specma[select3], order=10)
            #print(freq[select3][ilocmax])
            
            #v0 = 2208.52787431
            #mfreq = freq[select3][ilocmax]
            #rfreq = mfreq[mfreq < v0][-1]
            #pfreq = mfreq[mfreq > v0][0]
            ##print(mfreq)
            #if Ji and Ji <= 26:
                #B = (pfreq - rfreq)/(4.0*Ji)
                #if B < 1.0:
                    ##snum += 1.0
                    ##Bavg = Bavg*(snum - 1.0)/snum + B*1.0/snum
                    #Bavg[Ji] += B
                    #snum[Ji] += 1.0
                ##print(rfreq, pfreq, pfreq - rfreq, (pfreq - rfreq)/(4.0*Ji), Bavg)
                #temp = np.zeros_like(Bavg)
                #temp[snum > 0.0] = Bavg[snum > 0.0]/snum[snum > 0.0]
                #print(temp)
                #print(np.mean(temp[2:27]))
                
            # Add spectra
            avgspec += spec*P_J[Ji]#/float(Nsmpl)
            
        # Apply moving average
        avgfreq_P = 5.0
        Nave = int(avgfreq_P/(freq[1] - freq[0]))
        avgspec = moving_average(avgspec, Nave)

        # Scale avgspec
        select = np.logical_and(
            freq > rnge1[0], freq < rnge3[1])
        avgspec /= np.max(avgspec[select])
        
        # Plot
        label = r"$j=P_j(T)$"
        
        # Bending mode
        overlap = 1.0
        shift = len(sys_cdir) - 1 - ic
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
            'size'   : TINY_SIZE}
        axs1.text(
            rnge1[0] + (rnge1[1] - rnge1[0])*0.01, 
            shift + 0.3, 'x{:.0f}'.format(scale1),
            fontdict=scale_dict)
        axs2.text(
            rnge2[0] + (rnge2[1] - rnge2[0])*0.01, 
            shift + 0.3, 'x{:.0f}'.format(scale2),
            fontdict=scale_dict)
        axs3.text(
            rnge3[0] + (rnge3[1] - rnge3[0])*0.01, 
            shift + 0.3, 'x{:.0f}'.format(scale3),
            fontdict=scale_dict)
        
        # Load experimental spectra
        exp_spec = loadmat("ref/rho0p16.mat")["norm_0p16"].reshape(-1)
        exp_freq = loadmat("ref/xx.mat")["xx"].reshape(-1)
        
        # Prepare spectra interpolation
        spec1 = avgspec*scale3
        fspec1 = interp1d(freq, spec1, kind='cubic')
        spec2 = exp_spec/np.max(exp_spec)
        
        # Evaluation function
        ishift = 0.0
        def func_overlap(ishift):
            
            # Interpolate spectra on grid
            ifreq = np.arange(exp_freq[0], exp_freq[-1], 1.0)
            try:
                ispec1 = fspec1(exp_freq - ishift)
            except ValueError:
                return np.inf
            
            # Compute overlap
            opt_overlap = -np.sum(ispec1*spec2)
            
            return opt_overlap
        
        result = minimize_scalar(func_overlap)
        opt_shift = result.x
        
        axs3.plot(
            exp_freq - opt_shift, 
            overlap*exp_spec + shift, '--k')
        print(opt_shift)
        
    else:
        
        axs3.plot(
            freq[select3], 
            overlap*avgspec[select3]*scale3 + ic, '-', 
            color=color_scheme[ic], label=label)
            
    tbox = TextArea(
        clabels[ic], textprops=dict(color='k', fontsize=BIGGER_SIZE))

    shift = len(sys_cdir) - 1 - ic
    anchored_tbox = AnchoredOffsetbox(
        loc='upper left', child=tbox, pad=0., frameon=False,
        bbox_to_anchor=(0.05, (shift + 1.)/len(sys_cdir) - 0.02),
        bbox_transform=axs1.transAxes, borderpad=0.)
    
    axs1.add_artist(anchored_tbox)
    
            
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
axs3.legend(
    [handles[idx] for idx in order], [labels[idx] for idx in order],
    loc=(1.10, 0.05),  framealpha=1.0,
    title=r"N$_2$O model")

plt.savefig('Paper_IR_model.png', format='png', dpi=dpi)
#plt.show()
plt.close()
