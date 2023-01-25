# Basics
import os
import sys
import numpy as np

# Matplotlib
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Statistics
from statsmodels.tsa.stattools import acovf

# Miscellaneous
import ase.units as units
import scipy
from scipy.io import loadmat
from scipy.optimize import minimize, minimize_scalar
from scipy.interpolate import interp1d

# Temperatures [K]
T = 321.93

# Critical temperature and density of SF6
M = 146.05 # g/mol
Tcrit = 273.15 + 45.6 # K
rhocrit = 0.74 # g/cm**3

# Relative density
rhostar = [0.04, 0.10, 0.16, 0.30, 0.67, 0.86, 0.99, 1.17, 1.36, 1.51, 1.87]

slv_states = [
    'Gas', 'Gas', 'Gas', 'Gas', 
    'SCF', 'SCF', 'SCF', 'SCF', 'SCF', 'SCF',
    'Liquid']

# Experimental rotational lifetime
rho_exp =      [0.16, 0.30, 0.67, 0.86, 0.99, 1.17, 1.36, 1.51, 1.87]
con_exp =      [0.82, 1.51, 3.43, 4.41, 5.06, 5.79, 6.94, 7.70, 9.58]
tau_exp =      [9.5,  6.0,  2.8,  2.4,  2.3,  1.9,  1.4,  0.9, np.nan]
err_exp =      [0.5,  0.4,  0.3,  0.2,  0.1,  0.1,  0.1,  0.1, np.nan]
tau_coll_exp = [6.7,  3.6,  1.7,  1.5,  1.4,  1.0,  0.8,  0.7, np.nan]
Texp = 321.9

# Experimental IR spectra lines for N2O in gas phase (P-, Q- and R-branch)
exp_ir = [2211, 2221, 2236]

exp_fspc = [
    "source/SF6_FTIR/rho0p16.mat",
    "source/SF6_FTIR/rho0p30.mat",
    "source/SF6_FTIR/rho0p67.mat",
    "source/SF6_FTIR/rho0p86.mat",
    "source/SF6_FTIR/rho0p99.mat",
    "source/SF6_FTIR/rho1p17.mat",
    "source/SF6_FTIR/rho1p36.mat",
    "source/SF6_FTIR/rho1p51.mat",
    "source/SF6_FTIR/rho1p87.mat"]
exp_kspc = [
    "norm_0p16",
    "norm_0p30",
    "norm_0p67",
    "norm_0p86",
    "norm_1p0",
    "norm_1p17",
    "norm_1p36",
    "norm_1p51",
    "norm_1p87"]
exp_ffrq = "source/SF6_FTIR/xx.mat"
exp_kfrq = "xx"




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

# System psf file
traj_psffile = 'init.n2o_sf6.psf'

# System coord file
sys_fcrd = 'init.n2o_sf6.crd'

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
#plt.rc('legend.title_fontsize', titlesize=MEDIUM_SIZE)  # legend title fontsize
dpi = 200

color_scheme = [
    'b', 'r', 'g', 'purple', 'orange', 'magenta', 'brown', 'darkblue',
    'darkred', 'darkgreen', 'darkgrey', 'olive']

#-----------------------------
# Plot IR
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

# Only sample 0
ismpl = 0

# Experimental result counter
iexp = 0

# Optimal frequency shift
opt_shift = np.zeros_like(Vm)

figlabels = ['A', 'B', 'C']

for iv, Vmi in enumerate(Vm):
    
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(T)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(T), int(Vmi))
    
    # IR spectra file
    ifile = os.path.join(
        workdir, 'irspc_{:s}.npy'.format(tag))
    ffile = os.path.join(
        workdir, 'irfrq_{:s}.npy'.format(tag))
    
    # Dipole file
    dfile = os.path.join(
        workdir, 'dipos_{:s}.npy'.format(tag))
    
    # Time file
    tfile = os.path.join(
        workdir, 'times_{:s}.npy'.format(tag))
    
    # Number of frames and frequency points
    tlst = np.load(tfile)
    Nframes = len(tlst)
    Nfreq = int(Nframes/2) + 1
    
    # Frequency array
    dtime = tlst[1] - tlst[0]
    freq = np.arange(Nfreq)/float(Nframes)/dtime*jiffy
    
    if not os.path.exists(ifile) or False:
        
        # Load results
        dlst = np.load(dfile)
        
        # Weighting constant
        beta = 1.0/3.1668114e-6/float(T)
        hbar = 1.0
        cminvtoau = 1.0/2.1947e5
        const = beta*cminvtoau*hbar
        
        # Compute IR spectra
        acvx = acovf(dlst[:, 0, 0], fft=True)
        acvy = acovf(dlst[:, 0, 1], fft=True)
        acvz = acovf(dlst[:, 0, 2], fft=True)
        acv = acvx + acvy + acvz
        
        acv = acv*np.blackman(Nframes)
        spec = np.abs(np.fft.rfftn(acv))*np.tanh(const*freq/2.)
        
        # Save spectra
        np.save(ifile, spec)
        np.save(ffile, freq)
        
    else:
        
        spec = np.load(ifile)
        freq = np.load(ffile)
    
    # Apply moving average
    Nave = int(avgfreq/(freq[1] - freq[0]))
    avgspec = moving_average(spec, Nave)
    
    # Load experimental result
    if rhostar[iv] in rho_exp:
        exp_avail = True
        exp_spec = loadmat(exp_fspc[iexp])
        exp_spec = exp_spec[exp_kspc[iexp]].reshape(-1)
        iexp += 1
        exp_freq = loadmat(exp_ffrq)[exp_kfrq].reshape(-1)
        
    else:
        exp_avail = False
    
    # Select ranges
    select3 = np.logical_and(
        freq > rnge3[0], freq < rnge3[1])
    if exp_avail:
        selectexp = np.logical_and(
            exp_freq > rnge3[0], exp_freq < rnge3[1])
    
    # Overlap asymmetric band with experiment
    if exp_avail:
        
        # Prepare spectra interpolation
        spec1 = avgspec/np.max(avgspec[select3])
        fspec1 = interp1d(freq, spec1, kind='cubic')
        spec2 = exp_spec/np.max(exp_spec[selectexp])
        
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
        opt_shift[iv] = result.x
        
    else:
        
        opt_shift[iv] = np.nan
   
# Assign average shift to pure computational spectra
avg_shift = np.mean(opt_shift[~np.isnan(opt_shift)])
opt_shift[np.isnan(opt_shift)] = avg_shift

# Reset experimental result counter
iexp = 0

for iv, Vmi in enumerate(Vm):
    
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(T)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(T), int(Vmi))
    
    # IR spectra file
    ifile = os.path.join(
        workdir, 'irspc_{:s}.npy'.format(tag))
    ffile = os.path.join(
        workdir, 'irfrq_{:s}.npy'.format(tag))
    
    # Load results
    spec = np.load(ifile)
    freq = np.load(ffile)
    
    # Apply moving average
    Nave = int(avgfreq/(freq[1] - freq[0]))
    avgspec = moving_average(spec, Nave)
    
    # Load experimental result
    if rhostar[iv] in rho_exp or rhostar[iv]==1.55:
        exp_avail = True
        exp_spec = loadmat(exp_fspc[iexp])
        exp_spec = exp_spec[exp_kspc[iexp]].reshape(-1)
        iexp += 1
        exp_freq = loadmat(exp_ffrq)[exp_kfrq].reshape(-1)
        
    else:
        exp_avail = False
    
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
    if exp_avail:
        selectexp = np.logical_and(
            exp_freq > rnge3[0], exp_freq < rnge3[1])
    
    # Pressure ratio
    p = M/Vmi/rhocrit
    
    # Plot
    label = r'{:.2f} [{:.2f}]'.format(CM[iv], p)
    
    # Bending mode
    overlap = 1.5
    scale1 = 1./np.max(avgspec[select1])
    axs1.plot(
        freq[select1], overlap*avgspec[select1]*scale1 + iv, '-', 
        color=color_scheme[iv], label=label)
    
    # Symmetric stretch
    scale2 = 1./np.max(avgspec[select2])
    axs2.plot(
        freq[select2], overlap*avgspec[select2]*scale2 + iv, '-', 
        color=color_scheme[iv], label=label)
    
    # Asymmetric stretch
    scale3 = 1./np.max(avgspec[select3])
    axs3.plot(
        freq[select3] + opt_shift[iv], overlap*avgspec[select3]*scale3 + iv, '-', 
        color=color_scheme[iv], label=label)
    
    ilocmax = scipy.signal.argrelmax(avgspec[select3], order=4)
    print(Vmi, 'Comp, freq_max', (freq[select3] + opt_shift[iv])[ilocmax])
    ilocmax = scipy.signal.argrelmax(exp_spec[selectexp], order=4)
    print(Vmi, 'Comp, freq_max', exp_freq[selectexp][ilocmax])
    
    
    # Add scaling factors
    scale_dict = {
        'family' : 'monospace',
        'weight' : 'light',
        'size'   : TINY_SIZE}
    shift_dict = {
        'family' : 'monospace',
        'weight' : 'bold',
        'size'   : TINY_SIZE}
    axs1.text(
        rnge1[0] + (rnge1[1] - rnge1[0])*0.01, 
        iv + 0.5, 'x{:.0f}'.format(scale1),
        fontdict=scale_dict)
    axs2.text(
        rnge2[0] + (rnge2[1] - rnge2[0])*0.01, 
        iv + 0.5, 'x{:.0f}'.format(scale2),
        fontdict=scale_dict)
    axs3.text(
        rnge3[0] + (rnge3[1] - rnge3[0])*0.01, 
        iv + 0.5, 'x{:.0f}'.format(scale3),
        fontdict=scale_dict)
    axs3.text(
        rnge3[0] + (rnge3[1] - rnge3[0])*0.01, 
        iv + 0.1, r'$\omega${:+.0f}'.format(opt_shift[iv]),
        fontdict=shift_dict)
    
    # Add system state
    state_dict = {
        'style'  : 'normal',
        'weight' : 'bold',
        'size'   : XSMALL_SIZE}
    axs3.text(
        rnge3[0] + (rnge3[1] - rnge3[0])*0.99, 
        iv + 0.1, '{:s}'.format(slv_states[iv]),
        va='bottom', ha='right',
        fontdict=state_dict)
    
    # Plot Experimental spectra
    if exp_avail:
        scaleexp = 1./np.max(exp_spec[selectexp])
        axs3.plot(
            exp_freq[selectexp], 
            overlap*exp_spec[selectexp]*scaleexp + iv, '--', 
            color='black')#color_scheme[iv])
        
# Axis range
axs1.set_xlim(rnge1)
axs2.set_xlim(rnge2)
axs3.set_xlim(rnge3)
axs1.set_ylim([0, len(Vm) - 1 + overlap])
axs2.set_ylim([0, len(Vm) - 1 + overlap])
axs3.set_ylim([0, len(Vm) - 1 + overlap])

# Axis labels
#axs1.set_xticks([])
axs2.set_xticks([1200, 1250, 1300, 1350])
axs3.set_xticks([2150, 2200, 2250, 2300])
axs1.set_yticks([ii for ii in range(len(Vm))])
axs2.set_yticks([ii for ii in range(len(Vm))])
axs3.set_yticks([ii for ii in range(len(Vm))])
axs1.set_yticklabels([])
axs2.set_yticklabels([])
axs3.set_yticklabels([])

axs2.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)', fontweight='bold')
axs2.get_xaxis().set_label_coords(1.1, -0.1)

#axs1.set_title(r'$\nu_\mathrm{\delta}$', fontweight='bold')
#axs2.set_title(r'$\nu_\mathrm{s}$', fontweight='bold')
#axs3.set_title(r'$\nu_\mathrm{as}$', fontweight='bold')


tbox = TextArea(figlabels[0], textprops=dict(color='k', fontsize=BIGGER_SIZE))
anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.02, 0.99),
    bbox_transform=axs1.transAxes, borderpad=0.)
axs1.add_artist(anchored_tbox)

tbox = TextArea(figlabels[1], textprops=dict(color='k', fontsize=BIGGER_SIZE))
anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.02, 0.99),
    bbox_transform=axs2.transAxes, borderpad=0.)
axs2.add_artist(anchored_tbox)

tbox = TextArea(figlabels[2], textprops=dict(color='k', fontsize=BIGGER_SIZE))
anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.02, 0.99),
    bbox_transform=axs3.transAxes, borderpad=0.)
axs3.add_artist(anchored_tbox)



handles, labels = plt.gca().get_legend_handles_labels()
order = list(range(len(Vm)))[::-1]
axs3.legend(
    [handles[idx] for idx in order], [labels[idx] for idx in order],
    loc=(1.05, 0.05),  framealpha=1.0,
    title=r'$c$ (M) [$\rho*$]'.format(sys_tag[sys_solv]),
    title_fontsize=MEDIUM_SIZE)


plt.savefig(
    os.path.join(
        maindir, rsltdir, 'Paper_IR_{:s}.png'.format(sys_solv)),
    format='png', dpi=dpi)
plt.close()
#plt.show()























#-----------------------------
# Plot IR - SI
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
rnge21 = [1180, 1260]
rnge22 = [1260, 1350]
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

# Only sample 0
ismpl = 0

# Experimental result counter
iexp = 0

# Optimal frequency shift
opt_shift = np.zeros_like(Vm)

figlabels = ['A', 'B', 'C']

for iv, Vmi in enumerate(Vm):
    
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(T)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(T), int(Vmi))
    
    # IR spectra file
    ifile = os.path.join(
        workdir, 'irspc_{:s}.npy'.format(tag))
    ffile = os.path.join(
        workdir, 'irfrq_{:s}.npy'.format(tag))
    
    # Dipole file
    dfile = os.path.join(
        workdir, 'dipos_{:s}.npy'.format(tag))
    
    # Time file
    tfile = os.path.join(
        workdir, 'times_{:s}.npy'.format(tag))
    
    # Number of frames and frequency points
    tlst = np.load(tfile)
    Nframes = len(tlst)
    Nfreq = int(Nframes/2) + 1
    
    # Frequency array
    dtime = tlst[1] - tlst[0]
    freq = np.arange(Nfreq)/float(Nframes)/dtime*jiffy
    
    if not os.path.exists(ifile) or False:
        
        # Load results
        dlst = np.load(dfile)
        
        # Weighting constant
        beta = 1.0/3.1668114e-6/float(T)
        hbar = 1.0
        cminvtoau = 1.0/2.1947e5
        const = beta*cminvtoau*hbar
        
        # Compute IR spectra
        acvx = acovf(dlst[:, 0, 0], fft=True)
        acvy = acovf(dlst[:, 0, 1], fft=True)
        acvz = acovf(dlst[:, 0, 2], fft=True)
        acv = acvx + acvy + acvz
        
        acv = acv*np.blackman(Nframes)
        spec = np.abs(np.fft.rfftn(acv))*np.tanh(const*freq/2.)
        
        # Save spectra
        np.save(ifile, spec)
        np.save(ffile, freq)
        
    else:
        
        spec = np.load(ifile)
        freq = np.load(ffile)
    
    # Apply moving average
    Nave = int(avgfreq/(freq[1] - freq[0]))
    avgspec = moving_average(spec, Nave)
    
    # Load experimental result
    if rhostar[iv] in rho_exp:
        exp_avail = True
        exp_spec = loadmat(exp_fspc[iexp])
        exp_spec = exp_spec[exp_kspc[iexp]].reshape(-1)
        iexp += 1
        exp_freq = loadmat(exp_ffrq)[exp_kfrq].reshape(-1)
        
    else:
        exp_avail = False
    
    # Select ranges
    select3 = np.logical_and(
        freq > rnge3[0], freq < rnge3[1])
    if exp_avail:
        selectexp = np.logical_and(
            exp_freq > rnge3[0], exp_freq < rnge3[1])
    
    # Overlap asymmetric band with experiment
    if exp_avail:
        
        # Prepare spectra interpolation
        spec1 = avgspec/np.max(avgspec[select3])
        fspec1 = interp1d(freq, spec1, kind='cubic')
        spec2 = exp_spec/np.max(exp_spec[selectexp])
        
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
        opt_shift[iv] = result.x
        
    else:
        
        opt_shift[iv] = np.nan
   
# Assign average shift to pure computational spectra
avg_shift = np.mean(opt_shift[~np.isnan(opt_shift)])
opt_shift[np.isnan(opt_shift)] = avg_shift

# Reset experimental result counter
iexp = 0

for iv, Vmi in enumerate(Vm):
    
    workdir = os.path.join(
        maindir, 
        'T{:d}'.format(int(T)), 
        'V{:d}_{:d}'.format(int(Vmi), ismpl))
    
    # System tag
    tag = '{:d}_{:d}'.format(int(T), int(Vmi))
    
    # IR spectra file
    ifile = os.path.join(
        workdir, 'irspc_{:s}.npy'.format(tag))
    ffile = os.path.join(
        workdir, 'irfrq_{:s}.npy'.format(tag))
    
    # Load results
    spec = np.load(ifile)
    freq = np.load(ffile)
    
    # Apply moving average
    Nave = int(avgfreq/(freq[1] - freq[0]))
    avgspec = moving_average(spec, Nave)
    
    # Load experimental result
    if rhostar[iv] in rho_exp or rhostar[iv]==1.55:
        exp_avail = True
        exp_spec = loadmat(exp_fspc[iexp])
        exp_spec = exp_spec[exp_kspc[iexp]].reshape(-1)
        iexp += 1
        exp_freq = loadmat(exp_ffrq)[exp_kfrq].reshape(-1)
        
    else:
        exp_avail = False
    
    # Scale avgspec
    select = np.logical_and(
        freq > rnge1[0], freq < rnge3[1])
    avgspec /= np.max(avgspec[select])
    
    # Select ranges
    select1 = np.logical_and(
        freq > rnge1[0], freq < rnge1[1])
    select2 = np.logical_and(
        freq > rnge2[0], freq < rnge2[1])
    select21 = np.logical_and(
        freq > rnge21[0], freq < rnge21[1])
    select22 = np.logical_and(
        freq > rnge22[0], freq < rnge22[1])
    select3 = np.logical_and(
        freq > rnge3[0], freq < rnge3[1])
    if exp_avail:
        selectexp = np.logical_and(
            exp_freq > rnge3[0], exp_freq < rnge3[1])
    
    # Pressure ratio
    p = M/Vmi/rhocrit
    
    # Plot
    label = r'{:.2f} [{:.2f}]'.format(CM[iv], p)
    
    # Bending mode
    overlap = 1.5
    scale1 = 1./np.max(avgspec[select1])
    axs1.plot(
        freq[select1], overlap*avgspec[select1]*scale1 + iv, '-', 
        color=color_scheme[iv], label=label)
    
    ## Symmetric stretch
    #scale2 = 1./np.max(avgspec[select2])
    #axs2.plot(
        #freq[select2], overlap*avgspec[select2]*scale2 + iv, '-', 
        #color=color_scheme[iv], label=label)
        
    # Hot band
    overlap_hotband = 0.6
    scale21 = 1./np.max(avgspec[select21])*overlap_hotband/overlap
    axs2.plot(
        freq[select21], overlap*avgspec[select21]*scale21 + iv, '-', 
        color=color_scheme[iv], label=label)
    
    # Symmetric stretch
    scale22 = 1./np.max(avgspec[select22])
    axs2.plot(
        freq[select22], overlap*avgspec[select22]*scale22 + iv, '-', 
        color=color_scheme[iv])
    
    # Asymmetric stretch
    scale3 = 1./np.max(avgspec[select3])
    axs3.plot(
        freq[select3] + opt_shift[iv], overlap*avgspec[select3]*scale3 + iv, '-', 
        color=color_scheme[iv], label=label)
    
    #ilocmax = scipy.signal.argrelmax(avgspec[select3], order=4)
    #print(Vmi, 'Comp, freq_max', (freq[select3] + opt_shift[iv])[ilocmax])
    #ilocmax = scipy.signal.argrelmax(exp_spec[selectexp], order=4)
    #print(Vmi, 'Comp, freq_max', exp_freq[selectexp][ilocmax])
    
    
    # Add scaling factors
    scale_dict = {
        'family' : 'monospace',
        'weight' : 'light',
        'size'   : TINY_SIZE}
    shift_dict = {
        'family' : 'monospace',
        'weight' : 'bold',
        'size'   : TINY_SIZE}
    axs1.text(
        rnge1[0] + (rnge1[1] - rnge1[0])*0.01, 
        iv + 0.5, 'x{:.0f}'.format(scale1),
        fontdict=scale_dict)
    #axs2.text(
        #rnge2[0] + (rnge2[1] - rnge2[0])*0.01, 
        #iv + 0.5, 'x{:.0f}'.format(scale2),
        #fontdict=scale_dict)
    axs2.text(
        rnge21[0] + (rnge21[1] - rnge21[0])*0.01, 
        iv + 0.7, 'x{:.0f}'.format(scale21),
        fontdict=scale_dict)
    axs2.text(
        rnge22[0] + (rnge22[1] - rnge22[0])*0.01, 
        iv + 0.7, 'x{:.0f}'.format(scale22),
        fontdict=scale_dict)
    axs3.text(
        rnge3[0] + (rnge3[1] - rnge3[0])*0.01, 
        iv + 0.5, 'x{:.0f}'.format(scale3),
        fontdict=scale_dict)
    axs3.text(
        rnge3[0] + (rnge3[1] - rnge3[0])*0.01, 
        iv + 0.1, r'$\omega${:+.0f}'.format(opt_shift[iv]),
        fontdict=shift_dict)
    
    # Add system state
    state_dict = {
        'style'  : 'normal',
        'weight' : 'bold',
        'size'   : XSMALL_SIZE}
    axs3.text(
        rnge3[0] + (rnge3[1] - rnge3[0])*0.99, 
        iv + 0.1, '{:s}'.format(slv_states[iv]),
        va='bottom', ha='right',
        fontdict=state_dict)
    
    # Plot Experimental spectra
    if exp_avail:
        scaleexp = 1./np.max(exp_spec[selectexp])
        axs3.plot(
            exp_freq[selectexp], 
            overlap*exp_spec[selectexp]*scaleexp + iv, '--', 
            color='black')#color_scheme[iv])
        
# Axis range
axs1.set_xlim(rnge1)
axs2.set_xlim(rnge2)
axs3.set_xlim(rnge3)
axs1.set_ylim([0, len(Vm) - 1 + overlap])
axs2.set_ylim([0, len(Vm) - 1 + overlap])
axs3.set_ylim([0, len(Vm) - 1 + overlap])

# Vertical separator
axs2.plot([rnge21[-1], rnge21[-1]], [0, len(Vm) - 1 + overlap], '--k')

# Axis labels
#axs1.set_xticks([])
axs2.set_xticks([1200, 1250, 1300, 1350])
axs3.set_xticks([2150, 2200, 2250, 2300])
axs1.set_yticks([ii for ii in range(len(Vm))])
axs2.set_yticks([ii for ii in range(len(Vm))])
axs3.set_yticks([ii for ii in range(len(Vm))])
axs1.set_yticklabels([])
axs2.set_yticklabels([])
axs3.set_yticklabels([])

axs2.set_xlabel(r'Frequency $\omega$ (cm$^{-1}$)', fontweight='bold')
axs2.get_xaxis().set_label_coords(1.1, -0.1)

#axs1.set_title(r'$\nu_\mathrm{\delta}$', fontweight='bold')
#axs2.set_title(r'$\nu_\mathrm{s}$', fontweight='bold')
#axs3.set_title(r'$\nu_\mathrm{as}$', fontweight='bold')


tbox = TextArea(figlabels[0], textprops=dict(color='k', fontsize=BIGGER_SIZE))
anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.02, 0.99),
    bbox_transform=axs1.transAxes, borderpad=0.)
axs1.add_artist(anchored_tbox)

tbox = TextArea(figlabels[1], textprops=dict(color='k', fontsize=BIGGER_SIZE))
anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.02, 0.99),
    bbox_transform=axs2.transAxes, borderpad=0.)
axs2.add_artist(anchored_tbox)

tbox = TextArea(figlabels[2], textprops=dict(color='k', fontsize=BIGGER_SIZE))
anchored_tbox = AnchoredOffsetbox(
    loc='upper left', child=tbox, pad=0., frameon=False,
    bbox_to_anchor=(0.02, 0.99),
    bbox_transform=axs3.transAxes, borderpad=0.)
axs3.add_artist(anchored_tbox)



handles, labels = plt.gca().get_legend_handles_labels()
order = list(range(len(Vm)))[::-1]
axs3.legend(
    [handles[idx] for idx in order], [labels[idx] for idx in order],
    loc=(1.05, 0.05),  framealpha=1.0,
    title=r'$c$ (M) [$\rho*$]'.format(sys_tag[sys_solv]),
    title_fontsize=MEDIUM_SIZE)


plt.savefig(
    os.path.join(
        maindir, rsltdir, 'SI_Paper_IR_{:s}.png'.format(sys_solv)),
    format='png', dpi=dpi)
plt.close()
#plt.show()
