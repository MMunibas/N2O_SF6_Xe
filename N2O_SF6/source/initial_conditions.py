# Basics
import os
import sys
import numpy as np

# Miscellaneous
import ase.units as units


def get_velocities(j=None, nu_as=None, nu_s=None, nu_d=None, temperature=None):

    #--------------------------------
    # Parameters
    #--------------------------------

    # Temperature
    if temperature is None:
        temperature = 0.0
    
    # Rotational state
    if j is None:
        j = [0, 0, 0]
    
    # Rotation axis
    # idx = 0: rotation around smallest momentum of inertia
    # idx = 1: rotation around the middle momentum of inertia
    # idx = 2: rotation around largest momentum of inertia
    # idx = [0, 2], [0, 1, 2], ...: rotation around selected momenta
    idx = [0, 1, 2]

    # Asymmetric stretch vibrational state
    # nu = -1: no vibrational energy
    # nu =  0: vibrational ground state
    # nu >= 1: vibrational excited state
    # nu = None: thermal vibrational energy
    #nu_as = None
    
    # Symmetric stretch vibrational state
    # nu = -1: no vibrational energy
    # nu =  0: vibrational ground state
    # nu >= 1: vibrational excited state
    # nu = None: thermal vibrational energy
    #nu_s = None
    
    # Bend vibrational state
    # nu = -1: no vibrational energy
    # nu =  0: vibrational ground state
    # nu >= 1: vibrational excited state
    # nu = None: thermal vibrational energy
    #nu_d = None

    # Number of atoms
    Natoms = 3

    # Atom masses
    masses = np.array([14.0067, 14.0067, 15.9994])

    # Equilibrium atom positions
    positions = np.array([
        [0.0000000000,  0.0000000000, -1.1144822391],
        [0.0000000000,  0.0000000000,  0.0141385128],
        [0.0000000000,  0.0000000000,  1.1996346617]])

    # Normal mode and frequency (cm-1) - asymmetric stretch
    mode_as = np.array([
        [0.00000,  0.00000, -0.14078],
        [0.00000,  0.00000,  0.21594],
        [0.00000,  0.00000, -0.06579]])
    freq_as = 2179.600035
    
    # Normal mode and frequency (cm-1) - symmetric stretch
    mode_s = np.array([
        [0.00000,  0.00000, -0.16985],
        [0.00000,  0.00000, -0.04521],
        [0.00000,  0.00000,  0.18830]])
    freq_s = 1293.882828
    
    # Normal mode and frequency (cm-1) - bending mode
    mode_d = np.array([
        [ 0.00000, -0.11288,  0.00000],
        [ 0.00000,  0.22033,  0.00000],
        [ 0.00000, -0.09408,  0.00000]])
    freq_d = 617.023142

    # Boltzmann constant in Hartree / Kelvin
    kB_HaK = 3.166811563E-6
    
    #--------------------------------
    # Unit conversion
    #--------------------------------
    
    # Conversion parameter

    # Positions
    a02A = units.Bohr
    m2A = 1.e10

    # Energy
    kcalmol2Ha = units.kcal/units.mol/units.Hartree
    kcalmol2J = units.kcal/units.mol/units.J

    # Masses
    u2kg = units._amu
    u2au = units._amu/units._me

    # Time
    s2fs = 1.e15
    takma2fs = 48.88821

    # Velocities
    ms2Afs = 1e-5
    ms2au = units._me*units.Bohr*1e-10*2.0*np.pi/units._hplanck

    # Wavenumber
    rcm2Ha = units._hplanck*units._c/0.01*units.J/units.Hartree

    # Unit conversion
    positions = positions/a02A
    mode_as = mode_as/a02A
    freq_as = freq_as*rcm2Ha
    mode_s = mode_s/a02A
    freq_s = freq_s*rcm2Ha
    mode_d = mode_d/a02A
    freq_d = freq_d*rcm2Ha
    masses = masses*u2au

    #--------------------------------
    # Rotation - Preparations
    #--------------------------------

    # Convert index
    if isinstance(idx, int):
        idx = np.array([idx], dtype=int)
    else:
        idx = np.array(idx, dtype=int)

    # Convert j state
    if isinstance(j, int):
        j = np.array([j]*len(idx), dtype=int)
    else:
        j = np.array(j, dtype=int)

    # Check length
    if len(j) != len(idx):
        raise IOError("Mismatch in given j state numbers and rotation axis idx")

    # Center atoms 
    com = (np.sum(masses.reshape(-1, 1)*positions, axis=0)/np.sum(masses))
    positions = positions - com

    # Momentum and principle axis of inertia
    Icart = np.zeros([3, 3])
    Icart[0, 0] = np.sum(masses*(positions[:, 1]**2 + positions[:, 2]**2))
    Icart[1, 1] = np.sum(masses*(positions[:, 0]**2 + positions[:, 2]**2))
    Icart[2, 2] = np.sum(masses*(positions[:, 0]**2 + positions[:, 1]**2))
    Icart[0, 1] = Icart[1, 0] = np.sum(
        -masses*positions[:, 0]*positions[:, 1])
    Icart[0, 2] = Icart[2, 0] = np.sum(
        -masses*positions[:, 0]*positions[:, 2])
    Icart[1, 2] = Icart[2, 1] = np.sum(
        -masses*positions[:, 1]*positions[:, 2])

    Ip, Ib = np.linalg.eigh(Icart)
    Isort = np.argsort(Ip)
    Ip = Ip[Isort]
    Ib = Ib.T[Isort]

    # Rotational constant
    Bp = np.zeros(3)
    Bp[Ip != 0.0] = 1.0**2/(2.0*Ip[Ip != 0.0])

    # Rotational energy
    Ej = Bp*j*(j + 1)

    #print(Ip)
    #print(Ib)
    #print(Bp)
    #print(Ej)

    #--------------------------------
    # Rotation
    #--------------------------------

    # Total angular velocity
    w = np.sqrt(2.*Ej[idx]/Ip[idx])

    # Total angular momentum
    L = np.sqrt(2.*Ej[idx]*Ip[idx])

    # Atom momentum from angular velocity
    pi = (masses*np.sqrt(np.sum(positions**2, axis=1))).reshape(1, -1)*w.reshape(-1, 1)

    # Assign directions
    atom_directions = positions/np.sqrt(np.sum(positions**2, axis=1)).reshape(-1, 1)
    rotation_direction = np.array(
        [np.cross(atom_directions, Ib[idxi]) for idxi in idx])

    # Rotation momentum
    prot = pi.reshape(-1, positions.shape[0], 1)*rotation_direction

    # Rotation velocity
    vrot = prot/masses.reshape(1, -1, 1)

    # Convert from au to Angstrom / t_akma
    vrot_ms = vrot/ms2au
    vrot_akma = vrot_ms*m2A/(s2fs/takma2fs)

    # Check rotational energy
    Ej_assigned = np.sum(0.5*masses/u2au*np.sum(vrot_akma**2, axis=2), axis=1)

    # Combine velocities
    vrot_akma = np.nansum(vrot_akma, axis=0)

    print()
    print("Atomic angular velocities in Angstrom per akma time unit")
    print(vrot_akma)
    print("Respective kinetic energy (kcal / mol) and reference energy")
    for ii, idxi in enumerate(idx):
        print("EJ_{:d}(J={:d}) = {:.5f}, {:.5f}".format(
            idxi, j[ii], Ej_assigned[ii], Ej[idxi]/kcalmol2Ha))
    print()

    #--------------------------------
    # Vibration - Asymmetric Stretch
    #--------------------------------
    
    if nu_as is None:
        
        # Vibrational energy
        Enu_as = kB_HaK*temperature
        
        # Center normal mode distribution
        mode_as -= masses.reshape(-1, 1)*np.sum(mode_as, axis=0)/np.sum(masses)

        # Reference kinetic energy
        Enu_as_mode = np.sum(np.sum(mode_as**2, axis=1)/(2.*masses))

        # Momentum scaling factor 
        pfac = Enu_as/Enu_as_mode

        # Atomic vibrational momentums
        pnu_as = mode_as*np.sqrt(pfac)

        # Atomic vibrational velocities
        vnu_as = pnu_as/masses.reshape(-1, 1)

        # Convert from au to Angstrom / t_akma
        vnu_as_ms = vnu_as/ms2au
        vnu_as_akma = vnu_as_ms*m2A/(s2fs/takma2fs)

        # Check vibrational energy
        Enu_as_assigned = np.sum(0.5*masses/u2au*np.sum(vnu_as_akma**2, axis=1))
        
    elif nu_as >= 0:

        # Vibrational energy
        Enu_as = freq_as*(0.5 + nu_as)

        # Center normal mode distribution
        mode_as -= masses.reshape(-1, 1)*np.sum(mode_as, axis=0)/np.sum(masses)

        # Reference kinetic energy
        Enu_as_mode = np.sum(np.sum(mode_as**2, axis=1)/(2.*masses))

        # Momentum scaling factor 
        pfac = Enu_as/Enu_as_mode

        # Atomic vibrational momentums
        pnu_as = mode_as*np.sqrt(pfac)

        # Atomic vibrational velocities
        vnu_as = pnu_as/masses.reshape(-1, 1)

        # Convert from au to Angstrom / t_akma
        vnu_as_ms = vnu_as/ms2au
        vnu_as_akma = vnu_as_ms*m2A/(s2fs/takma2fs)

        # Check vibrational energy
        Enu_as_assigned = np.sum(0.5*masses/u2au*np.sum(vnu_as_akma**2, axis=1))
        
    else:
        
        # Zero atomic vibrational velocities
        vnu_as_akma = np.zeros_like(mode_as)
        
        # Zero vibrational energy
        Enu_as_assigned, Enu_as = 0.0, 0.0

    print()
    print(
        "Atomic asymmetric stretch vibrational velocities " 
        + "in Angstrom per akma time unit")
    print(vnu_as_akma)
    print("Respective kinetic energy (kcal / mol) and reference energy")
    print("Enu_as(nu_as={:s}) = {:.5f}, {:.5f}".format(
        str(nu_as), Enu_as_assigned, Enu_as/kcalmol2Ha))
    print()
    
    #--------------------------------
    # Vibration - Symmetric Stretch
    #--------------------------------
    
    if nu_s is None:
        
        # Vibrational energy
        Enu_s = kB_HaK*temperature
        
        # Center normal mode distribution
        mode_s -= masses.reshape(-1, 1)*np.sum(mode_s, axis=0)/np.sum(masses)

        # Reference kinetic energy
        Enu_s_mode = np.sum(np.sum(mode_s**2, axis=1)/(2.*masses))

        # Momentum scaling factor 
        pfac = Enu_s/Enu_s_mode

        # Atomic vibrational momentums
        pnu_s = mode_s*np.sqrt(pfac)

        # Atomic vibrational velocities
        vnu_s = pnu_s/masses.reshape(-1, 1)

        # Convert from au to Angstrom / t_akma
        vnu_s_ms = vnu_s/ms2au
        vnu_s_akma = vnu_s_ms*m2A/(s2fs/takma2fs)

        # Check vibrational energy
        Enu_s_assigned = np.sum(0.5*masses/u2au*np.sum(vnu_s_akma**2, axis=1))
        
    elif nu_s >= 0:

        # Vibrational energy
        Enu_s = freq_s*(0.5 + nu_s)

        # Center normal mode distribution
        mode_s -= masses.reshape(-1, 1)*np.sum(mode_s, axis=0)/np.sum(masses)

        # Reference kinetic energy
        Enu_s_mode = np.sum(np.sum(mode_s**2, axis=1)/(2.*masses))

        # Momentum scaling factor 
        pfac = Enu_s/Enu_s_mode

        # Atomic vibrational momentums
        pnu_s = mode_s*np.sqrt(pfac)

        # Atomic vibrational velocities
        vnu_s = pnu_s/masses.reshape(-1, 1)

        # Convert from au to Angstrom / t_akma
        vnu_s_ms = vnu_s/ms2au
        vnu_s_akma = vnu_s_ms*m2A/(s2fs/takma2fs)

        # Check vibrational energy
        Enu_s_assigned = np.sum(0.5*masses/u2au*np.sum(vnu_s_akma**2, axis=1))
        
    else:
        
        # Zero atomic vibrational velocities
        vnu_s_akma = np.zeros_like(mode_s)
        
        # Zero vibrational energy
        Enu_s_assigned, Enu_s = 0.0, 0.0

    print()
    print(
        "Atomic ssymmetric stretch vibrational velocities " 
        + "in Angstrom per akma time unit")
    print(vnu_s_akma)
    print("Respective kinetic energy (kcal / mol) and reference energy")
    print("Enu_s(nu_s={:s}) = {:.5f}, {:.5f}".format(
        str(nu_s), Enu_s_assigned, Enu_s/kcalmol2Ha))
    print()
    

    #--------------------------------
    # Vibration - Bending Mode
    #--------------------------------

    if nu_d is None:
        
        # Vibrational energy
        Enu_d = kB_HaK*temperature

        # Center normal mode distribution
        mode_d -= masses.reshape(-1, 1)*np.sum(mode_d, axis=0)/np.sum(masses)

        # Reference kinetic energy
        Enu_d_mode = np.sum(np.sum(mode_d**2, axis=1)/(2.*masses))

        # Momentum scaling factor 
        pfac = Enu_d/Enu_d_mode

        # Atomic vibrational momentums
        pnu_d = mode_d*np.sqrt(pfac)

        # Atomic vibrational velocities
        vnu_d = pnu_d/masses.reshape(-1, 1)

        # Convert from au to Angstrom / t_akma
        vnu_d_ms = vnu_d/ms2au
        vnu_d_akma = vnu_d_ms*m2A/(s2fs/takma2fs)

        # Check vibrational energy
        Enu_d_assigned = np.sum(0.5*masses/u2au*np.sum(vnu_d_akma**2, axis=1))
        
    elif nu_d >= 0:
        
        # Vibrational energy
        Enu_d = freq_d*(0.5 + nu_d)

        # Center normal mode distribution
        mode_d -= masses.reshape(-1, 1)*np.sum(mode_d, axis=0)/np.sum(masses)

        # Reference kinetic energy
        Enu_d_mode = np.sum(np.sum(mode_d**2, axis=1)/(2.*masses))

        # Momentum scaling factor 
        pfac = Enu_d/Enu_d_mode

        # Atomic vibrational momentums
        pnu_d = mode_d*np.sqrt(pfac)

        # Atomic vibrational velocities
        vnu_d = pnu_d/masses.reshape(-1, 1)

        # Convert from au to Angstrom / t_akma
        vnu_d_ms = vnu_d/ms2au
        vnu_d_akma = vnu_d_ms*m2A/(s2fs/takma2fs)

        # Check vibrational energy
        Enu_d_assigned = np.sum(0.5*masses/u2au*np.sum(vnu_d_akma**2, axis=1))
        
    else:
        
        # Zero atomic vibrational velocities
        vnu_d_akma = np.zeros_like(mode_as)
        
        # Zero vibrational energy
        Enu_d_assigned, Enu_d = 0.0, 0.0

    print()
    print(
        "Atomic bending vibrational velocities " 
        + "in Angstrom per akma time unit")
    print(vnu_d_akma)
    print("Respective kinetic energy (kcal / mol) and reference energy")
    print("Enu_d(nu_d={:s}) = {:.5f}, {:.5f}".format(
        str(nu_d), Enu_d_assigned, Enu_d/kcalmol2Ha))
    print()

    #--------------------------------
    # Summary
    #--------------------------------

    print()
    print(
        "Combined atomic velocities " 
        + "in Angstrom per akma time unit")
    v_akma = vrot_akma + vnu_as_akma + vnu_s_akma + vnu_d_akma
    print(v_akma)
    print()


    print("CHARMM input:")
    print()
    j_print = np.zeros(3, dtype=int)
    j_print[idx] = j
    lines = "! J_0,1,2 = ({:d}, {:d}, {:d}), ".format(*j_print)
    lines += "nu_as = {:s}, nu_s = {:s}, nu_d = {:s} ".format(
        str(nu_as), str(nu_s), str(nu_d))
    lines += "at T = {:s} K\n".format(str(temperature))
    for iatom, vatom in enumerate(v_akma):
        lines += (
            "coor set xdir {: 8f} ydir {: 8f} zdir {: 8f} ".format(*vatom)
            + "sele bynu {:d} end comp\n".format(iatom + 1)
            )
    print(lines)
    
    return v_akma
