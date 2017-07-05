import numpy as np 
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt

c_angPerSec = 2.998e18

def luminosity(M):
    return 10.0**((51.60-M)/2.5) # ergs s^-1 Hz^-1 

def fnu(nu, z):
    
    # SED power law index is from Beta's paper.
    M = -22.0
    L = luminosity(M)
    w = c_angPerSec / nu
    nu_912 = c_angPerSec / 912.0
    nu_1450 = c_angPerSec / 1450.0

    a = L 
    b = a * (912.0/1450.0)**0.61
    
    if w > 912.0:
        e = a*(w/1450.0)**0.61
    else:
        e = b*(w/912.0)**1.57
    
    return e 

vfnu = np.vectorize(fnu)

def plot_qso_emissivity():

    """Plot HM12 qso emissivity. 
    
    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'emissivity [$10^{39}$~erg s$^{-1}$ Hz$^{-1}$ cMpc$^{-3}$]')
    ax.set_xlabel(r'$\nu$~[Hz]')

    ax.set_yscale('log')
    ax.set_xscale('log')
    
    nu = np.logspace(13.0, 17.0, num=10000)
    e = vfnu(nu, 1.0)
    
    ax.plot(nu, nu*e, lw=2, c='k')

    ax.axvline(c_angPerSec/912.0, lw=1, c='k', dashes=[7,2])
    
    plt.savefig('e_qso.pdf', bbox_inches='tight')
    plt.close('all')

    return

plot_qso_emissivity()
