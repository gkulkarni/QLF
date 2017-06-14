import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
from scipy.integrate import quad

def emissivity_HM12(nu, z):

    # Haardt and Madau 2012 Equation (37) 

    nu0 = 3.288e15 # Hz; corresponds to 912 Ang
    
    e = 10.0**24.6 * (1.0+z)**4.68 * np.exp(-0.28*z) / (np.exp(1.77*z)+26.3)
    e = e*(nu/nu0)**-0.61 

    return e # erg s^-1 Hz^-1 Mpc^-3

def draw_emissivity():

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\epsilon_{912}$ [erg s$^{-1}$ Hz$^{-1}$ cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7.)

    ax.set_yscale('log')
    ax.set_ylim(1.0e23, 1.0e26)

    z = np.linspace(0, 7)

    nu0 = 3.288e15 # Hz; corresponds to 912 Ang
    e_HM12 = emissivity_HM12(nu0, z)
    ax.plot(z, e_HM12, lw=2, c='dodgerblue', label=r'$\nu=\nu_0$')

    e_HM12 = emissivity_HM12(10.0*nu0, z)
    ax.plot(z, e_HM12, lw=2, c='tomato', label=r'$\nu=10\nu_0$')
    
    plt.legend(loc='upper right', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01, scatterpoints=1)
    
    plt.savefig('emissivity.pdf',bbox_inches='tight')
    plt.close('all')

    return

# draw_emissivity()

yrbys = 3.154e7
cmbympc = 3.24077928965e-25
c = 2.998e10*yrbys*cmbympc # Mpc/yr 

numax=1.0e18
nu0 = 3.288e15 # Hz; corresponds to 912 Ang

nu = np.logspace(np.log10(nu0), 18.0, num=1000)
j = np.zeros_like(nu)
zmax = 7.0
zmin = 0.0
dz = 0.5
n = (zmax-zmin)/dz+1
zs = np.linspace(7,0,num=n)

omega_lambda = 0.7
omega_nr = 0.3
h = 0.7

def H(z):

    H0 = 1.023e-10*h
    hubp = H0*np.sqrt(omega_nr*(1+z)**3+omega_lambda) # yr^-1

    return hubp

def dzdt(z):

    return -(1+z)*H(z) # yr^-1

def dtdz(z):

    return 1/dzdt(z) # yr 


def f(N_HI, z):

    """
    HI column density distribution.  This parameterisation, and the
    best-fit values of the parameters, is taken from Becker and Bolton
    2013 (MNRAS 436 1023), Equation (5).

    """

    A = 0.93
    beta_N = 1.33
    beta_z = 1.92
    N_LL = 10.0**17.2 # cm^-2 

    return (A/N_LL) * ((N_HI/N_LL)**(-beta_N)) * (((1.0+z)/4.5)**beta_z) # cm^2 

def sigma_HI(nu):

    """Calculate the HI ionization cross-section.  

    This parameterisation is taken from Osterbrock and Ferland 2006
    (Sausalito, California: University Science Books), Equation (2.4).

    """
    
    nu0 = 3.288e15 # threshold freq for H I ionization; s^-1 (Hz)        
    a0 = 6.3e-18 # cm^2

    if nu < nu0:
        return 0.0
    elif nu/nu0-1.0 == 0.0:
        return a0*(nu0/nu)**4
    else:
        eps = np.sqrt(nu/nu0-1.0)
        return (a0 * (nu0/nu)**4 * np.exp(4.0-4.0*np.arctan(eps)/eps) /
                (1.0-np.exp(-2.0*np.pi/eps)))

def draw_sigma():

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\sigma$')
    ax.set_xlabel(r'$\nu$')

    plt.xlim(0,6)
    plt.ylim(0,8)

    n = np.logspace(14, 18, num=500)
    s_Osterbrock = [sigma_HI(x)/1.0e-18 for x in n]
    s_HG97 = sigma_HI_HG97(n)/1.0e-18
    
    ax.plot(n/1.0e16, s_Osterbrock, lw=4, c='forestgreen', label=r'Osterbrock')
    ax.plot(n/1.0e16, s_HG97, lw=2, c='royalblue', label=r'Gnedin')
    
    plt.legend(loc='upper right', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01, scatterpoints=1)
    
    plt.savefig('sigma.pdf',bbox_inches='tight')
    plt.close('all')

    return

def tau_eff(nu, z):

    """Calculate the effective opacity between redshifts z0 and z.

    There are two magic numbers: N_HI_min, N_HI_max.  These should
    ideally be 0 and infinity, but I have chosen to avoid improper
    integrals here.

    """
    
    N_HI_min = 1.0e13
    N_HI_max = 1.0e22 
    
    def integrand(logN_HI):

        N_HI = np.exp(logN_HI)
        tau = sigma_HI(nu)*N_HI 

        return N_HI * f(N_HI, z) * (1.0-np.exp(-tau))

    n = np.linspace(np.log(N_HI_min), np.log(N_HI_max), num=1000)
    return np.trapz(n, integrand(n))

nu0 = 3.288e15
print tau_eff(nu0, 3.)
    
for z in zs:

    e = emissivity_HM12(nu, z)
    j = j + e*c*dtdz(z)*dz/(4.0*np.pi)

    t = np.array([tau_eff(x, z) for x in nu])
    print nu[1], t[1]
    
