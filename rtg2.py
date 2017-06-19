import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import RectBivariateSpline
import sys

z_HM12 = np.loadtxt('z_HM12.txt')
data_HM12 = np.loadtxt('hm12.txt')
w_HM12 = data_HM12[:,0]
e_HM12 = data_HM12[:,1:]

# kx=1 and ky=1 required here to get good interpolation at the Lyman
# break.  Also remember that you will most likely need the
# 'grid=False' option while invoking emissivity_HM12.
emissivity_HM12 = RectBivariateSpline(w_HM12, z_HM12, e_HM12, kx=1, ky=1)

yrbys = 3.154e7
cmbympc = 3.24077928965e-25
c = 2.998e10*yrbys*cmbympc # Mpc/yr 

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

    n = np.linspace(np.log(N_HI_min), np.log(N_HI_max), num=100)

    return np.trapz(integrand(n), x=n)

def draw_j(j, w):

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'$j_\nu$ [$10^{-22}$ erg s$^{-1}$ Hz$^{-1}$ sr$^{-1}$ cm$^{-2}$]')
    ax.set_xlabel(r'$\lambda$')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1.0e-7, 1.0e3)
    ax.set_xlim(5.0, 4.0e3)
    plt.title('$z=6$')
    
    ax.plot(w, j/1.0e-22, lw=2, c='k')
    
    plt.savefig('j_z6.pdf',bbox_inches='tight')
    plt.close('all')

    return

wmin = 5.0
wmax = 5.0e3
ws = np.logspace(0.0, 4.0, num=1000)
nu = 2.998e18/ws 
hplanck = 6.626069e-27 # erg s

# numax=1.0e18
# nu0 = 3.288e15 # Hz; corresponds to 912 Ang
# nu = np.logspace(np.log10(nu0), 18.0, num=1000)

j = np.zeros_like(nu)

zmax = 7.0
zmin = 0.0
dz = 0.5
n = (zmax-zmin)/dz+1
zs = np.linspace(7, 0, num=n)

for z in zs:

    if z < zmax:
        vfactor = ((1.0+z)/(1.0+z-dz))**3 
        j = j/vfactor 

    # grid=False ensures that we get a flat array. 
    e = emissivity_HM12(ws*(1.0+z), z*np.ones_like(ws), grid=False)
    j = j + e*c*np.abs(dtdz(z))*dz/(4.0*np.pi)*(1.0+z)**3 

    t = np.array([tau_eff(x, z) for x in nu])
    # j = j*np.exp(-t*dz)

    j = j*(cmbympc**2) # erg s^-1 Hz^-1 cm^-2 sr^-1 

    if z == 6.0:
        draw_j(j, ws*(1.0+z))
    
    # print z, ws[1], nu[1], j[1]
    
    # n = 4.0*np.pi*j/(hplanck*nu)
    # s = np.array([sigma_HI(x) for x in nu])

    # There is a minus sign because nu = c/lambda is a decreasing
    # array so dnu is negative.
    # g = -np.trapz(n*s, x=nu) # s^-1 
    
    # print z, g 
    
