import numpy as np
from scipy.integrate import dblquad, quad 
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
from scipy import interpolate

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

    """
    HI ionization cross-section.  See Osterbrock's book. 

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

def tau_eff(nu0, z0, z):

    def integrand(logN_HI, z):

        N_HI = np.exp(logN_HI)
        nu = nu0*(1.0+z)/(1.0+z0) 
        tau = sigma_HI(nu)*N_HI 

        i = N_HI * f(N_HI, z) * (1.0-np.exp(-tau))

        return i 

    r = dblquad(integrand, z0, z, lambda x: np.log(1.0e13), lambda x: np.log(1.0e22))
    return r[0]

def H(z):

    h = 0.678
    onr = 0.308
    ol = 0.692 

    H0 = 1.023e-10*h # yr^-1 

    return H0 * np.sqrt(onr*(1.0+z)**3 + ol) # yr^-1 

def dlbydz(z):

    yrbys = 3.154e7
    cmbympc = 3.24077928965e-25
    c = 2.998e10*yrbys*cmbympc # Mpc/yr 
    
    return c/(1.0+z)/H(z) # Mpc 

def vscale(z0, z):

    return (1.0+z0)**3 / (1.0+z)**3


data = np.loadtxt('../21cm/hm12/emissivity.dat')
redshifts = data[0,:-1]
wavelengths = data[1:,0] # Angstrom 
emissivities = data[1:,1:] # ergs/s/Mpc^3/Hz
emissivities_at_z = interpolate.interp1d(redshifts, emissivities, axis=1) # ergs/s/Mpc^3/Hz

def emissivity(nu, z):

    c = 2.998e10 # cm/s 
    l = c*1.0e8/nu # Angstrom
    
    return np.interp(l, wavelengths, emissivities_at_z(z)) # ergs/s/Mpc^3/Hz

def j(nu0, z0, zmax, dz=0.1):

    def integrand(z):

        nu = nu0*(1.0+z)/(1.0+z0)
        return dlbydz(z)*vscale(z0,z)*emissivity(nu, z)*np.exp(-tau_eff(nu0, z0, z)) # ergs/s/Mpc^2/Hz

    rs = np.arange(2.0, zmax, dz)
    j = np.array([integrand(r) for r in rs])
    r = np.trapz(j, x=rs)
    r /= (4.0*np.pi)  
    
    return r # ergs/s/Mpc^2/Hz/sr 

x = j(3.29e15, 2.0, 6.0, dz=0.1)
print x 


