import numpy as np
from scipy.integrate import dblquad

""" Functions for calculating the hydrogen photoionisation rate.

Redshifting of radiation is taken into account.  Thus, we are mainly
evaluating Equation (2) of Haardt and Madau 2012 (ApJ 746 125).  The
main function here is gamma_HI(z, *args, *kwargs).

Bugs: 

-- Helium is ignored.  
-- There are several magic numbers.  See function docstrings.
-- Code is too slow.   
-- Module filename could be better.

A note on the integrator used: scipy.integrate.dblquad dblquad uses
qagse.f from FORTRAN QUADPACK.  qagse is an adaptive integrator for
definite integrations.  It is an upgrage of qags.f, which, according
to Wikipedia, "uses global adaptive quadrature based on 21-point
Gauss-Kronrod quadrature within each subinterval, with acceleration by
Peter Wynn's epsilon algorithm."

"""

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

def tau_eff(nu0, z0, z):

    """Calculate the effective opacity between redshifts z0 and z.

    There are two magic numbers: N_HI_min, N_HI_max.  These should
    ideally be 0 and infinity, but I have chosen to avoid improper
    integrals here.

    """
    
    N_HI_min = 1.0e13
    N_HI_max = 1.0e22 
    
    def integrand(logN_HI, z):

        N_HI = np.exp(logN_HI)
        nu = nu0*(1.0+z)/(1.0+z0) 
        tau = sigma_HI(nu)*N_HI 

        i = N_HI * f(N_HI, z) * (1.0-np.exp(-tau))

        return i 

    r = dblquad(integrand, z0, z, lambda x: np.log(N_HI_min),
                lambda x: np.log(N_HI_max))
    
    return r[0] # dimensionless 

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
    
    return c/((1.0+z)*H(z)) # Mpc 

def vscale(z0, z):

    return (1.0+z0)**3 / (1.0+z)**3

def luminosity(M):

    return 10.0**((51.60-M)/2.5) # ergs s^-1 Hz^-1 

def emissivity_integrand(nu, z, loglf, theta, M, individual=False):

    L = luminosity(M)

    c = 2.998e10 # cm s^-1  
    l = c*1.0e8/nu # Angstrom

    # Far UV spectral index (0.61) is from Lusso et al. 2015 (MNRAS
    # 449 4204).
    if individual:
        i = (10.0**loglf(theta, M))*L*((l/1450.0)**0.61) 
    else:
        i = (10.0**loglf(theta, M, z))*L*((l/1450.0)**0.61) 

    return i # ergs s^-1 Hz^-1 cMpc^-3 mag^-1
    
        
def emissivity(nu, z, loglf, theta, individual=False):

    """Calculate the emissivity at frequency nu and redshift z.

    Three magic numbers: mlims[0], mlims[1], and num.  See
    emissivity_integrand() above for how the arguments are used.

    """

    mlims = (-30.0, -20.0)
    m = np.linspace(mlims[0], mlims[1], num=1000)
    farr = emissivity_integrand(nu, z, loglf, theta, m, individual=individual)

    e = np.trapz(farr, m) # erg s^-1 Hz^-1 cMpc^-3
    e *= (1.0+z)**3 # erg s^-1 Hz^-1 pMpc^-3

    return e # erg s^-1 Hz^-1 pMpc^-3

def j(nu0, z0, *args, **kwargs):

    """Calculate the specific intensity at frequency nu0 and redshift z0.

    Two magic numbers: zmax and dz, which can affect the result.  args
    and kwargs are passed on to the emissivity() function above.

    """

    zmax = 6.6
    dz = 0.1
    
    def integrand(z, *args, **kwargs):

        nu = nu0*(1.0+z)/(1.0+z0)
        
        return (dlbydz(z)*vscale(z0,z)*emissivity(nu, z, *args, **kwargs)
                *np.exp(-tau_eff(nu0, z0, z))) # erg s^-1 Mpc^-2 Hz^-1

    rs = np.arange(z0, zmax, dz)
    # j has units of erg s^-1 Mpc^-2 Hz^-1.
    j = np.array([integrand(r, *args, **kwargs) for r in rs])
    
    r = np.trapz(j, x=rs) # erg s^-1 Mpc^-2 Hz^-1 
    r /= (4.0*np.pi) # erg s^-1 Mpc^-2 Hz^-1 sr^-1  
    
    return r # erg s^-1 Mpc^-2 Hz^-1 sr^-1  

def gamma_HI(z, *args, **kwargs):

    """Calculate the HI photoionisation rate.

    There are two magic numbers: numax and dnu, which can affect the
    convergence of the integral.  The integral is done over
    log(frequency) instead of the frequency.  The positional and
    keyword arguments are for the emissivity() function above. 

    """

    numax=1.0e18
    nu0 = 3.288e15 # Hz; corresponds to 912 Ang
    
    lognu_min = np.log(nu0) 
    lognu_max = np.log(numax)
    dnu=0.1
    lognu = np.arange(lognu_min, lognu_max, dnu)

    def integrand(lognu, *args, **kwargs):

        nu = np.exp(lognu) # Hz 
        hplanck = 6.626069e-34 # Js
        cmbympc = 3.24077928965e-25
        # Note the additional factor of nu because the integral is
        # going to be over log(nu).  This also reflects in the units.
        return (nu * j(nu, z, *args, **kwargs) * sigma_HI(nu) *
                cmbympc**2 / (hplanck * 1.0e7 * nu)) # s^-1 sr^-1 Hz^-1 Hz 

    g = np.array([integrand(n, *args, **kwargs) for n in lognu]) # s^-1 sr^-1 
    r = np.trapz(g, x=lognu) # s^-1 sr^-1 
    r *= 4.0*np.pi # s^-1

    return r # s^-1 


