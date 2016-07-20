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

"""
scipy.integrate.quad and dblquad use qagse.f from FORTRAN QUADPACK.
qagse is an adaptive integrator for definite integrations.  It is an
upgrage of qags.f, which, according to Wikipedia, "uses global adaptive
quadrature based on 21-point Gauss-Kronrod quadrature within each
subinterval, with acceleration by Peter Wynn's epsilon algorithm."

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
    
    return c/((1.0+z)*H(z)) # Mpc 

def vscale(z0, z):

    return (1.0+z0)**3 / (1.0+z)**3

def luminosity(M):

    return 10.0**((51.60-M)/2.5) # ergs s^-1 Hz^-1 

def emissivity_integrand(nu, z, loglf, theta, M, individual=False):

    # SED power law index is from Beta's paper.
    L = luminosity(M)

    c = 2.998e10 # cm/s 
    l = c*1.0e8/nu # Angstrom

    if individual:
        return (10.0**loglf(theta, M))*L*((l/1450.0)**0.61) # ergs s^-1 Hz^-1 cMpc^-3 mag^-1
    else:
        return (10.0**loglf(theta, M, z))*L*((l/1450.0)**0.61) # ergs s^-1 Hz^-1 cMpc^-3 mag^-1
    
        
def emissivity(nu, z, loglf, theta, individual=False):

    mlims = (-30.0, -20.0)
    m = np.linspace(mlims[0], mlims[1], num=1000)
    farr = emissivity_integrand(nu, z, loglf, theta, m, individual=individual)

    e = np.trapz(farr, m) # erg s^-1 Hz^-1 Mpc^-3
    e *= (1.0+z)**3 # erg s^-1 Hz^-1 pMpc^-3

    return e 

def j(nu0, z0, *args, **kwargs):

    zmax = 6.6
    dz = 0.1
    
    def integrand(z, *args):

        nu = nu0*(1.0+z)/(1.0+z0)
        
        return dlbydz(z)*vscale(z0,z)*emissivity(nu, z, *args)*np.exp(-tau_eff(nu0, z0, z)) # ergs/s/Mpc^2/Hz

    rs = np.arange(z0, zmax, dz)
    j = np.array([integrand(r, *args) for r in rs])
    r = np.trapz(j, x=rs)
    r /= (4.0*np.pi)  
    
    return r # ergs/s/Mpc^2/Hz/sr 

# def j(nu0, z0, *args):

#     zmax = 9.0
    
#     def integrand(z, *args):

#         nu = nu0*(1.0+z)/(1.0+z0)
        
#         return dlbydz(z)*vscale(z0,z)*emissivity(nu, z, *args)*np.exp(-tau_eff(nu0, z0, z)) # ergs/s/Mpc^2/Hz

#     r = quad(integrand, z0, zmax, args=args)
    
#     return r[0]/(4.0*np.pi) # ergs/s/Mpc^2/Hz/sr 

def gamma_HI(z, *args, **kwargs):

    numax=1.0e18

    dnu=0.1

    nu0 = 3.288e15 # Hz
    lognu_min = np.log(nu0) 
    lognu_max = np.log(numax)

    lognu = np.arange(lognu_min, lognu_max, dnu)

    def integrand(lognu, *args):

        nu = np.exp(lognu) # Hz 
        hplanck = 6.626069e-34 # Js
        cmbympc = 3.24077928965e-25 
        return nu * j(nu, z, *args) * sigma_HI(nu) * cmbympc**2 / (hplanck * 1.0e7 * nu) # s^-1 sr^-1 Hz^-1 

    g = np.array([integrand(n, *args) for n in lognu])
    r = np.trapz(g, x=lognu) # s^-1 sr^-1 
    r *= 4.0*np.pi # s^-1

    # print z, r 

    return r # s^-1 

def plot_gamma(composite, zlims=(2.0,6.5), dirname=''):

    zmin, zmax = zlims 
    z = np.linspace(zmin, zmax, num=10) 

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', which='major', length=4, width=1)
    ax.tick_params('both', which='minor', length=2, width=1)

    bf = composite.samples.mean(axis=0)
    g = np.array([gamma_HI(rs, composite.log10phi, bf) for rs in z])
    g = np.log10(g)+12.0
    ax.plot(z, g, color='k', zorder=2)

    ax.set_ylabel(r'$\log_{10}(\Gamma_\mathrm{HI}/10^{-12} \mathrm{s}^{-1})$')
    ax.set_xlabel('$z$')
    ax.set_xlim(2.,6.5)
    ax.set_ylim(-2.,1.)
    ax.set_xticks((2,3,4,5,6))
    
    zm, gm, gm_up, gm_low = np.loadtxt('Data/BeckerBolton.dat',unpack=True) 

    ax.errorbar(zm, gm, ecolor='#d7191c', capsize=0,
                yerr=np.vstack((gm_up, abs(gm_low))),
                fmt='o', zorder=3, mfc='#d7191c', mec='#d7191c',
                mew=1, ms=5, label='Becker and Bolton 2013')

    zm, gm, gm_sigma = np.loadtxt('Data/calverley.dat',unpack=True) 
    gm += 12.0 
    ax.errorbar(zm, gm, ecolor='#fdae61', capsize=0,
                yerr=gm_sigma, fmt='o', zorder=4, mfc='#fdae61',
                mec='#fdae61', mew=1, ms=5, label='Calverley et al.~2011')

    plt.legend(loc='lower left',fontsize=12,handlelength=3,frameon=False,framealpha=0.0,
            labelspacing=.1,handletextpad=0.4,borderpad=0.2,numpoints=1)
    
    plt.savefig('gammapirt.pdf',bbox_inches='tight')

    return


