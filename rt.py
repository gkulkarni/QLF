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

def j(nu0, z0, zmax=6.0, dz=0.1):

    def integrand(z):

        nu = nu0*(1.0+z)/(1.0+z0)
        return dlbydz(z)*vscale(z0,z)*emissivity(nu, z)*np.exp(-tau_eff(nu0, z0, z)) # ergs/s/Mpc^2/Hz

    rs = np.arange(2.0, zmax, dz)
    j = np.array([integrand(r) for r in rs])
    r = np.trapz(j, x=rs)
    r /= (4.0*np.pi)  
    
    return r # ergs/s/Mpc^2/Hz/sr 

# x = j(3.29e15, 2.0, dz=0.1)
# print x 

def gamma_HI(z, numax=1.0e18, dnu=0.1):

    nu0 = 3.288e15 # Hz
    lognu_min = np.log(nu0) 
    lognu_max = np.log(numax)

    lognu = np.arange(lognu_min, lognu_max, dnu)

    def integrand(lognu):

        nu = np.exp(lognu) # Hz 
        hplanck = 6.626069e-34 # Js
        cmbympc = 3.24077928965e-25 
        return nu * j(nu, z) * sigma_HI(nu) * cmbympc**2 / (hplanck * 1.0e7 * nu) # s^-1 sr^-1 Hz^-1 

    g = np.array([integrand(n) for n in lognu])
    r = np.trapz(g, x=lognu) # s^-1 sr^-1 
    r *= 4.0*np.pi # s^-1 

    return r # s^-1 

# import time
# t1 = time.time()
# x = gamma_HI(2.0)
# t2 = time.time()
# print x
# print t2-t1

def gamma_HI_alt(z, numax=1.0e18):

    nu0 = 3.288e15 # Hz
    lognu_min = np.log(nu0) 
    lognu_max = np.log(numax)

    def integrand(lognu):

        nu = np.exp(lognu) # Hz 
        hplanck = 6.626069e-34 # Js
        cmbympc = 3.24077928965e-25 
        return nu * j(nu, z) * sigma_HI(nu) * cmbympc**2 / (hplanck * 1.0e7 * nu) # s^-1 sr^-1 

    r = quad(integrand, lognu_min, lognu_max)
    return r[0]*4.0*np.pi

# import time
# t1 = time.time()
# x = gamma_HI_alt(2.0)
# t2 = time.time()
# print x
# print t2-t1

def plot_fHI():

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)

    logNHI = np.arange(10.0, 22.0, 0.1)

    fNHI = np.array([f(10.0**n, 2.0) for n in logNHI])
    logfNHI = np.log10(fNHI)
    plt.plot(logNHI, logfNHI, lw=2, c='k', label='$z=2$')

    fNHI = np.array([f(10.0**n, 3.5) for n in logNHI])
    logfNHI = np.log10(fNHI)
    plt.plot(logNHI, logfNHI, lw=2, c='r', label='$z=3.5$')
    
    fNHI = np.array([f(10.0**n, 5.0) for n in logNHI])
    logfNHI = np.log10(fNHI)
    plt.plot(logNHI, logfNHI, lw=2, c='b', label='$z=5$')

    plt.xlim(11, 21.5)
    plt.ylim(-25, -7)
    plt.xlabel(r'$\log_{10}\left[N_\mathrm{HI}/\mathrm{cm}^{-2}\right]$')
    plt.ylabel(r'$\log_{10}\left[f(N_\mathrm{HI},z)/\mathrm{cm}^2\right]$')
    
    plt.legend(loc='lower left', fontsize=14, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.2, scatterpoints=1)

    plt.savefig('fhi.pdf',bbox_inches='tight')

    return

def tau_integrand(log10N_HI, z):

    N_HI = 10.0**log10N_HI # Different from above!  To match HM12.
    nu0 = 3.288e15 # Hz
    tau = sigma_HI(nu0)*N_HI 

    i = N_HI * f(N_HI, z) * (1.0-np.exp(-tau))

    return i 

def plot_tau_integrand():

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)

    logNHI = np.arange(10.0, 22.0, 0.1)

    fNHI = np.array([tau_integrand(n, 2.0) for n in logNHI])
    logfNHI = np.log10(fNHI)
    plt.plot(logNHI, logfNHI, lw=2, c='k', label='$z=2$')

    fNHI = np.array([tau_integrand(n, 3.5) for n in logNHI])
    logfNHI = np.log10(fNHI)
    plt.plot(logNHI, logfNHI, lw=2, c='r', label='$z=3.5$')
    
    fNHI = np.array([tau_integrand(n, 5.0) for n in logNHI])
    logfNHI = np.log10(fNHI)
    plt.plot(logNHI, logfNHI, lw=2, c='b', label='$z=5$')

    plt.xlim(11, 21.5)
    # plt.ylim(-25, -7)
    plt.xlabel(r'$\log_{10}\left[N_\mathrm{HI}/\mathrm{cm}^{-2}\right]$')
    plt.ylabel(r'$\log_{10}\left[N_\mathrm{HI}f(N_\mathrm{HI},z)(1-e^{-N_\mathrm{HI}\sigma_{912}})\right]$')
    
    plt.legend(loc='lower left', fontsize=14, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.2, scatterpoints=1)

    plt.savefig('tint.pdf',bbox_inches='tight')

    return
    
def plot_tau_vs_nu():

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)

    nu0 = 3.288e15 # Hz
    numax = 1.0e18 # Hz 
    lognu_min = np.log10(nu0)
    lognu_max = np.log10(numax)
    lognu = np.arange(lognu_min, lognu_max, 0.1)
    
    tau = np.array([tau_eff(10.0**n, 2.0, 6.0) for n in lognu])
    plt.plot(lognu, tau, lw=2, c='k', label='$z=2$')

    tau = np.array([tau_eff(10.0**n, 3.5, 6.0) for n in lognu])
    plt.plot(lognu, tau, lw=2, c='r', label='$z=3.5$')

    tau = np.array([tau_eff(10.0**n, 5.0, 6.0) for n in lognu])
    plt.plot(lognu, tau, lw=2, c='b', label='$z=5$')

    # plt.xlim(11, 21.5)
    # plt.ylim(-25, -7)
    plt.xlabel(r'$\log_{10}\left[\nu/\mathrm{Hz}\right]$')
    plt.ylabel(r'$\tau(\nu, z, 6.0)$')

    plt.yscale('log')
    
    plt.legend(loc='upper right', fontsize=14, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.2, scatterpoints=1)

    plt.savefig('taunu.pdf',bbox_inches='tight')

def plot_tau_vs_z():

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)

    zs = np.arange(2.0, 6.0, 0.1)
    nu0 = 3.288e15 # Hz

    tau = np.array([tau_eff(nu0, r, 6.0) for r in zs])
    plt.plot(zs, tau, lw=2, c='k', label=r'$\nu_{912}$')

    tau = np.array([tau_eff(nu0*10.0, r, 6.0) for r in zs])
    plt.plot(zs, tau, lw=2, c='r', label=r'$10\nu_{912}$')

    tau = np.array([tau_eff(nu0*100.0, r, 6.0) for r in zs])
    plt.plot(zs, tau, lw=2, c='b', label=r'$100\nu_{912}$')
    
    plt.xlabel(r'$z$')
    plt.ylabel(r'$\tau(\nu, z, 6.0)$')

    plt.yscale('log')
    
    plt.legend(loc='upper right', fontsize=14, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.2, scatterpoints=1)

    plt.savefig('tauz.pdf',bbox_inches='tight')
    

plot_tau_vs_z() 
