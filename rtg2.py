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

jdata_HM12 = np.loadtxt('j_HM12.txt')
w_j_HM12 = jdata_HM12[:,0]

# Remove lines with same wavelength so that RectBivariateSpline can
# work.  Not sure why these lines occur in HM12 tables.
w_j_HM12, idx = np.unique(w_j_HM12, return_index=True)
j_HM12 = jdata_HM12[idx, 1:]

bkgintens_HM12 = RectBivariateSpline(w_j_HM12, z_HM12, j_HM12, kx=1, ky=1)

yrbys = 3.154e7
cmbympc = 3.24077928965e-25
c_mpcPerYr = 2.998e10*yrbys*cmbympc # Mpc/yr 
c_angPerSec = 2.998e18
nu0 = 3.288e15 # threshold freq for H I ionization; s^-1 (Hz)
hplanck = 6.626069e-27 # erg s

omega_lambda = 0.7
omega_nr = 0.3
h = 0.7

def qso_emissivity_hm12(nu, z):

    """HM12 qso comoving emissivity.

    This is given by equations (37) and (38) of HM12.

    """

    w = c_angPerSec/nu
    nu_912 = c_angPerSec/912.0
    nu_1300 = c_angPerSec/1300.0
    
    a = 10.0**24.6 * (1.0+z)**4.68 * np.exp(-0.28*z)/(np.exp(1.77*z)+26.3)
    b = a * (nu_1300/nu_912)**-1.57

    if w > 1300.0:
        e = b*(nu/nu_1300)**-0.44
    else:
        e = a*(nu/nu_912)**-1.57

    return e

vqso_emissivity_hm12 = np.vectorize(qso_emissivity_hm12, otypes=[np.float])

def H(z):

    H0 = 1.023e-10*h
    hubp = H0*np.sqrt(omega_nr*(1+z)**3+omega_lambda) # yr^-1

    return hubp

def dzdt(z):

    return -(1.0+z)*H(z) # yr^-1

def dtdz(z):

    return 1.0/dzdt(z) # yr 


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

def f_HM12(N_HI, z):

    """HI column density distribution from HM12.

    See their Table 1 and Section 3.

    """

    log10NHI = np.log10(N_HI)

    if z < 1.56:
        if 11.0 <= log10NHI < 15.0:
            a = 1.729816e8 
            b = 1.5
            g = 0.16
        elif 15.0 <= log10NHI < 17.5:
            a = 5.495409e15 
            b = 2.0
            g = 0.16 
        elif 17.5 <= log10NHI < 19.0:
            f17p5 = 5.495409e15 * (1+z)**0.16 * (10.0**17.5)**-2.0
            f19 = 1.279381 * (1+z)**0.16 * (10.0**19)**-1.05
            b = np.log10(f17p5/f19)/1.5
            a = f17p5 * (10.0**17.5)**b
            return a * N_HI**-b
        elif 19.0 <= log10NHI < 20.3:
            a = 1.279381
            b = 1.05 
            g = 0.16 
        elif 20.3 <= log10NHI:
            a = 2.471724e19 
            b = 2.0 
            g = 0.16 
        return a * (1+z)**g * N_HI**-b
    
    elif 1.56 <= z < 5.5:
        if 11.0 <= log10NHI < 15.0:
            a = 1.199499e7
            b = 1.5
            g = 3.0
        elif 15.0 <= log10NHI < 17.5:
            a = 3.801894e14
            b = 2.0
            g = 3.0
        elif 17.5 <= log10NHI < 19.0:
            f17p5 = 3.801894e14 * (1+z)**3.0 * (10.0**17.5)**-2.0
            f19 = 0.449780 * (1+z)**1.27 * (10.0**19)**-1.05
            b = np.log10(f17p5/f19)/1.5
            a = f17p5 * (10.0**17.5)**b
            return a * N_HI**-b
        elif 19.0 <= log10NHI < 20.3:
            a = 0.449780
            b = 1.05
            g = 1.27
        elif 20.3 <= log10NHI:
            a = 8.709636e18
            b = 2.0
            g = 1.27
        return a * (1+z)**g * N_HI**-b

    elif 5.5 <= z:
        if 11.0 <= log10NHI < 15.0:
            a = 29.512092
            b = 1.5
            g = 9.9
        elif 15.0 <= log10NHI < 17.5:
            a = 9.332543e8 
            b = 2.0
            g = 9.9
        elif 17.5 <= log10NHI < 19.0:
            f17p5 = 9.332543e8 * (1+z)**9.9 * (10.0**17.5)**-2.0
            f19 = 0.449780 * (1+z)**1.27 * (10.0**19)**-1.05
            b = np.log10(f17p5/f19)/1.5
            a = f17p5 * (10.0**17.5)**b
            return a * N_HI**-b
        elif 19.0 <= log10NHI < 20.3:
            a = 0.449780 
            b = 1.05
            g = 1.27
        elif 20.3 <= log10NHI:
            a = 8.709636e18
            b = 2.0
            g = 1.27
        return a * (1+z)**g * N_HI**-b

    return 0.0

vf_HM12 = np.vectorize(f_HM12, otypes=[np.float])

def sigma_HI(nu):

    """Calculate the HI ionization cross-section.  

    This parameterisation is taken from Osterbrock and Ferland 2006
    (Sausalito, California: University Science Books), Equation (2.4).

    """
    
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

    N_HI_min = 1.0e11
    N_HI_max = 10.0**21.55 # Limit taken in HM12 Table 1. 
    n = np.logspace(np.log(N_HI_min), np.log(N_HI_max), num=50, base=np.e)
    fn = n * vf_HM12(n, z) * (1.0-np.exp(-n*sigma_HI(nu)))
    return np.trapz(fn, x=np.log(n))

def luminosity(M):

    return 10.0**((51.60-M)/2.5) # ergs s^-1 Hz^-1 

def fnu(nu, M):
    
    L = luminosity(M)
    w = c_angPerSec / nu
    nu_912 = c_angPerSec / 912.0
    nu_1450 = c_angPerSec / 1450.0

    a = L 
    b = a * (912.0/1450.0)**0.61
    
    if w > 912.0:
        e = a*(w/1450.0)**0.61
    else:
        if M < -23.0: 
            e = b*(w/912.0)**1.70
        else:
            e = b*(w/912.0)**0.56 
            
    return e 

vfnu = np.vectorize(fnu, excluded=['M'], otypes=[np.float])

def emissivity(w, z, loglf, theta, mbright=-30, mfaint=-20):

    m = np.linspace(mbright, mfaint, num=1000)
    nu = c_angPerSec/w

    farr = np.array([10.0**loglf(theta, x, z)*vfnu(nu, x) for x in m])

    return np.trapz(farr, m, axis=0) # erg s^-1 Hz^-1 Mpc^-3

def g_lsa(z, loglf, theta):

    """Photoionisation rate with Local Source Approx.
    
    """

    w = 912.0 
    
    em = emissivity(w, z, loglf, theta, mbright=-30.0, mfaint=-23.0)
    alpha_EUV = -1.7
    part1 = 4.6e-13 * (em/1.0e24) * ((1.0+z)/5.0)**(-2.4) / (1.5-alpha_EUV) # s^-1 

    em = emissivity(w, z, loglf, theta, mbright=-23.0, mfaint=-20.0)    
    alpha_EUV = -0.56
    part2 = 4.6e-13 * (em/1.0e24) * ((1.0+z)/5.0)**(-2.4) / (1.5-alpha_EUV) # s^-1

    return part1 + part2

def gs_lsa(lfg, dz=0.1, zmax=7.0):

    zmin = 0.0
    n = (zmax-zmin)/dz+1
    zs = np.linspace(zmax, zmin, num=n)

    theta = np.median(lfg.samples, axis=0)
    
    gs = np.array([g_lsa(x, lfg.log10phi, theta) for x in zs])

    return zs, gs 
    
def j(emodel, loglf=None, theta=None, dz=0.1, n_ws=200, n_ws_int=100, zmax=7.0):

    """Calculate the mean specific intensity.

    Actually, directly outputs a redshift, photoionisation rate, pair. 

    Needs a model for emissivity.  From my tests, I have found that dz
    = 0.01, n_ws = 600, n_ws_int = 400, and zmax = 9 is necessary for
    convergence.

    """
    
    ws = np.logspace(0.0, 5.0, num=n_ws)
    nu = c_angPerSec/ws 

    j = np.zeros_like(nu)
    
    zmin = 0.0
    n = (zmax-zmin)/dz+1
    zs = np.linspace(zmax, zmin, num=n)
    gs = []

    for z in zs:

        if loglf is not None:
            e = emodel(ws/(1.0+z), z, loglf, theta)
        else:
            e = emodel(ws/(1.0+z), z)
        
        # [j] = erg s^-1 Hz^-1 cm^-2 sr^-1
        j = j + (e*c_mpcPerYr*np.abs(dtdz(z))*dz*cmbympc**2)/(4.0*np.pi) 

        nu_rest = c_angPerSec*(1.0+z)/ws 
        t = np.array([tau_eff(x, z) for x in nu_rest])
        j = j*np.exp(-t*dz)

        n = 4.0*np.pi*j*(1.0+z)**3/(hplanck*nu_rest)

        nu_int = np.logspace(np.log10(nu0), 18, num=n_ws_int)
        n_int = np.interp(nu_int, nu_rest[::-1], n[::-1])
        s = np.array([sigma_HI(x) for x in nu_int])
        g = np.trapz(n_int*s, x=nu_int) # s^-1 

        gs.append(g)

    gs = np.array(gs)

    return zs, gs

def em_hm12(w, z):

    """HM12 Galaxies+QSO emissivity."""

    return emissivity_HM12(w, z*np.ones_like(w), grid=False)

def em_qso_hm12(w, z):

    """HM12 QSOs-only emissivity."""

    return vqso_emissivity_hm12(c_angPerSec/w, z)

def calverley(ax):

    zm, gm, gm_sigma = np.loadtxt('Data/calverley.dat',unpack=True) 
    gm += 12.0

    gml = 10.0**gm
    gml_up = 10.0**(gm+gm_sigma)-10.0**gm
    gml_low = 10.0**gm - 10.0**(gm-gm_sigma)
    
    ax.scatter(zm, gml, c='darkorange', edgecolor='None',
               label='Calverley et al.~(2011)', s=64) 
    ax.errorbar(zm, gml, ecolor='darkorange', capsize=5,
                elinewidth=1.5, capthick=1.5,
                yerr=np.vstack((gml_low, gml_up)),
                fmt='None', zorder=1, mfc='darkorange',
                mec='darkorange', markeredgewidth=1, ms=5)

    return

def bb13(ax):

    zm, gm, gm_up, gm_low = np.loadtxt('Data/BeckerBolton.dat', unpack=True)
    
    gml = 10.0**gm
    gml_up = 10.0**(gm+gm_up)-10.0**gm
    gml_low = 10.0**gm - 10.0**(gm-np.abs(gm_low))

    ax.scatter(zm, gml, c='#d7191c', edgecolor='None',
               label='Becker and Bolton (2013)', s=64)
    ax.errorbar(zm, gml, ecolor='#d7191c', capsize=5,
                elinewidth=1.5, capthick=1.5,
                yerr=np.vstack((gml_low, gml_up)),
                fmt='None', zorder=1, mfc='#d7191c', mec='#d7191c',
                markeredgewidth=1, ms=5)

    return

def g_hm12_total(ax):

    zs = np.linspace(0.0, 7.0, num=200)
    nu = np.logspace(np.log10(nu0), 18, num=1000)
    g_hm12 = []
    
    for r in zs:
        j_hm12 = bkgintens_HM12(c_angPerSec/nu, r*np.ones_like(nu), grid=False)
        n = 4.0*np.pi*j_hm12/(hplanck*nu)
        s = np.array([sigma_HI(x) for x in nu])
        g_hm12.append(np.trapz(n*s, x=nu)) # s^-1

    ax.plot(zs, np.array(g_hm12)/1.0e-12, c='forestgreen', lw=2,
            dashes=[7,2], label='Haardt and Madau (2012)')

    return

def lfis(individuals, ax):

    c = np.array([x.gammapi[2]+12.0 for x in individuals])
    u = np.array([x.gammapi[0]+12.0 for x in individuals])
    l = np.array([x.gammapi[1]+12.0 for x in individuals])

    gml = 10.0**c
    gml_up = 10.0**u-10.0**c
    gml_low = 10.0**c - 10.0**l
    
    zs = np.array([x.z.mean() for x in individuals])
    uz = np.array([x.z.max() for x in individuals])
    lz = np.array([x.z.min() for x in individuals])

    uz = np.array([x.zlims[0] for x in individuals])
    lz = np.array([x.zlims[1] for x in individuals])
    
    uzerr = uz-zs
    lzerr = zs-lz 

    ax.scatter(zs, gml, c='#ffffff', edgecolor='k',
               label='Individual fits ($M<-20$, local source approximation)',
               s=44, zorder=4, linewidths=1.5) 
    ax.errorbar(zs, gml, ecolor='k', capsize=0, fmt='None', elinewidth=1.5,
                yerr=np.vstack((gml_low,gml_up)),
                xerr=np.vstack((lzerr,uzerr)), 
                mfc='#ffffff', mec='#404040', zorder=3, mew=1,
                ms=5)

    return 
    
def draw_g(lfg, z2=None, g2=None, individuals=None):

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'$\Gamma_\mathrm{HI}~[10^{-12} \mathrm{s}^{-1}]$')
    ax.set_xlabel('$z$')

    ax.set_yscale('log')
    ax.set_ylim(1.0e-2, 50)
    ax.set_xlim(0.,7)

    locs = (0.01, 0.1, 1, 10)
    labels = ('0.01', '0.1', '1', '10')
    plt.yticks(locs, labels)

    for x in lfg.samples[np.random.randint(len(lfg.samples), size=3)]:
        z, g = j(emissivity, loglf=lfg.log10phi, theta=x, zmax=9.7)
        ax.plot(z, g/1.0e-12, c='goldenrod', lw=2)
    
    theta = np.median(lfg.samples, axis=0)
    z, g = j(emissivity, loglf=lfg.log10phi, theta=theta, zmax=9.7)
    ax.plot(z, g/1.0e-12, c='k', lw=2, label=r'Global model ($M<-20$)')
    
    if z2 is not None:
        ax.plot(z2, g2/1.0e-12, c='k', lw=2,
                label=r'Global model ($M<-20$) local source approximation')
    
    zs_hm12, gs_hm12 = j(em_qso_hm12)
    ax.plot(zs_hm12, gs_hm12/1.0e-12, c='forestgreen', lw=2,
            label='Haardt and Madau (2012) QSO contribution')

    g_hm12_total(ax)
    bb13(ax)
    calverley(ax)

    if individuals is not None:
        lfis(individuals, ax)

    plt.legend(loc='upper left', fontsize=10, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.3, scatterpoints=1)
    
    plt.savefig('g.pdf'.format(z),bbox_inches='tight')
    plt.close('all')

    return

# gs = j(em_hm12)
# zs_hm12, gs_hm12 = j(em_qso_hm12)
# draw_g(zs_hm12, gs_hm12)

