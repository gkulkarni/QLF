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

def f_HM12(N_HI, z):

    log10NHI = np.log10(N_HI)

    if z < 1.56:
        if 11.0 <= log10NHI < 15.0:
            a = 1.73e8 
            b = 1.5
            g = 0.16
        elif 15.0 <= log10NHI < 17.5:
            a = 5.49e15 
            b = 2.0
            g = 0.16 
        elif 17.5 <= log10NHI < 19.0:
            f17p5 = 5.49e15 * (1+z)**0.16 * (10.0**17.5)**-2
            f19 = 1.28 * (1+z)**0.16 * (10.0**19)**-1.05
            b = np.log10(f17p5/f19)/1.5
            a = f17p5 * (10.0**17.5)**b
            return a * N_HI**-b
        elif 19.0 <= log10NHI < 20.3:
            a = 1.28
            b = 1.05 # Not specified by HM12; assuming same as high z
            g = 0.16 
        elif 20.3 <= log10NHI:
            a = 2.47e19 
            b = 2.0 # Not specified by HM12; assuming same as high z
            g = 0.16 
        return a * (1+z)**g * N_HI**-b
    
    elif 1.56 <= z < 5.5:
        if 11.0 <= log10NHI < 15.0:
            a = 1.2e7
            b = 1.5
            g = 3.0
        elif 15.0 <= log10NHI < 17.5:
            a = 3.8e14
            b = 2.0
            g = 3.0
        elif 17.5 <= log10NHI < 19.0:
            f17p5 = 3.8e14 * (1+z)**3 * (10.0**17.5)**-2
            f19 = 0.45 * (1+z)**1.27 * (10.0**19)**-1.05
            b = np.log10(f17p5/f19)/1.5
            a = f17p5 * (10.0**17.5)**b
            return a * N_HI**-b
        elif 19.0 <= log10NHI < 20.3:
            a = 0.45
            b = 1.05
            g = 1.27
        elif 20.3 <= log10NHI:
            a = 8.7e18
            b = 2.0
            g = 1.27
        return a * (1+z)**g * N_HI**-b

    elif 5.5 <= z:
        if 11.0 <= log10NHI < 15.0:
            a = 29.5
            b = 1.5
            g = 9.9
        elif 15.0 <= log10NHI < 17.5:
            a = 9.35e8 
            b = 2
            g = 9.9
        elif 17.5 <= log10NHI < 19.0:
            # f(N_HI) for log10NHI >= 17.5 not specified by HM12 at
            # these redshifts.  I am assuming same behaviour as low z.
            f17p5 = 9.35e8 * (1+z)**9.9 * (10.0**17.5)**-2
            f19 = 0.45 * (1+z)**1.27 * (10.0**19)**-1.05
            b = np.log10(f17p5/f19)/1.5
            a = f17p5 * (10.0**17.5)**b
            return a * N_HI**-b
        elif 19.0 <= log10NHI < 20.3:
            a = 0.45
            b = 1.05
            g = 1.27
        elif 20.3 <= log10NHI:
            a = 8.7e18
            b = 2.0
            g = 1.27
        return a * (1+z)**g * N_HI**-b

    return 0.0

vf_HM12 = np.vectorize(f_HM12, otypes=[np.float])

def plot_f():

    """Plot HM12 HI column density distribution.

    Compare result to left panel of Figure 1 of HM12. 

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'$\log_{10} f(N_\mathrm{HI},z)$')
    ax.set_xlabel(r'$\log_{10}(N_\mathrm{HI}/\mathrm{cm}^{-2})$') 

    ax.set_ylim(-25.0, -7.0)
    ax.set_xlim(11.0, 21.5)

    locs = range(-24, -6, 2)
    labels = ['$'+str(x)+'$' for x in locs]
    plt.yticks(locs, labels)

    n = np.logspace(11.0,23.0,num=1000)
    z = 3.5
    f = vf_HM12(n, z)
    ax.plot(np.log10(n), np.log10(f), lw=2, c='k', label='$z=3.5$') 
    
    z = 2.0
    f = vf_HM12(n, z)
    ax.plot(np.log10(n), np.log10(f/50), lw=2, c='r', label='$z=2.0$') 

    z = 5.0
    f = vf_HM12(n, z)
    ax.plot(np.log10(n), np.log10(f*50), lw=2, c='b', label='$z=5.0$')

    plt.legend(loc='lower left', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.1, scatterpoints=1)

    
    plt.savefig('f.pdf'.format(z),bbox_inches='tight')
    plt.close('all')

# plot_f()

def plot_f_vs_z():

    """Plot HM12 HI column density distribution evolution. 

    Just to confirm continuity.

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'$\log_{10} f(N_\mathrm{HI},z)$')
    ax.set_xlabel(r'$z$') 

    n = 1.0e12
    z = np.linspace(0,7,num=100)
    f = vf_HM12(n, z)
    ax.plot(z, np.log10(f), lw=2, c='g', label='$N_\mathrm{HI}=10^{12} \mathrm{cm}^{-2}$') 
    
    n = 1.0e16
    z = np.linspace(0,7,num=100)
    f = vf_HM12(n, z)
    ax.plot(z, np.log10(f), lw=2, c='k', label='$N_\mathrm{HI}=10^{16} \mathrm{cm}^{-2}$') 

    n = 1.0e18
    z = np.linspace(0,7,num=100)
    f = vf_HM12(n, z)
    ax.plot(z, np.log10(f), lw=2, c='b', label='$N_\mathrm{HI}=10^{20} \mathrm{cm}^{-2}$') 
    
    n = 1.0e20
    z = np.linspace(0,7,num=100)
    f = vf_HM12(n, z)
    ax.plot(z, np.log10(f), lw=2, c='r', label='$N_\mathrm{HI}=10^{20} \mathrm{cm}^{-2}$') 

    n = 1.0e22
    z = np.linspace(0,7,num=100)
    f = vf_HM12(n, z)
    ax.plot(z, np.log10(f), lw=2, c='brown', label='$N_\mathrm{HI}=10^{22} \mathrm{cm}^{-2}$') 
    
    plt.legend(loc='lower left', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.1, scatterpoints=1)

    plt.savefig('fz.pdf'.format(z),bbox_inches='tight')
    plt.close('all')

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
    N_HI_max = 1.0e17
    
    def integrand(logN_HI):

        N_HI = np.exp(logN_HI)
        tau = sigma_HI(nu)*N_HI 

        return N_HI * f(N_HI, z) * (1.0-np.exp(-tau))

    n = np.linspace(np.log(N_HI_min), np.log(N_HI_max), num=100)

    return np.trapz(integrand(n), x=n)

def draw_j(j, w, z):

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

    ax.plot(w, j/1.0e-22, lw=2, c='k')
    j_hm12 = bkgintens_HM12(w, z*np.ones_like(w), grid=False)
    ax.plot(w, j_hm12/1.0e-22, lw=2, c='tomato')

    ax.axvline(1216.0, lw=1, c='k', dashes=[7,2])
    ax.axvline(912.0, lw=1, c='k', dashes=[7,2])

    plt.title('$z={:g}$'.format(z))
    plt.savefig('j_z{:g}.pdf'.format(z),bbox_inches='tight')
    plt.close('all')

    return

wmin = 5.0
wmax = 5.0e3
ws = np.logspace(0.0, 4.0, num=100)
nu = 2.998e18/ws 
hplanck = 6.626069e-27 # erg s

j = np.zeros_like(nu)

zmax = 15.0
zmin = 0.0
dz = 0.1
n = (zmax-zmin)/dz+1
zs = np.linspace(zmax, zmin, num=n)

gs = []

for z in zs:

    # grid=False ensures that we get a flat array. 
    e = emissivity_HM12(ws*(1.0+z), z*np.ones_like(ws), grid=False)
    j = j + e*c*np.abs(dtdz(z))*dz/(4.0*np.pi)*(1.0+z)**3 

    t = np.array([tau_eff(x, z) for x in 2.998e18/(ws*(1.0+z))])
    j = j*np.exp(-t*dz)

    j = j*(cmbympc**2) # erg s^-1 Hz^-1 cm^-2 sr^-1

    # if z == 5.0:
    #     draw_j(j, ws*(1.0+z), z)
    
    nu_local = 2.998e18/(ws*(1.0+z))
    n = 4.0*np.pi*j/(hplanck*nu_local)
    s = np.array([sigma_HI(x) for x in nu_local])

    # There is a minus sign because nu = c/lambda is a decreasing
    # array so dnu is negative.
    g = -np.trapz(n*s, x=nu_local) # s^-1 
    
    # Compare with HM12. 
    j_hm12 = bkgintens_HM12(ws*(1.0+z), z*np.ones_like(ws), grid=False)
    n = 4.0*np.pi*j_hm12/(hplanck*nu_local)
    s = np.array([sigma_HI(x) for x in nu_local])
    
    g_hm12 = -np.trapz(n*s, x=nu_local) # s^-1

    gs.append(g_hm12)

    # print z, g, g_hm12, g_hm12/g

check_gamma_HM12 = False
if check_gamma_HM12:
    for z in zs:

        nu_local = 2.998e18/(ws*(1.0+z))
        j_hm12 = bkgintens_HM12(ws*(1.0+z), z*np.ones_like(ws), grid=False)
        n = 4.0*np.pi*j_hm12/(hplanck*nu_local)
        s = np.array([sigma_HI(x) for x in nu_local])
        
        g_hm12 = -np.trapz(n*s, x=nu_local) # s^-1
        gs.append(g_hm12)
    
def draw_g(z, g):

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'$\Gamma_\mathrm{HI}~[10^{-12} \mathrm{s}^{-1}]$')
    ax.set_xlabel('$z$')

    ax.set_yscale('log')
    ax.set_ylim(1.0e-1, 2)
    ax.set_xlim(1.,7)

    # locs = (1.0e-2, 1.0e-1, 1.0, 10.0)
    # labels = ('0.01', '0.1', '1', '10')
    # plt.yticks(locs, labels)

    locs = (1.0e-1, 1.0, 2.0)
    labels = ('0.1', '1', '2')
    plt.yticks(locs, labels)
    

    ax.plot(z, g/1.0e-12, c='k', lw=2)
    # ax.scatter(z, g/1.0e-12, c='#ffffff', edgecolor='k',
    #            s=44, zorder=4, linewidths=1.5) 


    zm, gm, gm_up, gm_low = np.loadtxt('Data/BeckerBolton.dat',unpack=True)
    
    gml = 10.0**gm
    gml_up = 10.0**(gm+gm_up)-10.0**gm
    gml_low = 10.0**gm - 10.0**(gm-np.abs(gm_low))

    ax.scatter(zm, gml, c='#d7191c', edgecolor='None', label='Becker and Bolton 2013', s=64)
    ax.errorbar(zm, gml, ecolor='#d7191c', capsize=5, elinewidth=1.5, capthick=1.5,
                yerr=np.vstack((gml_low, gml_up)),
                fmt='None', zorder=1, mfc='#d7191c', mec='#d7191c',
                mew=1, ms=5)

    zm, gm, gm_sigma = np.loadtxt('Data/calverley.dat',unpack=True) 
    gm += 12.0

    gml = 10.0**gm
    gml_up = 10.0**(gm+gm_sigma)-10.0**gm
    gml_low = 10.0**gm - 10.0**(gm-gm_sigma)
    
    ax.scatter(zm, gml, c='#99cc66', edgecolor='None', label='Calverley et al.~2011', s=64) 
    ax.errorbar(zm, gml, ecolor='#99CC66', capsize=5, elinewidth=1.5, capthick=1.5,
                yerr=np.vstack((gml_low, gml_up)), fmt='None', zorder=1, mfc='#99CC66',
                mec='#99CC66', mew=1, ms=5)
    
    plt.savefig('g.pdf'.format(z),bbox_inches='tight')
    plt.close('all')

    return

# gs = np.array(gs)
# draw_g(zs,gs)

def qso_emissivity_hm12(nu, z):

    """HM12 qso comoving emissivity.

    This is given by equations (37) and (38) of HM12.

    """

    w = 2.998e18/nu
    nu_912 = 2.998e18/912.0
    nu_1300 = 2.998e18/1300.0
    
    a = 10.0**24.6 * (1.0+z)**4.68 * np.exp(-0.28*z)/(np.exp(1.77*z)+26.3)
    b = a * (nu_1300/nu_912)**-1.57

    if w > 1300.0:
        e = b*(nu/nu_1300)**-0.44
    else:
        e = a*(nu/nu_912)**-1.57

    return e

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
    
    nu = np.logspace(13.0,17.0,num=10000)

    z = 1.1
    e = np.array([qso_emissivity_hm12(x, z) for x in nu])
    ax.plot(nu, e, lw=2, c='k', label='$z=1.1$') 
    
    z = 3.0
    e = np.array([qso_emissivity_hm12(x, z) for x in nu])
    ax.plot(nu, e, lw=2, c='r', label='$z=3.0$') 

    z = 4.9
    e = np.array([qso_emissivity_hm12(x, z) for x in nu])
    ax.plot(nu, e, lw=2, c='g', label='$z=4.9$') 

    z = 8.1
    e = np.array([qso_emissivity_hm12(x, z) for x in nu])
    ax.plot(nu, e, lw=2, c='b', label='$z=8.1$')

    ax.axvline(2.998e18/912.0, lw=1, c='k', dashes=[7,2])
    
    plt.legend(loc='lower left', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.1, scatterpoints=1)
    
    plt.savefig('e_qso.pdf'.format(z),bbox_inches='tight')
    plt.close('all')

    return
    
def check_emissivity():

    """Plot HM12 galaxy emissivity.

    This for comparison with HM12 figure 15 (right panel).  Galaxy
    emissivity is obtained by subtracting the qso emissivity from the
    published total emissivity.

    If show_qso_spectrum is set, also plots qsi emissivity.

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'comoving emissivity per log bandwidth '+
                  '[$10^{39}$~erg s$^{-1}$ cMpc$^{-3}$]', fontsize=14)
    ax.set_xlabel(r'$E$~[eV]', fontsize=14)

    ax.set_yscale('log')
    ax.set_xscale('log')
    
    ax.set_ylim(0.3, 3000.0)
    ax.set_xlim(1.0, 60.0)

    locs = (1.0, 5.0, 10.0, 50.0)
    labels = ('1', '5', '10', '50')
    plt.xticks(locs, labels)

    locs = (1.0, 10.0, 100.0, 1000.0)
    labels = ('1', '10', '100', '1000')
    plt.yticks(locs, labels)
    
    nu = np.logspace(13.0,17.0,num=10000)

    dnu = np.diff(nu)
    num = (nu[1:]+nu[:-1])/2.0
    erg_to_eV = 6.2415091e11

    dnu2 = np.diff(np.log(nu))
    print np.unique(dnu2)

    show_qso_spectrum = False
    
    z = 1.1
    e_qso = np.array([qso_emissivity_hm12(x, z) for x in num])
    e = emissivity_HM12(2.998e18/num, z*np.ones_like(num), grid=False)
    if show_qso_spectrum: 
        ax.plot(hplanck*num*erg_to_eV, 1.0e-39*e_qso*dnu/dnu2,
                lw=2, c='k', label='$z=1.1$', dashes=[7,2]) 
    ax.plot(hplanck*num*erg_to_eV, 1.0e-39*(e-e_qso)*dnu/dnu2,
            lw=2, c='k', label='$z=1.1$') 
    
    z = 3.0
    e_qso = np.array([qso_emissivity_hm12(x, z) for x in num])
    e = emissivity_HM12(2.998e18/num, z*np.ones_like(num), grid=False)
    if show_qso_spectrum: 
        ax.plot(hplanck*num*erg_to_eV, 1.0e-39*e_qso*dnu/dnu2,
                lw=2, c='r', label='$z=3.0$', dashes=[7,2]) 
    ax.plot(hplanck*num*erg_to_eV, 1.0e-39*(e-e_qso)*dnu/dnu2,
            lw=2, c='r', label='$z=3.0$') 

    z = 4.9
    e_qso = np.array([qso_emissivity_hm12(x, z) for x in num])
    e = emissivity_HM12(2.998e18/num, z*np.ones_like(num), grid=False)
    if show_qso_spectrum: 
        ax.plot(hplanck*num*erg_to_eV, 1.0e-39*e_qso*dnu/dnu2,
                lw=2, c='g', label='$z=4.9$', dashes=[7,2]) 
    ax.plot(hplanck*num*erg_to_eV, 1.0e-39*(e-e_qso)*dnu/dnu2,
            lw=2, c='g', label='$z=4.9$') 

    z = 8.1
    e_qso = np.array([qso_emissivity_hm12(x, z) for x in num])
    e = emissivity_HM12(2.998e18/num, z*np.ones_like(num), grid=False)
    if show_qso_spectrum: 
        ax.plot(hplanck*num*erg_to_eV, 1.0e-39*e_qso*dnu/dnu2,
                lw=2, c='b', label='$z=8.1$', dashes=[7,2]) 
    ax.plot(hplanck*num*erg_to_eV, 1.0e-39*(e-e_qso)*dnu/dnu2,
            lw=2, c='b', label='$z=8.1$') 
    
    ax.axvline(13.6, lw=1, c='k', dashes=[7,2])

    plt.legend(loc='lower left', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.1, scatterpoints=1)
    
    plt.savefig('e.pdf'.format(z),bbox_inches='tight')
    plt.close('all')

    return

    

