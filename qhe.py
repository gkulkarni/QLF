"""Calculates Q(z) for hydrogen and helium.

See also: q.py. 

"""

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import RectBivariateSpline

if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg') 
    mpl.rcParams['text.usetex'] = True 
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'cm'
    mpl.rcParams['font.size'] = '22'
    import matplotlib.pyplot as plt
    import sys 

hplanck = 6.62e-27
c_angPerSec = 2.998e18
nu0 = 3.288e15 # threshold freq for H I ionization; s^-1 (Hz)

cosmo_params_MH15 = True 
if cosmo_params_MH15:
    # Madau and Haardt 2015 parameters
    omega_b = 0.045
    omega_lambda = 0.7
    omega_nr = 0.3
else:
    # Sherwood parameters
    omega_b = 0.0482
    omega_lambda = 0.692
    omega_nr = 0.308

z_HM12 = np.loadtxt('z_HM12.txt')
data_HM12 = np.loadtxt('hm12.txt')
w_HM12 = data_HM12[:,0]
e_HM12 = data_HM12[:,1:]

# kx=1 and ky=1 required here to get good interpolation at the Lyman
# break.  Also remember that you will most likely need the
# 'grid=False' option while invoking emissivity_HM12.
emissivity_HM12 = RectBivariateSpline(w_HM12, z_HM12, e_HM12, kx=1, ky=1)


def em_hm12(w, z):

    """HM12 Galaxies+QSO emissivity."""

    return emissivity_HM12(w, z*np.ones_like(w), grid=False)


Y_He = 0.24
h = 0.678
rho_critical = 2.775e1 * h**2 # 10^10 M_solar/Mpc^3
msolkg = 1.988435e30 # solar mass; kg
mproton = 1.672622e-27 # kg; mass of proton
cmbympccb = 3.4036771916e-74 # (cm/Mpc)^3 conversion factor
yrbys = 3.154e7 # yr/s conversion factor 

nH = rho_critical * 1.0e10 * msolkg * omega_b * (1-Y_He) / mproton # Mpc^-3
nHe = rho_critical * 1.0e10 * msolkg * omega_b * Y_He / (4*mproton) # Mpc^-3

def qso_emissivity_MH15(z):

    return 10.0**(25.15*np.exp(-0.0026*z)-1.5*np.exp(-1.3*z))


def qso_emissivity_HM12(z):

    return 10.0**24.6 * (1.0+z)**4.68 * np.exp(-0.28*z)/(np.exp(1.77*z)+26.3) 


def qso_emissivity_m18(z):

    # Fit obtained using gammapi.py

    a, b, c, d, e = 24.44478825, 5.17788495, 0.35096674, 1.57899542,  17.36182159
    return 10.0**a * (1.0+z)**b * np.exp(-c*z) / (np.exp(d*z)+e)


def qso_emissivity_m21(z):

    # Fit obtained using gammapi.py

    a, b, c, d, e = 24.2257009, 5.77594251, 0.21447066, 2.00733499, 23.23731843
    return 10.0**a * (1.0+z)**b * np.exp(-c*z) / (np.exp(d*z)+e)
    
def plot_epsqso():
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel(r'$\epsilon_{912}\left[\mathrm{erg}\;\mathrm{s}^{-1}'+
                  '\mathrm{cMpc}^{-3} \mathrm{Hz}^{-1}\right]$')
    ax.set_xlabel('$z$')

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)
    ax.tick_params('x', which='major', pad=6)

    z = np.linspace(0, 10, 1000)
    n = qso_emissivity_MH15(z)
    plt.plot(z, n, c='k', lw=2)

    plt.yscale('log')

    plt.savefig('epsqso.pdf', bbox_inches='tight')
    return 


def plot_nqso():
    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel(r'$\dot n_\mathrm{ion}' +
                  '\left[\mathrm{s}^{-1}\mathrm{cMpc}^{-3}\right]$')
    ax.set_xlabel('$z$')

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)
    ax.tick_params('x', which='major', pad=6)

    z = np.linspace(0, 10, 1000)

    # Integrate qso emissivity from 1 to 4 Ry 
    n = (qso_emissivity_MH15(z)/(1.70*hplanck) -
         qso_emissivity_MH15(z)*0.25**1.7/(1.70*hplanck)) # s^-1 Mpc^-3 
    
    plt.plot(z, n, c='k', lw=2)

    # Integrate qso emissivity from 4 to 10 Ry 
    n = (qso_emissivity_MH15(z)/(1.70*hplanck) *
         (4.0**-1.7 - 10.0**-1.7)) # s^-1 Mpc^-3 
    
    plt.plot(z, n, c='c', lw=2)

    plt.yscale('log')

    plt.savefig('nqso.pdf', bbox_inches='tight')

    return


def clumping_factor(z):
    """
    MH15.  See text after Equation (4). 

    """

    return 2.9 * ((1.0+z)/6.0)**-1.1


def alpha_B_HII_Draine2011(T):
    """
    Draine equation 14.6 

    """

    return 2.54e-13*(T/1.0e4)**(-0.8163-0.0208*np.log(T/1.0e4)) # cm^3 s^-1


def alpha_B_HII_HuiGnedin1997(T):
    """Case B H II recombination coefficient. 
    
    Hui and Gnedin 1997 (MNRAS 292 27; Appendix A).

    """

    t_HI = 1.57807E5 # K; H ionization threshold 
    reclambda = 2.0*t_HI/T
    
    return (2.753e-14 * (reclambda**1.5) /
            (1.0 + (reclambda/2.74)**0.407)**2.242) # cm^3 s^-1 


def alpha_B_HeII_HuiGnedin1997(T):
    """Case B He II recombination coefficient.

    Hui and Gnedin 1997 (MNRAS 292 27; Appendix A).

    """

    t_HeI = 2.85335E5 # K; He single ionization threshold 
    reclambda = 2.0*t_HeI/T

    return 1.26e-14 * reclambda**0.75 # cm^3 s^-1 
    

def alpha_A_HeIII_HuiGnedin1997(T):
    """Case A He III recombination coefficient.

    Hui and Gnedin 1997 (MNRAS 292 27; Appendix A).

    """

    t_HeII = 6.31515E5 # K; He single ionization threshold 
    reclambda = 2.0*t_HeII/T

    return (2.0 * 1.269e-13 *
            (reclambda**1.503 /
             (1.0 + (reclambda/0.522)**0.470)**1.923)) # cm^3 s^-1


def alpha_B_HeIII_HuiGnedin1997(T):
    """Case B He III recombination coefficient.

    Hui and Gnedin 1997 (MNRAS 292 27; Appendix A).

    """

    t_HeII = 6.31515E5 # K; He single ionization threshold 
    reclambda = 2.0*t_HeII/T

    return (2.0 * 2.753e-14 *
            (reclambda**1.5 /
             (1.0 + (reclambda/2.740)**0.407)**2.242)) # cm^3 s^-1


def trec_HII(z):

    chi = 0.083
    temperature = 2.0e4 # K 
    alpha_r = alpha_B_HII_Draine2011(temperature) # cm^3 s^-1

    r = (1+chi) * (1+z)**3 * alpha_r * nH * cmbympccb # s^-1

    return 1/r # s 


def trec_HeIII(z):

    chi = 0.083
    temperature = 2.0e4 # K 
    alpha_r = alpha_B_HeIII_HuiGnedin1997(temperature/4.0) # cm^3 s^-1

    r = (1+2*chi) * (1+z)**3 * alpha_r * nH * cmbympccb * clumping_factor(z) # s^-1

    return 1/r # s 


def trec_HeIII_noclump(z):

    chi = 0.083
    temperature = 2.0e4 # K 
    alpha_r = alpha_B_HeIII_HuiGnedin1997(temperature/4.0) # cm^3 s^-1

    r = (1+2*chi) * (1+z)**3 * alpha_r * nH * cmbympccb # s^-1

    return 1/r # s 


def dqdt_HII(q, z, emissivity):

    yrbys = 3.154e7 # yr/s conversion factor

    # Integrate qso emissivity from 1 to 4 Ry 
    n = (emissivity(z)/(1.70*hplanck) -
         emissivity(z)*0.25**1.7/(1.70*hplanck)) # s^-1 Mpc^-3 
    
    d = n/nH - q/trec_HII(z) # s^-1

    return d * yrbys # yr


def dqdt_HII_HM12(q, z):

    yrbys = 3.154e7 # yr/s conversion factor
    nu = np.logspace(np.log10(nu0), 20, num=5000)
    w = c_angPerSec/nu
    e = em_hm12(w, z) # ergs/s/Mpc^3/Hz
    n = np.trapz(e/(hplanck*nu), nu) # s^-1 Mpc^-3 
    
    d = n/nH - q/trec_HII(z) # s^-1

    return d * yrbys # yr


def dqdz_HII_HM12(q, z):

    return dqdt_HII_HM12(q, z)*dtdz(z)


def dqdt_HeIII(q, z, emissivity):

    yrbys = 3.154e7 # yr/s conversion factor

    # Integrate qso emissivity from 4 to 10 Ry 
    n = (emissivity(z)/(1.70*hplanck) *
         (4**-1.7 - 10.0**-1.7)) # s^-1 Mpc^-3 
    
    # d = n*(1-q)/nHe - q/trec_HeIII(z) # s^-1
    d = n/nHe - q/trec_HeIII(z) # s^-1

    return d * yrbys # yr

def dqdt_HeIII_noclump(q, z, emissivity):

    yrbys = 3.154e7 # yr/s conversion factor

    # Integrate qso emissivity from 4 to 10 Ry 
    n = (emissivity(z)/(1.70*hplanck) *
         (4**-1.7 - 10.0**-1.7)) # s^-1 Mpc^-3 
    
    d = n/nHe - q/trec_HeIII_noclump(z) # s^-1

    return d * yrbys # yr

def dqdt_HeIII_bolton_llsonly(q, z, emissivity):

    yrbys = 3.154e7 # yr/s conversion factor

    # Integrate qso emissivity from 4 to 10 Ry 
    n = (emissivity(z)/(1.70*hplanck) *
         (4**-1.7 - 10.0**-1.7)) # s^-1 Mpc^-3 
    
    d = n*(1-q)/nHe # s^-1

    return d * yrbys # yr


def dqdt_HeIII_bolton_c1(q, z, emissivity):

    yrbys = 3.154e7 # yr/s conversion factor

    # Integrate qso emissivity from 4 to 10 Ry 
    n = (emissivity(z)/(1.70*hplanck) *
         (4**-1.7 - 10.0**-1.7)) # s^-1 Mpc^-3 
    
    d = n*(1-q)/nHe - q/trec_HeIII_noclump(z) # s^-1

    return d * yrbys # yr


def dqdt_HeIII_bolton_cHM12(q, z, emissivity):

    yrbys = 3.154e7 # yr/s conversion factor

    # Integrate qso emissivity from 4 to 10 Ry 
    n = (emissivity(z)/(1.70*hplanck) *
         (4**-1.7 - 10.0**-1.7)) # s^-1 Mpc^-3 
    
    d = n*(1-q)/nHe - q/trec_HeIII(z) # s^-1

    return d * yrbys # yr


def dqdt_HeIII_norec(q, z, emissivity):

    yrbys = 3.154e7 # yr/s conversion factor

    # Integrate qso emissivity from 4 to 10 Ry 
    n = (emissivity(z)/(1.70*hplanck) *
         (4**-1.7 - 10.0**-1.7)) # s^-1 Mpc^-3 
    
    d = n/nHe # s^-1

    return d * yrbys # yr


def H(z):

    H0 = 1.023e-10*h
    hubp = H0*np.sqrt(omega_nr*(1+z)**3+omega_lambda) # yr^-1

    return hubp

def dzdt(z):

    return -(1+z)*H(z) # yr^-1


def dtdz(z):

    return 1/dzdt(z) # yr 


def dqdz_HII(q, z, emissivity):

    return dqdt_HII(q, z, emissivity)*dtdz(z)


def dqdz_HeIII(q, z, emissivity):

    return dqdt_HeIII(q, z, emissivity)*dtdz(z)


def dqdz_HeIII_noclump(q, z, emissivity):

    return dqdt_HeIII_noclump(q, z, emissivity)*dtdz(z)


def dqdz_HeIII_norec(q, z, emissivity):

    return dqdt_HeIII_norec(q, z, emissivity)*dtdz(z)


def dqdz_HeIII_bolton_llsonly(q, z, emissivity):

    return dqdt_HeIII_bolton_llsonly(q, z, emissivity)*dtdz(z)


def dqdz_HeIII_bolton_c1(q, z, emissivity):

    return dqdt_HeIII_bolton_c1(q, z, emissivity)*dtdz(z)


def dqdz_HeIII_bolton_cHM12(q, z, emissivity):

    return dqdt_HeIII_bolton_cHM12(q, z, emissivity)*dtdz(z)


def laplante16(ax):

    zc, qc = np.loadtxt('Data_new/laplante16_c.dat', unpack=True)
    zl, ql = np.loadtxt('Data_new/laplante16_l.dat', unpack=True)
    zu, qu = np.loadtxt('Data_new/laplante16_u.dat', unpack=True) 

    z = np.linspace(12, 2, 1000)
    assert(np.all(np.diff(zl) > 0))
    q_dn = np.interp(z, zl, ql)
    assert(np.all(np.diff(zu) > 0))
    q_up = np.interp(z, zu, qu)
    ax.fill_between(z, q_dn, y2=q_up, color='#7F9E9A', zorder=8, alpha=0.7)

    assert(np.all(np.diff(zc) > 0))
    q = np.interp(z, zc, qc)
    ax.plot(z, q, c='#7F9E9A', lw=2, label=r'La Plante \& Trac 2016', zorder=8)
    
    return

def puchwein18(ax):

    dat = np.loadtxt("Data_new/puchwein18_qheiii.txt")
    ax.plot(1.0/dat[:,0]-1.0, dat[:,8]/7.894736842105262720e-02, lw=2,
            c='steelblue', label=r'Puchwein et al.\ 2018', zorder=9)

def mcquinn09(ax):    

    data = np.loadtxt("Data_new/mcquinn09.txt")
    z = data[:,0]
    x = data[:,1]
    ax.plot(z, x, lw=2,
            c='b', label=r'McQuinn et al.\ 2009', zorder=9)

def compostella14(ax):    

    data = np.loadtxt("Data_new/compostella14.txt")
    z = data[:,0]
    x = data[:,1]
    ax.plot(z, x, lw=2,
            c='g', label=r'Compostella et al.\ 2014', zorder=9)
    
def plotq():

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel(r'$Q_V^\mathrm{HeIII}$')
    ax.set_xlabel('$z$')

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)
    ax.tick_params('x', which='major', pad=6)

    z = np.linspace(12, 1, 1000)
    q0 = 1.0e-10
    q = odeint(dqdz_HeIII, q0, z, args=(qso_emissivity_MH15,))
    q = np.where(q<1.0, q, 1.0) 
    plt.plot(z, q, c='grey', lw=2, label=r'Madau \& Haardt 2015', zorder=-3, dashes=[7,2])

    q_file = 'hm12/q.dat'
    z_model, q_model = np.loadtxt(q_file, usecols=(0,2), unpack=True)
    q_model = np.where(q_model<1.0, q_model, 1.0) 
    plt.plot(z_model, q_model, lw=2, c='grey', label=r'Haardt \& Madau 2012', zorder=-5)

    laplante16(ax)

    puchwein18(ax)

    mcquinn09(ax)

    compostella14(ax)
    
    z = np.linspace(12, 0, 1000)
    q0 = 1.0e-10
    q = odeint(dqdz_HeIII, q0, z, args=(qso_emissivity_m18,))
    q = np.where(q<1.0, q, 1.0) 
    plt.plot(z, q, c='peru', lw=3, label=r'This work ($M_{1450}<-18$)', zorder=10)

    z = np.linspace(12, 0, 1000)
    q0 = 1.0e-10
    q = odeint(dqdz_HeIII, q0, z, args=(qso_emissivity_m21,))
    q = np.where(q<1.0, q, 1.0) 
    plt.plot(z, q, c='k', lw=3, label=r'This work ($M_{1450}<-21$)', zorder=10)

    plt.ylim(0,1.1)
    plt.xlim(1,12)

    plt.legend(loc='upper right', fontsize=14, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.5)
    
    plt.savefig('q.pdf', bbox_inches='tight')
    return 

def plotq_lls():

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel(r'$Q_V^\mathrm{HeIII}$')
    ax.set_xlabel('$z$')

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)
    ax.tick_params('x', which='major', pad=6)

    z = np.linspace(12, 0, 1000)
    q0 = 1.0e-10
    q = odeint(dqdz_HeIII_norec, q0, z, args=(qso_emissivity_m18,))
    q = np.where(q<1.0, q, 1.0) 
    plt.plot(z, q, c='b', lw=2, label=r'$\mathcal{C}=0$')

    z = np.linspace(12, 0, 1000)
    q0 = 1.0e-10
    q = odeint(dqdz_HeIII_noclump, q0, z, args=(qso_emissivity_m18,))
    q = np.where(q<1.0, q, 1.0) 
    plt.plot(z, q, c='b', lw=2, label=r'$\mathcal{C}=1$', dashes=[7,2])

    z = np.linspace(12, 0, 1000)
    q0 = 1.0e-10
    q = odeint(dqdz_HeIII, q0, z, args=(qso_emissivity_m18,))
    q = np.where(q<1.0, q, 1.0) 
    plt.plot(z, q, c='b', lw=2, label=r'$\mathcal{C}=\mathrm{HM12}$', dashes=[7,2,2,2])
    
    z = np.linspace(12, 0, 1000)
    q0 = 1.0e-10
    q = odeint(dqdz_HeIII_bolton_llsonly, q0, z, args=(qso_emissivity_m18,))
    q = np.where(q<1.0, q, 1.0) 
    plt.plot(z, q, c='r', lw=2, label=r'with LLS, $\mathcal{C}=0$')

    z = np.linspace(12, 0, 1000)
    q0 = 1.0e-10
    q = odeint(dqdz_HeIII_bolton_c1, q0, z, args=(qso_emissivity_m18,))
    q = np.where(q<1.0, q, 1.0) 
    plt.plot(z, q, c='r', lw=2, label=r'with LLS, $\mathcal{C}=1$', dashes=[7,2])

    z = np.linspace(12, 0, 1000)
    q0 = 1.0e-10
    q = odeint(dqdz_HeIII_bolton_cHM12, q0, z, args=(qso_emissivity_m18,))
    q = np.where(q<1.0, q, 1.0) 
    plt.plot(z, q, c='r', lw=2, label=r'with LLS, $\mathcal{C}=\mathrm{HM12}$', dashes=[7,2,2,2])
    
    plt.ylim(0,1.3)
    plt.xlim(1,10)

    plt.legend(loc='upper right', fontsize=14, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.5)
    
    plt.savefig('q.pdf', bbox_inches='tight')
    return 

plotq()


