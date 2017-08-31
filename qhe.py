"""Calculates Q(z) for hydrogen and helium.

See also: q.py. 

"""

import numpy as np
if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg') 
    mpl.rcParams['text.usetex'] = True 
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'cm'
    mpl.rcParams['font.size'] = '22'
    import matplotlib.pyplot as plt
    from scipy.integrate import odeint
    import sys 

hplanck = 6.62e-27

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


def trec_HeII(z):

    chi = 0.083
    temperature = 2.0e4 # K 
    alpha_r = alpha_B_HeII_HuiGnedin1997(temperature) # cm^3 s^-1

    r = (1+chi) * (1+z)**3 * alpha_r * nHe * cmbympccb # s^-1

    return 1/r # s 


def trec_HeIII(z):

    chi = 0.083
    temperature = 2.0e4 # K 
    alpha_r = alpha_B_HeIII_HuiGnedin1997(temperature/4.0) # cm^3 s^-1

    r = (1+2*chi) * (1+z)**3 * alpha_r * nH * cmbympccb # s^-1

    return 1/r # s 


def dqdt_HII(q, z):

    yrbys = 3.154e7 # yr/s conversion factor

    # Integrate qso emissivity from 1 to 4 Ry 
    n = (qso_emissivity_MH15(z)/(1.70*hplanck) -
         qso_emissivity_MH15(z)*0.25**1.7/(1.70*hplanck)) # s^-1 Mpc^-3 
    
    d = n/nH - q/trec_HII(z) # s^-1

    return d * yrbys # yr


def dqdt_HeII(q, z):

    yrbys = 3.154e7 # yr/s conversion factor

    # Integrate qso emissivity from 1.8 to 10 Ry 
    n = (qso_emissivity_MH15(z)/(1.70*hplanck) *
         (1.8**-1.7 - 10.0**-1.7)) # s^-1 Mpc^-3 
    
    d = n/nHe - q/trec_HeII(z) # s^-1

    return d * yrbys # yr


def dqdt_HeIII(q, z):

    yrbys = 3.154e7 # yr/s conversion factor

    # Integrate qso emissivity from 4 to 10 Ry 
    n = (qso_emissivity_MH15(z)/(1.70*hplanck) *
         (4**-1.7 - 10.0**-1.7)) # s^-1 Mpc^-3 
    
    d = n/nHe - q/trec_HeIII(z) # s^-1

    return d * yrbys # yr


def H(z):

    H0 = 1.023e-10*h
    hubp = H0*np.sqrt(omega_nr*(1+z)**3+omega_lambda) # yr^-1

    return hubp


def dzdt(z):

    return -(1+z)*H(z) # yr^-1


def dtdz(z):

    return 1/dzdt(z) # yr 


def dqdz_HII(q, z):

    return dqdt_HII(q, z)*dtdz(z)


def dqdz_HeII(q, z):

    return dqdt_HeII(q, z)*dtdz(z)


def dqdz_HeIII(q, z):

    return dqdt_HeIII(q, z)*dtdz(z)


def plotq():

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel(r'$Q_V$')
    ax.set_xlabel('$z$')

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)
    ax.tick_params('x', which='major', pad=6)

    z = np.linspace(12, 2, 1000)
    q0 = 1.0e-10

    q = odeint(dqdz_HII, q0, z)
    q = np.where(q<1.0, q, 1.0) 
    plt.plot(z, q, c='k', lw=3, label=r'H~\textsc{ii} (Madau and Haardt 2015; AGN only)')

    # q = odeint(dqdz_HeII, q0, z)
    # q = np.where(q<1.0, q, 1.0) 
    # plt.plot(z, q, c='c', lw=3, label='He~II')

    q = odeint(dqdz_HeIII, q0, z)
    q = np.where(q<1.0, q, 1.0) 
    plt.plot(z, q, c='tomato', lw=3, label=r'He~\textsc{iii} (Madau and Haardt 2015)')

    plt.ylim(0,1.1)
    plt.xlim(2,15)

    plt.legend(loc='upper right', fontsize=10, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.5)
    
    plt.savefig('q.pdf', bbox_inches='tight')
    return 

plotq()


