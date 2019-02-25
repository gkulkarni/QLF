"""Calculates Q(z) for hydrogen and helium.

See also: q.py. 

"""

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import RectBivariateSpline
import cosmolopy.distance as cd
import cosmolopy.density as crho 

if __name__ == '__main__':
    import matplotlib as mpl
    mpl.use('Agg') 
    mpl.rcParams['text.usetex'] = True 
    mpl.rcParams['font.family'] = 'serif'
    mpl.rcParams['font.serif'] = 'cm'
    mpl.rcParams['font.size'] = '22'
    import matplotlib.pyplot as plt
    import sys

data = np.load('e1450_18_21feb.npz')
z18 = data['z']
median18 = data['median']*((912.0/1450.0)**0.61)
up18 = data['up']*((912.0/1450.0)**0.61)
down18 = data['down']*((912.0/1450.0)**0.61)
    
# data = np.load('e1450_18_test.npz')
# z18 = data['z']
# medianbright18 = data['medianbright']*((912.0/1450.0)**0.61)
# downbright18 = data['downbright']*((912.0/1450.0)**0.61)
# upbright18 = data['upbright']*((912.0/1450.0)**0.61)
# medianfaint18 = data['medianfaint']*((912.0/1450.0)**0.61)
# downfaint18 = data['downfaint']*((912.0/1450.0)**0.61)
# upfaint18 = data['upfaint']*((912.0/1450.0)**0.61)

data = np.load('e1450_21_21feb.npz')
z21 = data['z']
median21 = data['median']*((912.0/1450.0)**0.61)
up21 = data['up']*((912.0/1450.0)**0.61)
down21 = data['down']*((912.0/1450.0)**0.61)

# data = np.load('e1450_21_test.npz')
# z21 = data['z']
# medianbright21 = data['medianbright']*((912.0/1450.0)**0.61)
# downbright21 = data['downbright']*((912.0/1450.0)**0.61)
# upbright21 = data['upbright']*((912.0/1450.0)**0.61)
# medianfaint21 = data['medianfaint']*((912.0/1450.0)**0.61)
# downfaint21 = data['downfaint']*((912.0/1450.0)**0.61)
# upfaint21 = data['upfaint']*((912.0/1450.0)**0.61)

    
# data = np.load('e912_18_test.npz')
# z18 = data['z']
# medianbright18 = data['median']
# downbright18 = data['down']
# upbright18 = data['up']
# medianfaint18 = 0.0*data['median']
# downfaint18 = 0.0*data['down']
# upfaint18 = 0.0*data['up']

# data = np.load('e912_21_test.npz')
# z21 = data['z']
# medianbright21 = data['median']
# downbright21 = data['down']
# upbright21 = data['up']
# medianfaint21 = 0.0*data['median']
# downfaint21 = 0.0*data['down']
# upfaint21 = 0.0*data['up']

print 'data loaded'

hplanck = 6.62e-27
c_angPerSec = 2.998e18
nu0 = 3.288e15 # threshold freq for H I ionization; s^-1 (Hz)
msolkg = 1.988435e30 # solar mass; kg
mH = 1.67372353855e-27 # kg; mass of hydrogen atom
mHe = 6.64647616211e-27 # kg; mass of helium atom
cmbympccb = 3.4036771916e-74 # (cm/Mpc)^3 conversion factor
yrbys = 3.154e7 # yr/s conversion factor 

cosmo_params_MH15 = True
if cosmo_params_MH15:
    # Madau and Haardt 2015 parameters
    omega_b = 0.045
    omega_lambda = 0.7
    omega_nr = 0.3
    Y_He = 0.24
    h = 0.7
    cosmo = {'omega_M_0': omega_nr,
             'omega_lambda_0': omega_lambda,
             'omega_k_0': 1.0-omega_nr-omega_lambda,
             'omega_b_0': omega_b,
             'h': h,
             'X_H': 1-Y_He }
else:
    # Planck parameters (cf. Sherwood) 
    omega_b = 0.0482
    omega_lambda = 0.692
    omega_nr = 0.308
    Y_He = 0.24
    h = 0.678
    cosmo = {'omega_M_0': omega_nr,
             'omega_lambda_0': omega_lambda,
             'omega_k_0': 1.0-omega_nr-omega_lambda,
             'omega_b_0': omega_b,
             'h': h,
             'X_H': 1-Y_He }
    
rho_critical = crho.baryon_densities(**cosmo)[0] * 1.0e-10 # 10^10 M_solar/Mpc^3
nH = rho_critical * 1.0e10 * msolkg * omega_b * (1-Y_He) / mH # Mpc^-3
nHe = rho_critical * 1.0e10 * msolkg * omega_b * Y_He / mHe # Mpc^-3


def qso_emissivity_m18(z):

    e1 = np.interp(z, z18, median18)

    return e1, 0.0


def qso_emissivity_m18_down(z):

    e1 = np.interp(z, z18, down18)

    return e1, 0.0


def qso_emissivity_m18_up(z):

    # Fit obtained using gammapi.py
    e1 = np.interp(z, z18, up18)

    return e1, 0.0


def qso_emissivity_m21(z):

    # Fit obtained using gammapi.py
    e1 = np.interp(z, z21, median21)

    return e1, 0.0


def qso_emissivity_m21_down(z):

    # Fit obtained using gammapi.py
    e1 = np.interp(z, z21, down21)

    return e1, 0.0


def qso_emissivity_m21_up(z):

    # Fit obtained using gammapi.py
    e1 = np.interp(z, z21, up21)

    return e1, 0.0


def clumping_factor(z):
    """
    MH15.  See text after Equation (4). 

    """

    return 2.9 * ((1.0+z)/6.0)**-1.1


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


def trec_HeIII(z):

    chi = Y_He/(4*(1-Y_He))
    temperature = 2.0e4 # K 
    alpha_r = alpha_B_HeIII_HuiGnedin1997(temperature) # cm^3 s^-1

    r = (1+2*chi) * (1+z)**3 * alpha_r * nH * cmbympccb * clumping_factor(z) # s^-1

    return 1/r # s 


def dqdt_HeIII(q, z, emissivity):

    yrbys = 3.154e7 # yr/s conversion factor

    # Integrate qso emissivity from 4 Ry to infinity.  We are assuming
    # the same EUV spectral slope of -1.7 (Lusso et al. 2015) for both
    # faint and bright qsos.  A slope of -0.56 (Scott et al. 2004)
    # seems to be ruled out by HeII tau_eff data as it reionizes HeII
    # at z > 4.5.
    
    ebright, efaint = emissivity(z)
    
    nbright = (ebright/(1.7*hplanck) *
         (4**-1.7)) # s^-1 Mpc^-3 

    nfaint = (efaint/(1.7*hplanck) *
         (4**-1.7)) # s^-1 Mpc^-3

    n = nfaint+nbright
    
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


def dqdz_HeIII(q, z, emissivity):

    return dqdt_HeIII(q, z, emissivity)*dtdz(z)


def laplante16(ax, minus=False):

    zc, qc = np.loadtxt('Data_new/laplante16_c.dat', unpack=True)
    zl, ql = np.loadtxt('Data_new/laplante16_l.dat', unpack=True)
    zu, qu = np.loadtxt('Data_new/laplante16_u.dat', unpack=True)

    if minus:
        z = np.linspace(12, 2, 1000)
        assert(np.all(np.diff(zc) > 0))
        q = np.interp(z, zc, qc)
        ax.plot(z, 1-q, c='orange', lw=2, label=r'La Plante \& Trac 2016', zorder=8)
        return 

    # We decided not to plot La Plante error bars.
    z = np.linspace(12, 2, 1000)
    assert(np.all(np.diff(zc) > 0))
    q = np.interp(z, zc, qc)
    ax.plot(z, q, c='peru', lw=2, label=r'La Plante \& Trac 2016', zorder=8)

    return 
    
def puchwein18(ax, minus=False):


    dat = np.loadtxt("Data_new/puchwein18_qheiii.txt")

    if minus:
        ax.plot(1.0/dat[:,0]-1.0, 1-dat[:,8]/7.894736842105262720e-02, lw=2,
                c='steelblue', label=r'Puchwein et al.\ 2018', zorder=9)
        return 
    
    ax.plot(1.0/dat[:,0]-1.0, dat[:,8]/7.894736842105262720e-02, lw=2,
            c='steelblue', label=r'Puchwein et al.\ 2018', zorder=9)
    return

def mcquinn09(ax, minus=False):    

    data = np.loadtxt("Data_new/mcquinn09.txt")
    z = data[:,0]
    x = data[:,1]

    if minus: 
        ax.plot(z, 1-x, lw=2,
                c='k', label=r'McQuinn et al.\ 2009', zorder=9)
        return
    
    ax.plot(z, x, lw=2,
            c='brown', label=r'McQuinn et al.\ 2009', zorder=9)
    return

def compostella14(ax, minus=False):    

    data = np.loadtxt("Data_new/compostella14.txt")
    z = data[:,0]
    x = data[:,1]

    if minus: 
        ax.plot(z, 1-x, lw=2,
                c='brown', label=r'Compostella et al.\ 2014', zorder=9)
        return
    
    ax.plot(z, x, lw=2,
            c='g', label=r'Compostella et al.\ 2014', zorder=9)

    return
    
def plotq():

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_ylabel(r'$Q_\mathrm{HeIII}$')
    ax.set_xlabel('$z$')

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)
    ax.tick_params('x', which='major', pad=6)

    #---------- 

    q_file = 'Data_new/qhe_mh15.txt'
    z_model, q_model = np.loadtxt(q_file, unpack=True)

    # Add Q=1 extension at low z
    z_add = np.linspace(0.0,4.2)
    q_add = np.ones_like(z_add)
    z_model = np.concatenate((z_add, z_model))
    q_model = np.concatenate((q_add, q_model))

    q_model = np.where(q_model<1.0, q_model, 1.0)
    plt.plot(z_model, q_model, c='grey', lw=2, label=r'Madau \& Haardt 2015', zorder=-3, dashes=[7,2])

    #----------

    q_file = 'hm12/q.dat'
    z_model, q_model = np.loadtxt(q_file, usecols=(0,2), unpack=True)
    q_model = np.where(q_model<1.0, q_model, 1.0) 
    plt.plot(z_model, q_model, lw=2, c='grey', label=r'Haardt \& Madau 2012', zorder=-5)

    #----------

    laplante16(ax)

    puchwein18(ax)

    mcquinn09(ax)

    compostella14(ax)

    #----------
    
    z = np.linspace(12, 0, 1000)
    q0 = 1.0e-10
    q = odeint(dqdz_HeIII, q0, z, args=(qso_emissivity_m18,))
    q = np.where(q<1.0, q, 1.0) 

    qdown = odeint(dqdz_HeIII, q0, z, args=(qso_emissivity_m18_down,))
    qdown = np.where(qdown<1.0, qdown, 1.0) 

    qup = odeint(dqdz_HeIII, q0, z, args=(qso_emissivity_m18_up,))
    qup = np.where(qup<1.0, qup, 1.0)

    q18 = ax.fill_between(z, qup.flatten(), y2=qdown.flatten(), color='red', zorder=10, alpha=0.6, edgecolor='None')
    q18bf, = plt.plot(z, q.flatten(), c='red', lw=2, label=r'Kulkarni et al.\ 2019 (this work; $M_{1450}<-18$)', zorder=10)

    #----------

    q = odeint(dqdz_HeIII, q0, z, args=(qso_emissivity_m21,))
    q = np.where(q<1.0, q, 1.0) 

    qdown = odeint(dqdz_HeIII, q0, z, args=(qso_emissivity_m21_down,))
    qdown = np.where(qdown<1.0, qdown, 1.0) 

    qup = odeint(dqdz_HeIII, q0, z, args=(qso_emissivity_m21_up,))
    qup = np.where(qup<1.0, qup, 1.0)
    
    q21 = ax.fill_between(z, qup.flatten(), y2=qdown.flatten(), color='blue', zorder=10, alpha=0.6, edgecolor='None')
    q21bf, = plt.plot(z, q.flatten(), c='blue', lw=2, label=r'Kulkarni et al.\ 2019 (this work; $M_{1450}<-21$)', zorder=10)

    #----------
    
    plt.ylim(0,1.2)
    plt.xlim(2,7)
    plt.xticks(np.arange(2,7.5,1))

    handles, labels = ax.get_legend_handles_labels()
    myorder = [6,7,2,5,4,3,1,0]
    handles = [handles[x] for x in myorder]
    labels = [labels[x] for x in myorder]

    handles[0] = (q18, q18bf)
    handles[1] = (q21, q21bf)

    l1 = plt.legend(handles[:2], labels[:2], loc='upper right', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.5)

    l2 = plt.legend(handles[2:], labels[2:], loc='upper right', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
                    handletextpad=0.4, borderpad=0.5, bbox_to_anchor=[0.99, 0.92])

    ax.add_artist(l1)

    plt.savefig('q.pdf', bbox_inches='tight')
    return 

plotq()


