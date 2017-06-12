import numpy as np 
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
from numpy.polynomial import Chebyshev as T

p_log10phiStar = [-7.73388053, 1.06477161, -0.11304974]
# p_MStar = [-17.84979944, -4.90153699, 0.49748768, -0.01925119]
p_MStar = [-0.07700476, 0.99497536, -4.84378342, -18.34728712] #[-45.54987107,  66.51882126, -37.01499714, -18.47003438]
p_alpha = [-3.22779072, -0.27456505]
p_beta = [-2.42666272, 0.91088261, 3.48830563, 9.96182361, -0.1530638]

# p_log10phiStar = np.array([-8.73893404,  1.62542678, -0.14453768])
# p_MStar = np.array([-18.46966787,  -4.8580583 ,   0.57781389,  -0.02531129])
# p_alpha = np.array([-4.45250745,  0.20357239])
# p_beta = np.array([-2.33424594,  1.19579103,  3.01011911,  8.31975455, -1.02475454])


def lfParams(z, pPhiStar, pMStar, pAlpha, pBeta):

    log10phiStar = T(pPhiStar)(1+z)
    # mStar = T(pMStar)(1+z)
    # mStar = np.polyval(pMStar, np.log10(1.0+z))
    mStar = np.polyval(pMStar, 1.0+z)
    alpha = T(pAlpha)(1+z)

    h, f0, z0, a, b = pBeta
    zeta = np.log10((1.0+z)/(1.0+z0))
    beta = h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))

    return log10phiStar, mStar, alpha, beta

def phi(z, m, *params):

    log10phiStar, mStar, alpha, beta = lfParams(z, *params)
    phi = 10.0**log10phiStar / (10.0**(0.4*(alpha+1)*(m-mStar)) +
                                 10.0**(0.4*(beta+1)*(m-mStar)))
    return phi

def dlfParamsdz(z, pPhiStar, pMStar, pAlpha, pBeta):

    dlog10phiStardz = T.deriv(T(pPhiStar))(1+z)
    dmStardz = T.deriv(T(pMStar))(1+z)
    dalphadz = T.deriv(T(pAlpha))(1+z)

    h, f0, z0, a, b = pBeta
    zeta = np.log10((1.0+z)/(1.0+z0))
    dbetadz = (-f0*(a*10.0**((a-1)*zeta)/(1.0+z0)
                   + b*10.0**((b-1)*zeta)/(1.0+z0))/
               (10.0**(a*zeta) + 10.0**(b*zeta))**2)

    return dlog10phiStardz, dmStardz, dalphadz, dbetadz

def dphidphiStar(z, m, *params):

    return phi(z, m, *params) * np.log(10.0)

def dphidalpha(z, m, *params):

    log10phiStar, mStar, alpha, beta = lfParams(z, *params)
    phi = 10.0**log10phiStar / (10.0**(0.4*(alpha+1)*(m-mStar)) +
                                 10.0**(0.4*(beta+1)*(m-mStar)))
    d = (- phi * np.log(10.0) * 0.4 * (m-mStar) *  10.0**(0.4*(alpha+1)*(m-mStar)) /
         (10.0**(0.4*(alpha+1)*(m-mStar)) + 10.0**(0.4*(beta+1)*(m-mStar))))
    return d

def dphidbeta(z, m, *params):

    log10phiStar, mStar, alpha, beta = lfParams(z, *params)
    phi = 10.0**log10phiStar / (10.0**(0.4*(alpha+1)*(m-mStar)) +
                                 10.0**(0.4*(beta+1)*(m-mStar)))
    d = (- phi * np.log(10.0) * 0.4 * (m-mStar) *  10.0**(0.4*(beta+1)*(m-mStar)) /
         (10.0**(0.4*(alpha+1)*(m-mStar)) + 10.0**(0.4*(beta+1)*(m-mStar))))
    return d
    
def dphidMStar(z, m, *params):
    log10phiStar, mStar, alpha, beta = lfParams(z, *params)
    phi = 10.0**log10phiStar / (10.0**(0.4*(alpha+1)*(m-mStar)) +
                                 10.0**(0.4*(beta+1)*(m-mStar)))
    d1 = 10.0**(0.4*(alpha+1)*(m-mStar)) * np.log(10.0) * (-0.4*(alpha+1))
    d2 = 10.0**(0.4*(beta+1)*(m-mStar)) * np.log(10.0) * (-0.4*(beta+1))
    d = (- phi * (d1+d2) /
         (10.0**(0.4*(alpha+1)*(m-mStar)) + 10.0**(0.4*(beta+1)*(m-mStar))))
    return d 
    
def dphidz(z, m, *params):

    dphiStardz, dmStardz, dalphadz, dbetadz = dlfParamsdz(z, *params)
    
    return (dphidphiStar(z, m, *params)*dphiStardz + 
            dphidMStar(z, m, *params)*dmStardz + 
            dphidalpha(z, m, *params)*dalphadz + 
            dphidbeta(z, m, *params)*dbetadz)
    
def plotLF(*params):

    m = np.linspace(-55., -10., num=500)
    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.set_ylabel(r'$\phi$')
    ax.set_xlabel(r'$M$')
    ax.set_yscale('log')

    for z in range(5, 15, 2):
        p = [phi(z, x, *params) for x in m]
        plt.plot(m, p, lw=2, label='$z='+str(z)+'$')

    leg = plt.legend(loc='lower left', fontsize=10, handlelength=3,
                     frameon=False, framealpha=0.0, labelspacing=.1,
                     handletextpad=0.1, borderpad=0.01, scatterpoints=1)
        
    plt.xlim(-10,-55)
    plt.savefig('lf.pdf', bbox_inches='tight')

    return 
    
def rhoqso(mlim, z, *params):

    m = np.linspace(-35.0, mlim, num=1000)
    farr = phi(z, m, *params) 
    return np.trapz(farr, m)

def drhoqsodz(mlim, z, *params):

    m = np.linspace(-35.0, mlim, num=1000)
    farr = dphidz(z, m, *params) 
    return np.trapz(farr, m)

def plotRhoQso(zmin, zmax): 
    fig = plt.figure(figsize=(7, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    # ax.set_ylabel(r'$\rho(z, M_{1450} < M_\mathrm{lim})$ [cMpc$^{-3}$]')
    ax.set_ylabel(r'$\rho_\mathrm{qso}$ [cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    zmin = 0
    zmax = 15
    ax.set_xlim(zmin, zmax)

    zc = np.linspace(zmin, zmax, num=500)
    rho = [rhoqso(-18, x, p_log10phiStar, p_MStar, p_alpha, p_beta) for x in zc]
    ax.plot(zc, rho, c='k', lw=2)
    #plt.text(5.0, 3e-5, '$M<-18$', fontsize=16, rotation=-61)
    
    rho = [rhoqso(-21, x, p_log10phiStar, p_MStar, p_alpha, p_beta) for x in zc]
    ax.plot(zc, rho, c='k', lw=2)
    #plt.text(4.5, 1.4e-6, '$M<-21$', fontsize=16, rotation=-63)

    rho = [rhoqso(-24, x, p_log10phiStar, p_MStar, p_alpha, p_beta) for x in zc]
    ax.plot(zc, rho, c='k', lw=2)
    #plt.text(4.2, 6.0e-8, '$M<-24$', fontsize=16, rotation=-64)
    
    rho = [rhoqso(-27, x, p_log10phiStar, p_MStar, p_alpha, p_beta) for x in zc]
    ax.plot(zc, rho, c='k', lw=2)
    #plt.text(4.5, 6.0e-10, '$M<-27$', fontsize=16, rotation=-62)

    ax.set_yscale('log')
    ax.set_ylim(1.0e-40, 1.0e-3)

    plt.savefig('rhoqso_test.pdf',bbox_inches='tight')
    return 

def plotdRhoQsodz(zmin, zmax): 
    fig = plt.figure(figsize=(7, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    # ax.set_ylabel(r'$d\rho(z, M_{1450} < M_\mathrm{lim})/dz$ [cMpc$^{-3}$]')
    ax.set_ylabel(r'$d\rho_\mathrm{qso}/dz$ [cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    zmin = 0
    zmax = 15
    ax.set_xlim(zmin, zmax)

    zc = np.linspace(zmin, zmax, num=500)
    rho = np.array([drhoqsodz(-18, x, p_log10phiStar, p_MStar, p_alpha, p_beta) for x in zc])
    rhop = np.where(rho>0.0, rho, 0.0)
    rhon = np.where(rho<0.0, -rho, 0.0)
    ax.plot(zc, rhop, c='k', lw=2)
    ax.plot(zc, rhon, c='tomato', lw=2)
    #plt.text(9.0, 10, '$M<-18$', fontsize=16, rotation=52)

    zc = np.linspace(zmin, zmax, num=500)
    rho = np.array([drhoqsodz(-21, x, p_log10phiStar, p_MStar, p_alpha, p_beta) for x in zc])
    rhop = np.where(rho>0.0, rho, 0.0)
    rhon = np.where(rho<0.0, -rho, 0.0)
    ax.plot(zc, rhop, c='k', lw=2)
    ax.plot(zc, rhon, c='tomato', lw=2)

    zc = np.linspace(zmin, zmax, num=500)
    rho = np.array([drhoqsodz(-24, x, p_log10phiStar, p_MStar, p_alpha, p_beta) for x in zc])
    rhop = np.where(rho>0.0, rho, 0.0)
    rhon = np.where(rho<0.0, -rho, 0.0)
    ax.plot(zc, rhop, c='k', lw=2)
    ax.plot(zc, rhon, c='tomato', lw=2)

    zc = np.linspace(zmin, zmax, num=500)
    rho = np.array([drhoqsodz(-27, x, p_log10phiStar, p_MStar, p_alpha, p_beta) for x in zc])
    rhop = np.where(rho>0.0, rho, 0.0)
    rhon = np.where(rho<0.0, -rho, 0.0)
    ax.plot(zc, rhop, c='k', lw=2, label=r'positive values')
    ax.plot(zc, rhon, c='tomato', lw=2, label=r'negative values')
    #plt.text(10.0, 1.0e-5, '$M<-27$', fontsize=16, rotation=55)
    
    ax.set_yscale('log')
    ax.set_ylim(1.0e-20, 1.0e30)

    leg = plt.legend(loc='upper left', fontsize=14, handlelength=3,
                     frameon=False, framealpha=0.0, labelspacing=.1,
                     handletextpad=0.1, borderpad=0.5, scatterpoints=1)

    plt.savefig('drhoqsodz_test.pdf',bbox_inches='tight')
    return 

def plotParams(zmin, zmax): 

    mpl.rcParams['font.size'] = '14'

    fig = plt.figure(figsize=(6, 6), dpi=100)

    K = 4
    nplots_x = 2
    nplots_y = 2
    nplots = 4
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.1         # w/hspace size
    plotdim = factor * K + factor * (K - 1.) * whspace
    dim = lbdim + plotdim + trdim
    lb = lbdim / dim
    tr = (lbdim + plotdim) / dim
    fig.subplots_adjust(left=lb, bottom=lb, right=tr, top=tr,
                        wspace=whspace, hspace=whspace)

    zmin = 0
    zmax = 15
    zc = np.linspace(zmin, zmax, num=500)

    ax = fig.add_subplot(nplots_x, nplots_y, 1)
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-50, 0)
    ax.plot(zc, T(p_log10phiStar)(1+zc), c='k', lw=2) 
    ax.set_ylabel(r'$\log_{10}\phi_*$')
    ax.set_xticklabels('')

    ax = fig.add_subplot(nplots_x, nplots_y, 2)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('right')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-150, -20)
    ax.plot(zc, T(p_MStar)(1+zc), c='k', lw=2) 
    ax.set_ylabel(r'$M_*$')
    ax.set_xticklabels('')

    ax = fig.add_subplot(nplots_x, nplots_y, 3)
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-8, -3)
    ax.plot(zc, T(p_alpha)(1+zc), c='k', lw=2) 
    ax.set_ylabel(r'$\alpha$')

    ax = fig.add_subplot(nplots_x, nplots_y, 4)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('right')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-2.6, -1.5)
    h, f0, z0, a, b = p_beta
    zeta = np.log10((1.0+zc)/(1.0+z0))
    beta = h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))
    ax.plot(zc, beta, c='k', lw=2) 
    ax.set_ylabel(r'$\beta$')

    plt.savefig('evolutionFC.pdf',bbox_inches='tight')

    mpl.rcParams['font.size'] = '22'

    return

plotRhoQso(0, 15)
# plotdRhoQsodz(0, 15)
