import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '14'
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit
from numpy.polynomial import Chebyshev as T
from scipy.optimize import curve_fit

colors = ['tomato', 'forestgreen', 'goldenrod', 'saddlebrown'] 
nplots_x = 2
nplots_y = 2
nplots = 4
plot_number = 0 

zlims=(0.0,7.0)
zmin, zmax = zlims
z = np.linspace(zmin, zmax, num=50)

def plot_phi_star(fig, composite, sample=False, lfs=None, lfsMock=None):

    
    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+1)
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-13, -4)

    if composite is not None: 
        bf = np.mean(composite.samples, axis=0)
        if sample:
            for theta in composite.samples[np.random.randint(len(composite.samples),
                                                             size=500)]:
                params = composite.getparams(theta)
                phi = composite.atz(z, params[0]) 
                ax.plot(z, phi, color='k', alpha=0.02, zorder=1) 
        phi = composite.atz(z, composite.getparams(bf)[0])
        ax.plot(z, phi, color='k', zorder=2, lw=2, label=r'composite fit to $z > 3.7$ qsos')

    if lfs is not None:
        zmean = np.array([x.z.mean() for x in lfs])
        zl = np.array([x.zlims[0] for x in lfs])
        zu = np.array([x.zlims[1] for x in lfs])
        u = np.array([x.phi_star[0] for x in lfs])
        l = np.array([x.phi_star[1] for x in lfs])
        c = np.array([x.phi_star[2] for x in lfs])
    else:
        zmean, zl, zu, u, l, c = np.loadtxt('phi_star.dat', unpack=True)

    select = zmean > 3.6
    zl = zl[select]
    zu = zu[select]
    u = u[select]
    l = l[select]
    c = c[select]
    zmean = zmean[select]

    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color='k', edgecolor='None', zorder=5, s=36, label='individual fits')
    ax.errorbar(zmean, c, ecolor='k', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=5)

    zc = np.linspace(0, 7, 500)
    coeffs = chebfit(zmean+1, c, 1)

    def func(z, p0, p1):
        return T([p0, p1], domain=[3.7,7.])(z)

    sigma = uperr + downerr 
    popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
    print popt
    plt.plot(zc, func(zc+1, *popt), lw=2, c='r', dashes=[8,3], label='linear fit to black points')

    zm, cm, uperr, downerr = np.loadtxt('Data/manti.txt',
                                        usecols=(0,1,2,3), unpack=True)
    ax.errorbar(zm, cm, ecolor='grey', capsize=0,
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zm, cm, color='#ffffff', edgecolor='grey', zorder=4, s=30, label='Manti et al.\ 2017')

    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\log_{10}\left(\phi_*/\mathrm{mag}^{-1}\mathrm{cMpc}^{-3}\right)$')
    ax.set_xticklabels('')

    if lfsMock is not None:
        zmean = np.array([x.z.mean() for x in lfsMock])
        zl = np.array([x.zlims[0] for x in lfsMock])
        zu = np.array([x.zlims[1] for x in lfsMock])
        u = np.array([x.phi_star[0] for x in lfsMock])
        l = np.array([x.phi_star[1] for x in lfsMock])
        c = np.array([x.phi_star[2] for x in lfsMock])

        select = zmean > 3.6
        zl = zl[select]
        zu = zu[select]
        u = u[select]
        l = l[select]
        c = c[select]
        zmean = zmean[select]

        left = zmean-zl
        right = zu-zmean
        uperr = u-c
        downerr = c-l
        ax.scatter(zmean, c, color='cornflowerblue', edgecolor='None', zorder=5, s=36, label='mock data')
        ax.errorbar(zmean, c, ecolor='cornflowerblue', capsize=0,
                    xerr=np.vstack((left, right)), 
                    yerr=np.vstack((uperr, downerr)),
                    fmt='None', zorder=5)
    
    plt.legend(loc='lower left', fontsize=8, handlelength=2.5,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01, scatterpoints=1)

    return

def plot_m_star(fig, composite, sample=False, lfs=None, lfsMock=None):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+2)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('right')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-32, -21)

    if composite is not None: 
        bf = np.mean(composite.samples, axis=0)
        if sample:
            for theta in composite.samples[np.random.randint(len(composite.samples), size=500)]:
                params = composite.getparams(theta) 
                M = composite.atz(z, params[1]) 
                ax.plot(z, M, color='k', alpha=0.02, zorder=1)
        M = composite.atz(z, composite.getparams(bf)[1])
        ax.plot(z, M, color='k', zorder=2, lw=2)

    if lfs is not None:
        zmean = np.array([x.z.mean() for x in lfs])
        zl = np.array([x.zlims[0] for x in lfs])
        zu = np.array([x.zlims[1] for x in lfs])
        u = np.array([x.M_star[0] for x in lfs])
        l = np.array([x.M_star[1] for x in lfs])
        c = np.array([x.M_star[2] for x in lfs])
    else:
        zmean, zl, zu, u, l, c = np.loadtxt('M_star.dat', unpack=True)
        
    select = zmean > 3.6
    zl = zl[select]
    zu = zu[select]
    u = u[select]
    l = l[select]
    c = c[select]
    zmean = zmean[select]
    
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color='k', edgecolor='None', zorder=5, s=36)
    ax.errorbar(zmean, c, ecolor='k', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=5)

    zc = np.linspace(0, 7, 500)

    coeffs = chebfit(zmean+1, c, 1)

    def func(z, p0, p1):
        return T([p0, p1], domain=[3.7,7.])(z)

    sigma = np.abs(uperr + downerr)
    popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
    print popt
    plt.plot(zc, func(zc+1, *popt), lw=2, c='r', dashes=[8,3])

    zm, cm, uperr, downerr = np.loadtxt('Data/manti.txt',
                                        usecols=(0,4,5,6), unpack=True)
    ax.errorbar(zm, cm, ecolor='grey', capsize=0,
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zm, cm, color='#ffffff', edgecolor='grey', zorder=4, s=30)
        
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$M_*$')
    ax.set_xticklabels('')

    if lfsMock is not None:
        zmean = np.array([x.z.mean() for x in lfsMock])
        zl = np.array([x.zlims[0] for x in lfsMock])
        zu = np.array([x.zlims[1] for x in lfsMock])
        u = np.array([x.M_star[0] for x in lfsMock])
        l = np.array([x.M_star[1] for x in lfsMock])
        c = np.array([x.M_star[2] for x in lfsMock])

        select = zmean > 3.6
        zl = zl[select]
        zu = zu[select]
        u = u[select]
        l = l[select]
        c = c[select]
        zmean = zmean[select]

        left = zmean-zl
        right = zu-zmean
        uperr = u-c
        downerr = c-l
        ax.scatter(zmean, c, color='cornflowerblue', edgecolor='None', zorder=5, s=36, label='mock data')
        ax.errorbar(zmean, c, ecolor='cornflowerblue', capsize=0,
                    xerr=np.vstack((left, right)), 
                    yerr=np.vstack((uperr, downerr)),
                    fmt='None', zorder=5)
    

    return

def plot_alpha(fig, composite, sample=False, lfs=None, lfsMock=None):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+3)
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-7, -1)

    if composite is not None: 
        bf = np.mean(composite.samples, axis=0)
        if sample:
            for theta in composite.samples[np.random.randint(len(composite.samples), size=500)]:
                params = composite.getparams(theta)
                alpha = composite.atz(z, params[2])
                ax.plot(z, alpha, color='k', alpha=0.02, zorder=1) 
        alpha = composite.atz(z, composite.getparams(bf)[2])
        ax.plot(z, alpha, color='k', zorder=2, lw=2)

    if lfs is not None:
        zmean = np.array([x.z.mean() for x in lfs])
        zl = np.array([x.zlims[0] for x in lfs])
        zu = np.array([x.zlims[1] for x in lfs])
        u = np.array([x.alpha[0] for x in lfs])
        l = np.array([x.alpha[1] for x in lfs])
        c = np.array([x.alpha[2] for x in lfs])
    else:
        zmean, zl, zu, u, l, c = np.loadtxt('alpha.dat', unpack=True)
        
    select = zmean > 3.6
    zl = zl[select]
    zu = zu[select]
    u = u[select]
    l = l[select]
    c = c[select]
    zmean = zmean[select]
    
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color='k', edgecolor='None', zorder=5, s=36)
    ax.errorbar(zmean, c, ecolor='k', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=5)
    
    zc = np.linspace(0, 7, 500)

    coeffs = chebfit(zmean+1.0, c, 1)

    def func(z, p0, p1):
        return T([p0, p1], domain=[3.7,7.])(z)

    sigma = uperr + downerr
    popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
    print popt
    plt.plot(zc, func(zc+1, *popt), lw=2, c='r', dashes=[8,3])

    zm, cm, uperr, downerr = np.loadtxt('Data/manti.txt',
                                        usecols=(0,10,11,12), unpack=True)
    ax.errorbar(zm, cm, ecolor='grey', capsize=0,
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zm, cm, color='#ffffff', edgecolor='grey',
               zorder=4, label='Manti et al.\ 2017', s=30)

    # plt.legend(loc='upper left', fontsize=10, handlelength=1,
    #            frameon=False, framealpha=0.0, labelspacing=.1,
    #            handletextpad=0.1, borderpad=0.01, scatterpoints=1)

    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\alpha$ (bright-end slope)')
    ax.set_xlabel('$z$')

    if lfsMock is not None:
        zmean = np.array([x.z.mean() for x in lfsMock])
        zl = np.array([x.zlims[0] for x in lfsMock])
        zu = np.array([x.zlims[1] for x in lfsMock])
        u = np.array([x.alpha[0] for x in lfsMock])
        l = np.array([x.alpha[1] for x in lfsMock])
        c = np.array([x.alpha[2] for x in lfsMock])

        select = zmean > 3.6
        zl = zl[select]
        zu = zu[select]
        u = u[select]
        l = l[select]
        c = c[select]
        zmean = zmean[select]

        left = zmean-zl
        right = zu-zmean
        uperr = u-c
        downerr = c-l
        ax.scatter(zmean, c, color='cornflowerblue', edgecolor='None', zorder=5, s=36, label='mock data')
        ax.errorbar(zmean, c, ecolor='cornflowerblue', capsize=0,
                    xerr=np.vstack((left, right)), 
                    yerr=np.vstack((uperr, downerr)),
                    fmt='None', zorder=5)
    

    return

def plot_beta(fig, composite, sample=False, lfs=None, lfsMock=None):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+4)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('right')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-3, 0)

    if composite is not None: 
        bf = np.mean(composite.samples, axis=0)
        if sample: 
            for theta in composite.samples[np.random.randint(len(composite.samples), size=500)]:
                params = composite.getparams(theta)
                beta = composite.atz(z, params[3]) 
                ax.plot(z, beta, color='k', alpha=0.01, zorder=1) 
        beta = composite.atz(z, composite.getparams(bf)[3])
        ax.plot(z, beta, color='k', zorder=2, lw=2)

    if lfs is not None:
        zmean = np.array([x.z.mean() for x in lfs])
        zl = np.array([x.zlims[0] for x in lfs])
        zu = np.array([x.zlims[1] for x in lfs])
        u = np.array([x.beta[0] for x in lfs])
        l = np.array([x.beta[1] for x in lfs])
        c = np.array([x.beta[2] for x in lfs])
    else:
        zmean, zl, zu, u, l, c = np.loadtxt('beta.dat', unpack=True)
        
    select = zmean > 3.6
    zl = zl[select]
    zu = zu[select]
    u = u[select]
    l = l[select]
    c = c[select]
    zmean = zmean[select]
    
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color='k', edgecolor='None', zorder=5, s=36)
    ax.errorbar(zmean, c, ecolor='k', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=5)

    zc = np.linspace(0, 7, 500)
    coeffs = chebfit(zmean+1, c, 1)

    def func(z, p0, p1):
        return T([p0, p1], domain=[3.7,7.])(z)

    sigma = uperr + downerr 
    popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
    print popt
    plt.plot(zc, func(zc+1, *popt), lw=2, c='r', dashes=[8,3])

    zm, cm, uperr, downerr = np.loadtxt('Data/manti.txt',
                                        usecols=(0,7,8,9), unpack=True)
    ax.errorbar(zm, cm, ecolor='grey', capsize=0,
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)
    ax.scatter(zm, cm, color='#ffffff', edgecolor='grey', zorder=4, s=30)

    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\beta$ (faint-end slope)')
    ax.set_xlabel('$z$')

    if lfsMock is not None:
        zmean = np.array([x.z.mean() for x in lfsMock])
        zl = np.array([x.zlims[0] for x in lfsMock])
        zu = np.array([x.zlims[1] for x in lfsMock])
        u = np.array([x.beta[0] for x in lfsMock])
        l = np.array([x.beta[1] for x in lfsMock])
        c = np.array([x.beta[2] for x in lfsMock])

        select = zmean > 3.6
        zl = zl[select]
        zu = zu[select]
        u = u[select]
        l = l[select]
        c = c[select]
        zmean = zmean[select]

        left = zmean-zl
        right = zu-zmean
        uperr = u-c
        downerr = c-l
        ax.scatter(zmean, c, color='cornflowerblue', edgecolor='None', zorder=5, s=36, label='mock data')
        ax.errorbar(zmean, c, ecolor='cornflowerblue', capsize=0,
                    xerr=np.vstack((left, right)), 
                    yerr=np.vstack((uperr, downerr)),
                    fmt='None', zorder=5)
    

    return 

def summary_plot(composite=None, sample=False, lfs=None, lfsMock=None):

    mpl.rcParams['font.size'] = '14'
    
    fig = plt.figure(figsize=(6, 6), dpi=100)

    print 'laying out figure'

    K = 4
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

    print 'plotting now'
    
    plot_phi_star(fig, composite, sample=sample, lfs=lfs, lfsMock=lfsMock)
    plot_m_star(fig, composite, sample=sample, lfs=lfs, lfsMock=lfsMock)
    plot_alpha(fig, composite, sample=sample, lfs=lfs, lfsMock=lfsMock)
    plot_beta(fig, composite, sample=sample, lfs=lfs, lfsMock=lfsMock)

    plt.savefig('evolution.pdf',bbox_inches='tight')
    plt.close('all')

    mpl.rcParams['font.size'] = '22'
    
    return

# summary_plot()

