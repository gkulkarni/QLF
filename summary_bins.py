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
from scipy.interpolate import UnivariateSpline

# These redshift bins are labelled "bad" and are plotted differently.
reject = [0, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

colors = ['tomato', 'forestgreen', 'goldenrod', 'saddlebrown'] 
nplots_x = 2
nplots_y = 2
nplots = 4
plot_number = 0 

zlims=(0.0,7.0)
zmin, zmax = zlims
z = np.linspace(zmin, zmax, num=50)
cfit = False

show_bad_data_differently = True

def getParam(individuals, param, which='old', dtype='good'):


    if individuals is not None: 
    
        zmean = np.array([x.z.mean() for x in individuals])
        zl = np.array([x.zlims[0] for x in individuals])
        zu = np.array([x.zlims[1] for x in individuals])

        if param == 0: 
            u = np.array([x.phi_star[0] for x in individuals])
            l = np.array([x.phi_star[1] for x in individuals])
            c = np.array([x.phi_star[2] for x in individuals])
        elif param == 1:
            u = np.array([x.M_star[0] for x in individuals])
            l = np.array([x.M_star[1] for x in individuals])
            c = np.array([x.M_star[2] for x in individuals])
        elif param == 2:
            u = np.array([x.alpha[0] for x in individuals])
            l = np.array([x.alpha[1] for x in individuals])
            c = np.array([x.alpha[2] for x in individuals])
        elif param == 3:
            u = np.array([x.beta[0] for x in individuals])
            l = np.array([x.beta[1] for x in individuals])
            c = np.array([x.beta[2] for x in individuals])

    else:

        zmean, zl, zu, u, l, c = np.loadtxt('bins.dat', usecols=(0,1,2,3+param*3,4+param*3,5+param*3), unpack=True)

    m = np.ones_like(zmean, dtype=bool)
    m[reject] = False
    minv = np.logical_not(m) 

    if dtype == 'good': 
        return zmean[m], zl[m], zu[m], u[m], l[m], c[m]
    else:
        return zmean[minv], zl[minv], zu[minv], u[minv], l[minv], c[minv]

        
def plot_phi_star(fig, composite, individuals=None, compOpt=None, sample=False):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+1)

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=4, width=1, direction='in')
    ax.tick_params('both', which='minor', length=2, width=1, direction='in')
    
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-12, -5)
    ax.set_yticks(np.arange(-12, -4, 1))

    if compOpt is not None: 
        phi = compOpt.atz(z, compOpt.getparams(compOpt.bf.x)[0])
        ax.plot(z, phi, color='g', zorder=2, dashes=[7,2])

    if composite is not None: 
        bf = np.median(composite.samples, axis=0)
        if sample:
            for theta in composite.samples[np.random.randint(len(composite.samples),
                                                             size=900)]:
                params = composite.getparams(theta)
                phi = composite.atz(z, params[0]) 
                ax.plot(z, phi, color=colors[0], alpha=0.02, zorder=1) 
        phi = composite.atz(z, composite.getparams(bf)[0])
        ax.plot(z, phi, color='k', zorder=2, lw=2)

    zmean, zl, zu, u, l, c = getParam(individuals, 0, which='new', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor=colors[0], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[0], edgecolor='None', zorder=6, s=40)

    zmean, zl, zu, u, l, c = getParam(individuals, 0, which='new', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor=colors[0], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    if show_bad_data_differently: 
        ax.scatter(zmean, c, color='#ffffff', edgecolor=colors[0], zorder=6, s=35)
    else:
        ax.scatter(zmean, c, color=colors[0], edgecolor='None', zorder=6, s=40)
        
    if cfit:
        zc = np.linspace(0, 7, 500)
        coeffs = chebfit(zmean+1, c, 2)
        print coeffs 
        # plt.plot(zc, T(coeffs)(zc+1), lw=1, c='k', dashes=[7,2],
        #          label='Least-square Chebyshev French curve', zorder=3)

        def func(z, p0, p1, p2):
            return T([p0, p1, p2])(z)

        sigma = uperr + downerr 
        popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
        print popt
        plt.plot(zc, func(zc+1, *popt), lw=1, c='k', dashes=[7,2])

    curvefit = False
    if curvefit:
        zc = np.linspace(0, 7, 500)
        
        def func(z, h, f0, z0, a, b):
            zeta = np.log10((1.0+z)/(1.0+z0))
            return h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))

        sigma = u - l 
        popt, pcov = curve_fit(func, zmean, c, sigma=sigma, p0=[-12.2, 6.6, 4.6, 4.9, -0.1])
        print popt
        plt.plot(zc, func(zc, *popt), lw=1, c='r', dashes=[7,2])

    # zm, cm, uperr, downerr = np.loadtxt('Data/manti.txt',
    #                                     usecols=(0,1,2,3), unpack=True)
    # ax.errorbar(zm, cm, ecolor='grey', capsize=0,
    #             yerr=np.vstack((downerr, uperr)),
    #             fmt='None', zorder=4)
    # ax.scatter(zm, cm, color='#ffffff', edgecolor='grey', zorder=4, s=30)

    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\log_{10}\left(\phi_*/\mathrm{mag}^{-1}'+
                  r'\mathrm{cMpc}^{-3}\right)$')
    ax.yaxis.labelpad = 8
    ax.set_xticklabels('')

    return

def plot_m_star(fig, composite, individuals=None, compOpt=None, sample=False):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+2)

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=4, width=1, direction='in')
    ax.tick_params('both', which='minor', length=2, width=1, direction='in')
    
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('right')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-32, -20)
    ax.set_yticks(np.arange(-32, -19, 2))

    if compOpt is not None:
        M = compOpt.atz(z, compOpt.getparams(compOpt.bf.x)[1])
        print M
        ax.plot(z, M, color='g', zorder=2, dashes=[7,2])

    if composite is not None:
        bf = np.median(composite.samples, axis=0)
        if sample:
            for theta in composite.samples[np.random.randint(
                    len(composite.samples), size=900)]:
                params = composite.getparams(theta) 
                M = composite.atz(z, params[1]) 
                ax.plot(z, M, color='darkgrey', zorder=1)
        M = composite.atz(z, composite.getparams(bf)[1])
        ax.plot(z, M, color='k', zorder=2, lw=2)

    zmean, zl, zu, u, l, c = getParam(individuals, 1, which='new', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor=colors[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[1], edgecolor='None', zorder=6, s=40)

    # zm, cm, uperr, downerr = np.loadtxt('Data/manti.txt',
    #                                     usecols=(0,4,5,6), unpack=True)
    # ax.errorbar(zm, cm, ecolor='grey', capsize=0,
    #             yerr=np.vstack((downerr, uperr)),
    #             fmt='None', zorder=4)
    # ax.scatter(zm, cm, color='#ffffff', edgecolor='grey', zorder=4, s=30)

    cfit = False
    if cfit:
        zc = np.linspace(0, 7, 500)
        coeffs = chebfit(zmean+1, c, 1)
        print 'cm=', coeffs
        # plt.plot(zc, T(coeffs)(zc+1), lw=1, c='k', dashes=[7,2], zorder=3) 

        def func(z, p0, p1):
            return T([p0, p1])(z)

        sigma = np.abs(u-l)
        popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
        print popt
        plt.plot(zc, func(zc+1, *popt), lw=1, c='k', dashes=[7,2])

    zmean, zl, zu, u, l, c = getParam(individuals, 1, which='new', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor=colors[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)

    if show_bad_data_differently: 
        ax.scatter(zmean, c, color='#ffffff', edgecolor=colors[1], zorder=6, s=35)
    else:
        ax.scatter(zmean, c, color=colors[1], edgecolor='None', zorder=6, s=40)

    curvefit = False
    if curvefit:
        zc = np.linspace(0, 7, 500)
        
        # def func(z, h, f0, z0, a, b):
        #     zeta = np.log10((1.0+z)/(1.0+z0))
        #     return h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))

        # def func(z, p0, p1, p2):
        #     zeta = np.log10((1.0+z)/(1.0+3.5))
        #     return T([p0, p1, p2])(zeta)

        def func(z, p0, p1, p2):
            zeta = np.log10((1.0+z)/(1.0+3.5))
            return T([p0, p1, p2])(1+z)

        sigma = u - l 
        popt, pcov = curve_fit(func, zmean, c, sigma=sigma, p0=[-22.,1,1])
        print popt
        plt.plot(zc, func(zc, *popt), lw=1, c='r', dashes=[7,2])
        
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$M_*$')
    ax.yaxis.labelpad = 12
    ax.set_xticklabels('')

    return

def plot_alpha(fig, composite, individuals=None, compOpt=None, sample=False):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+3)

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=4, width=1, direction='in')
    ax.tick_params('both', which='minor', length=2, width=1, direction='in')
    
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-7, -1)
    ax.set_yticks(np.arange(-7, -0.9, 1))


    if compOpt is not None: 
        alpha = compOpt.atz(z, compOpt.getparams(compOpt.bf.x)[2])
        ax.plot(z, alpha, color='g', zorder=2, dashes=[7,2],
                label='likelihood maximum')

    if composite is not None:
        bf = np.median(composite.samples, axis=0)
        if sample:
            for theta in composite.samples[np.random.randint(
                    len(composite.samples), size=900)]:
                params = composite.getparams(theta)
                alpha = composite.atz(z, params[2])
                ax.plot(z, alpha, color=colors[2], alpha=0.02, zorder=1) 
        alpha = composite.atz(z, composite.getparams(bf)[2])
        ax.plot(z, alpha, color='k', zorder=2, lw=2)

    zmean, zl, zu, u, l, c = getParam(individuals, 2, which='new', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor=colors[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[2], edgecolor='None', zorder=6, s=40)

    zmean, zl, zu, u, l, c = getParam(individuals, 2, which='new', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor=colors[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)

    if show_bad_data_differently: 
        ax.scatter(zmean, c, color='#ffffff', edgecolor=colors[2], zorder=6, s=35)
    else:
        ax.scatter(zmean, c, color=colors[2], edgecolor='None', zorder=6, s=40)
    
    cfit = False
    if cfit: 
        zc = np.linspace(0, 7, 500)
        coeffs = chebfit(zmean+1.0, c, 2)
        print 'c=', coeffs

        def func(z, p0, p1, p2):
            return T([p0, p1, p2])(z)

        sigma = u-l
        popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
        print popt
        plt.plot(zc, func(zc+1, *popt), lw=1, c='k', dashes=[7,2],
                 label=r'French curve')

    curvefit = False
    if curvefit:
        zc = np.linspace(0, 7, 500)
        
        def func(z, h, f0, z0, a, b):
            zeta = np.log10((1.0+z)/(1.0+z0))
            return h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))

        sigma = u - l 
        popt, pcov = curve_fit(func, zmean, c, sigma=sigma, p0=[-4,4.2,2.0,1.4,-0.7])
        print popt
        plt.plot(zc, func(zc, *popt), lw=1, c='r', dashes=[7,2])
        
        
    plt.legend(loc='upper left', fontsize=10, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01, scatterpoints=1)

    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\alpha$ (bright-end slope)')
    ax.set_xlabel('$z$')

    return

def plot_beta(fig, composite, individuals=None, compOpt=None, sample=False):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+4)

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=4, width=1, direction='in')
    ax.tick_params('both', which='minor', length=2, width=1, direction='in')
    
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('right')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-3, 0)
    ax.set_yticks(np.arange(-3, 0.2, 0.5))
    
    if compOpt is not None:
        beta = compOpt.atz_beta(z, compOpt.getparams(compOpt.bf.x)[3])
        ax.plot(z, beta, color='g', zorder=2, dashes=[7,2])

    if composite is not None:
        bf = np.median(composite.samples, axis=0)
        if sample: 
            for theta in composite.samples[np.random.randint(
                    len(composite.samples), size=900)]:
                params = composite.getparams(theta)
                beta = composite.atz_beta(z, params[3]) 
                ax.plot(z, beta, color=colors[3], alpha=0.02, zorder=1) 
        beta = composite.atz_beta(z, composite.getparams(bf)[3])
        ax.plot(z, beta, color='k', zorder=2, lw=2)

    zmean, zl, zu, u, l, c = getParam(individuals, 3, which='new', dtype='good')
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor=colors[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    ax.scatter(zmean, c, color=colors[3], edgecolor='None', zorder=6, s=40)

    cfit = False
    if cfit:
        zc = np.linspace(0, 7, 500)
        coeffs = chebfit(zmean+1, c, 3)
        print coeffs
        # plt.plot(zc, T(coeffs)(zc+1), lw=1, c='k', dashes=[7,2], zorder=3)

        def func(z, p0, p1, p2, p3):
                return T([p0, p1, p2, p3])(z)

        sigma = u - l 
        popt, pcov = curve_fit(func, zmean+1, c, sigma=sigma, p0=[coeffs])
        print 'cb=', popt
        plt.plot(zc, func(zc+1, *popt), lw=1, c='r', dashes=[7,2])

    polyfit = False
    if polyfit:
        zc = np.linspace(0, 7, 500)
        p = np.polyfit(np.log10(zmean+10), c, 2)
        print p
        # plt.plot(zc, np.polyval(p, np.log10((zc+1))), lw=1, c='k', dashes=[7,2], zorder=3)
        plt.plot(zc, np.polyval(p, np.log10((zc+10))), lw=1, c='k', dashes=[7,2], zorder=3)

    curvefit = False
    if curvefit:
        zc = np.linspace(0, 7, 500)
        
        def func(z, h, f0, z0, a, b):
            zeta = np.log10((1.0+z)/(1.0+z0))
            return h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))

        sigma = u - l 
        popt, pcov = curve_fit(func, zmean, c, sigma=sigma, p0=[-4,4.2,2.0,1.4,-0.7])
        print popt
        plt.plot(zc, func(zc, *popt), lw=1, c='k', dashes=[7,2])

    zmean, zl, zu, u, l, c = getParam(individuals, 3, which='new', dtype='bad')
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor=colors[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=6)
    if show_bad_data_differently: 
        ax.scatter(zmean, c, color='#ffffff', edgecolor=colors[3], zorder=6, s=35)
    else:
        ax.scatter(zmean, c, color=colors[3], edgecolor='None', zorder=6, s=40)

    # zm, cm, uperr, downerr = np.loadtxt('Data/manti.txt',
    #                                     usecols=(0,7,8,9), unpack=True)
    # ax.errorbar(zm, cm, ecolor='grey', capsize=0,
    #             yerr=np.vstack((downerr, uperr)),
    #             fmt='None', zorder=4)
    # ax.scatter(zm, cm, color='#ffffff', edgecolor='grey', zorder=4, s=30)
    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\beta$ (faint-end slope)')
    ax.set_xlabel('$z$')

    return 

def summary_plot(composite=None, individuals=None, compOpt=None, sample=False):

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
    
    plot_phi_star(fig, composite, individuals=individuals, compOpt=compOpt, sample=sample)
    plot_m_star(fig, composite, individuals=individuals, compOpt=compOpt, sample=sample)
    plot_alpha(fig, composite, individuals=individuals, compOpt=compOpt, sample=sample)
    plot_beta(fig, composite, individuals=individuals, compOpt=compOpt, sample=sample)

    plt.savefig('evolution_individuals.pdf',bbox_inches='tight')

    mpl.rcParams['font.size'] = '22'
    
    return

# summary_plot()

