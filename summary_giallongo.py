import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '14'
import matplotlib.pyplot as plt

"""

Creates a plot showing the effect of Giallongo qsos on the individual
LFs.

"""

colors = ['tomato', 'forestgreen', 'goldenrod', 'saddlebrown'] 
nplots_x = 2
nplots_y = 2
nplots = 4
plot_number = 0 

zlims=(0.0,7.0)
zmin, zmax = zlims
z = np.linspace(zmin, zmax, num=50)

        
def plot_phi_star(fig, composite):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+1)
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-13, -5)

    if composite is not None: 
        bf = composite.getparams(composite.samples.mean(axis=0))
        for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
            params = composite.getparams(theta) 
            phi = composite.atz(z, params[0]) 
            ax.plot(z, phi, color=colors[0], alpha=0.02, zorder=1) 
        phi = composite.atz(z, bf[0]) 
        ax.plot(z, phi, color='k', zorder=2)

    zmean, zl, zu, u, l, c = np.loadtxt('Scratch/phi_star_withGiallongo.dat', unpack=True)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color=colors[0], edgecolor='None', zorder=2, s=36)
    ax.errorbar(zmean, c, ecolor=colors[0], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)

    zmean, zl, zu, u, l, c = np.loadtxt('Scratch/phi_star_withoutGiallongo.dat', unpack=True)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor='k', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)
    ax.scatter(zmean, c, color='#ffffff', edgecolor='k', zorder=2, s=30)
    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\log_{10}\left(\phi_*/\mathrm{mag}^{-1}\mathrm{cMpc}^{-3}\right)$')
    ax.set_xticklabels('')

    return

def plot_m_star(fig, composite):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+2)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('right')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-32, -22.0)

    if composite is not None: 
        bf = composite.getparams(composite.samples.mean(axis=0))
        for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
            params = composite.getparams(theta) 
            M = composite.atz(z, params[1]) 
            ax.plot(z, M, color=colors[1], alpha=0.02, zorder=3)
        M = composite.atz(z, bf[1]) 
        ax.plot(z, M, color='k', zorder=4)
    
    zmean, zl, zu, u, l, c = np.loadtxt('Scratch/M_star_withGiallongo.dat', unpack=True)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color=colors[1], edgecolor='None', zorder=2, s=36)
    ax.errorbar(zmean, c, ecolor=colors[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)

    zmean, zl, zu, u, l, c = np.loadtxt('Scratch/M_star_withoutGiallongo.dat', unpack=True)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor='k', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)
    ax.scatter(zmean, c, color='#ffffff', edgecolor='k', zorder=2, s=30)
    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$M_*$')
    ax.set_xticklabels('')

    return

def plot_alpha(fig, composite):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+3)
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-7, -2)

    if composite is not None: 
        bf = composite.getparams(composite.samples.mean(axis=0))
        for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
            params = composite.getparams(theta)
            alpha = composite.atz(z, params[2])
            ax.plot(z, alpha, color=colors[2], alpha=0.02, zorder=3) 
        alpha = composite.atz(z, bf[2]) 
        ax.plot(z, alpha, color='k', zorder=4)
    
    zmean, zl, zu, u, l, c = np.loadtxt('Scratch/alpha_withGiallongo.dat', unpack=True)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color=colors[2], edgecolor='None', zorder=2, s=36)
    ax.errorbar(zmean, c, ecolor=colors[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)

    zmean, zl, zu, u, l, c = np.loadtxt('Scratch/alpha_withoutGiallongo.dat', unpack=True)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor='k', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)
    ax.scatter(zmean, c, color='#ffffff', edgecolor='k', zorder=2, s=30)
    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\alpha$ (bright end slope)')
    ax.set_xlabel('$z$')

    return

def plot_beta(fig, composite):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+4)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('right')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-3, -1)

    if composite is not None: 
        bf = composite.getparams(composite.samples.mean(axis=0))
        for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
            params = composite.getparams(theta)
            beta = composite.atz(z, params[3]) 
            ax.plot(z, beta, color=colors[3], alpha=0.02, zorder=3) 
        beta = composite.atz(z, bf[3]) 
        ax.plot(z, beta, color='k', zorder=4)
    
    zmean, zl, zu, u, l, c = np.loadtxt('Scratch/beta_withGiallongo.dat', unpack=True)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color=colors[3], edgecolor='None', zorder=2, s=36)
    ax.errorbar(zmean, c, ecolor=colors[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)

    zmean, zl, zu, u, l, c = np.loadtxt('Scratch/beta_withoutGiallongo.dat', unpack=True)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor='k', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)
    ax.scatter(zmean, c, color='#ffffff', edgecolor='k', zorder=2, label='without Giallongo qsos', s=30)

    plt.legend(loc='lower left', fontsize=10, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=-0.4, borderpad=0.2, scatterpoints=1)

    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\beta$ (faint end slope)')
    ax.set_xlabel('$z$')

    return 

def summary_plot(composite=None):

    mpl.rcParams['font.size'] = '14'
    
    fig = plt.figure(figsize=(6, 6), dpi=100)

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

    plot_phi_star(fig, composite)
    plot_m_star(fig, composite)
    plot_alpha(fig, composite)
    plot_beta(fig, composite)

    plt.savefig('evolution.pdf',bbox_inches='tight')

    mpl.rcParams['font.size'] = '22'
    
    return

summary_plot()

