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

colors = ['k', 'k', 'k', 'k'] 
nplots_x = 2
nplots_y = 2
nplots = 4
plot_number = 0 

zlims=(0.0,7.0)
zmin, zmax = zlims
z = np.linspace(zmin, zmax, num=50)

def getParam(param, dtype='notwithg'):

    if dtype=='withg':
        zmean, zl, zu, u, l, c = np.loadtxt('bins_withg.dat',
                                            usecols=(0,1,2,3+param*3,4+param*3,5+param*3),
                                            unpack=True)
    else:
        zmean, zl, zu, u, l, c = np.loadtxt('bins.dat',
                                            usecols=(0,1,2,3+param*3,4+param*3,5+param*3),
                                            unpack=True)
    return zmean, zl, zu, u, l, c


def plot_phi_star(fig):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+1)

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=4, width=1, direction='in')
    ax.tick_params('both', which='minor', length=2, width=1, direction='in')
    
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-12, -5)

    zmean, zl, zu, u, l, c = getParam(0)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color=colors[0], edgecolor='None', zorder=2, s=36)
    ax.errorbar(zmean, c, ecolor=colors[0], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)

    zmean, zl, zu, u, l, c = getParam(0, dtype='withg')
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor='tomato', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)
    ax.scatter(zmean, c, color='#ffffff', edgecolor='tomato', zorder=2, s=30)
    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\log_{10}\left(\phi_*/\mathrm{mag}^{-1}\mathrm{cMpc}^{-3}\right)$')
    ax.set_xticklabels('')

    return

def plot_m_star(fig):

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

    zmean, zl, zu, u, l, c = getParam(1)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color=colors[1], edgecolor='None', zorder=2, s=36)
    ax.errorbar(zmean, c, ecolor=colors[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)

    zmean, zl, zu, u, l, c = getParam(1, dtype='withg')
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor='tomato', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)
    ax.scatter(zmean, c, color='#ffffff', edgecolor='tomato', zorder=2, s=30)
    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$M_*$')
    ax.set_xticklabels('')

    return

def plot_alpha(fig):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+3)

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=4, width=1, direction='in')
    ax.tick_params('both', which='minor', length=2, width=1, direction='in')
    
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-7, -1)

    zmean, zl, zu, u, l, c = getParam(2)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color=colors[2], edgecolor='None', zorder=2, s=36)
    ax.errorbar(zmean, c, ecolor=colors[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)

    zmean, zl, zu, u, l, c = getParam(2, dtype='withg')
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor='tomato', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)
    ax.scatter(zmean, c, color='#ffffff', edgecolor='tomato', zorder=2, s=30, label='with Giallongo et al.\ 2015 AGN')

    plt.legend(loc='upper right', fontsize=10, handlelength=2,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=-0.4, borderpad=0.1, scatterpoints=1, borderaxespad=0.3)
    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\alpha$ (bright end slope)')
    ax.set_xlabel('$z$')

    return

def plot_beta(fig):

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

    zmean, zl, zu, u, l, c = getParam(3)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color=colors[3], edgecolor='None', zorder=2, s=36)
    ax.errorbar(zmean, c, ecolor=colors[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)

    zmean, zl, zu, u, l, c = getParam(3, dtype='withg')
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.errorbar(zmean, c, ecolor='tomato', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2)
    ax.scatter(zmean, c, color='#ffffff', edgecolor='tomato', zorder=2, s=30)


    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\beta$ (faint end slope)')
    ax.set_xlabel('$z$')

    return 

def summary_plot():

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

    plot_phi_star(fig)
    plot_m_star(fig)
    plot_alpha(fig)
    plot_beta(fig)

    plt.savefig('evolution_g.pdf',bbox_inches='tight')

    mpl.rcParams['font.size'] = '22'
    
    return

summary_plot()

