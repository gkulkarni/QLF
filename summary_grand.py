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

clr = {'masters12': 'dodgerblue',
       'jiang16': 'dodgerblue',
       'glikman11': 'dodgerblue'}
colors = ['k','k','k','k']
nplots_x = 2
nplots_y = 2
nplots = 4
plot_number = 0 

zlims=(-0.4,7.0)
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


def ross13_s82(ax, param):

    # Ross using M_i not M1450. 

    data = np.loadtxt('Data_new/ross13_s82.txt')
    z = data[:,0]

    p = param

    rs, corr = np.loadtxt('Data_new/magconv_boss_kwh18.dat', usecols=(1,2), unpack=True)

    # Ross's alpha is our beta and vv. 
    if param == 3:
        p = 2

    if param == 2:
        p = 3

    c = data[:,3*p+1]
    l = data[:,3*p+2]
    u = data[:,3*p+3]

    if param == 1:
        mc = np.interp(z, rs, corr)
        c = c+mc
        l = l+mc
        u = u+mc 

    uperr = u-c
    downerr = c-l
    
    ax.errorbar(z, c, ecolor='dodgerblue', capsize=2,
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
    ax.scatter(z, c, color='dodgerblue', edgecolor='None', zorder=2, s=20, label='Ross et al.\ 2013 (S82)')

    return

def masters12(ax, param):

    z = np.array([3.2, 4.0])
    zlow = np.array([3.0, 3.5])
    zup = np.array([3.5, 5.0])

    zdownerr = z-zlow
    zuperr = zup-z

    if param == 0:
        # yerr on phi is nonsense in Masters 2012; gives negative
        # phistar.
        
        p = np.array([np.log10(2.65e-7), np.log10(7.5e-8)])

        ax.errorbar(z, p, ecolor=clr['masters12'], capsize=2,
                    xerr=np.vstack((zdownerr, zuperr)),
                    fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
        ax.scatter(z, p, color=clr['masters12'], edgecolor='None', zorder=2, s=20, label='Masters et al.\ 2012', marker='^')

        return

    if param == 1:
        p = np.array([-25.54, -25.64])
        perr = np.array([0.68, 2.99])

    if param == 2:
        p = np.array([-2.98, -2.60])
        perr = np.array([0.21, 0.63])

    if param == 3:
        p = np.array([-1.73, -1.72])
        perr = np.array([0.11, 0.28])

    ax.errorbar(z, p, ecolor=clr['masters12'], capsize=2,
                xerr=np.vstack((zdownerr, zuperr)),
                yerr=perr,
                fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
    ax.scatter(z, p, color=clr['masters12'], edgecolor='None', zorder=2, s=20, label='Masters et al.\ 2012', marker='^')

    return

def jiang16(ax, param):

    z = np.array([6.0])
    zlow = np.array([5.7])
    zup = np.array([6.4])

    zdownerr = z-zlow
    zuperr = zup-z

    if param == 0:
        # yerr on phi is nonsense in Masters 2012; gives negative
        # phistar.

        p = np.log10(9.93e-9)
        ax.errorbar(z, p, ecolor=clr['jiang16'], capsize=2,
                    xerr=np.vstack((zdownerr, zuperr)),
                    fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
        ax.scatter(z, p, color=clr['jiang16'], edgecolor='None', zorder=2, s=20, label='Jiang et al.\ 2016', marker='s')
        
        return

    if param == 1:
        p = np.array([-25.2])
        puperr = np.array([1.2])
        pdownerr = np.array([3.8])

        ax.errorbar(z, p, ecolor=clr['jiang16'], capsize=2,
                    xerr=np.vstack((zdownerr, zuperr)),
                    yerr=np.vstack((pdownerr, puperr)),
                    fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
        ax.scatter(z, p, color=clr['jiang16'], edgecolor='None', zorder=2, s=20, label='Jiang et al.\ 2016', marker='s')

        return 

    if param == 3:
        # Jiang's beta is our alpha and vice versa.
        p = np.array([-1.90])
        puperr = np.array([0.58])
        pdownerr = np.array([0.44])

        ax.errorbar(z, p, ecolor=clr['jiang16'], capsize=2,
                    xerr=np.vstack((zdownerr, zuperr)),
                    yerr=np.vstack((pdownerr, puperr)),
                    fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
        ax.scatter(z, p, color=clr['jiang16'], edgecolor='None', zorder=2, s=20, label='Jiang et al.\ 2016', marker='s')

        return

    if param == 2:
        p = np.array([-2.8])

    ax.errorbar(z, p, ecolor=clr['jiang16'], capsize=2,
                xerr=np.vstack((zdownerr, zuperr)),
                fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
    ax.scatter(z, p, color=clr['jiang16'], edgecolor='None', zorder=2, s=20, label='Jiang et al.\ 2016', marker='s')

    return

def glikman11(ax, param):

    z = np.array([4.15])
    zlow = np.array([3.8])
    zup = np.array([5.2])

    zdownerr = z-zlow
    zuperr = zup-z

    if param == 0:
        # yerr on phi is nonsense in Masters 2012; gives negative
        # phistar.
        p = np.array([np.log10(1.3e-6)])
        pup = np.array([np.log10(1.3e-6+1.8e-6)])
        pdown = np.array([np.log10(1.3e-6-0.2e-6)])
        
        puperr = pup-p
        pdownerr = p-pdown

    if param == 1:
        p = np.array([-24.1])
        puperr = np.array([0.7])
        pdownerr = np.array([1.9])

    if param == 2:
        p = np.array([-3.3])
        puperr = np.array([0.2])
        pdownerr = np.array([0.2])

    if param == 3:
        p = np.array([-1.6])
        puperr = np.array([0.11])
        pdownerr = np.array([0.17])

    ax.errorbar(z, p, ecolor=clr['glikman11'], capsize=2,
                xerr=np.vstack((zdownerr, zuperr)),
                yerr=np.vstack((pdownerr, puperr)),
                fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
    ax.scatter(z, p, color=clr['glikman11'], edgecolor='None', zorder=2, s=20, marker='D')

    return


def giallongo15(ax, param):

    z = np.array([4.25, 4.75, 5.75])
    zlow = np.array([4.0, 4.5, 5.0])
    zup = np.array([4.5, 5.0, 6.5])

    zdownerr = z-zlow
    zuperr = zup-z

    if param == 0:
        p = np.array([-5.2, -5.7, -5.8])

    if param == 1:
        p = np.array([-23.2, -23.6, -23.4]) 

    if param == 2:
        p = np.array([-3.13, -3.14, -3.35])

    if param == 3:
        p = np.array([-1.52, -1.81, -1.66])

    ax.errorbar(z, p, ecolor=clr['glikman11'], capsize=2,
                xerr=np.vstack((zdownerr, zuperr)),
                fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
    ax.scatter(z, p, color=clr['glikman11'], edgecolor='None', zorder=2, s=20, marker='v')

    return

def akiyama18(ax, param):

    z = np.array([3.9])
    zlow = np.array([3.5])
    zup = np.array([4.3])

    zdownerr = z-zlow
    zuperr = zup-z

    if param == 0:
        # yerr on phi is nonsense in Masters 2012; gives negative
        # phistar.
        p = np.array([np.log10(2.66e-7)])
        pup = np.array([np.log10(2.66e-7+0.05e-7)])
        pdown = np.array([np.log10(2.66e-7-0.05e-7)])
        
        puperr = pup-p
        pdownerr = p-pdown

        ax.errorbar(z, p, ecolor=clr['glikman11'], capsize=2,
                    xerr=np.vstack((zdownerr, zuperr)),
                    yerr=np.vstack((pdownerr, puperr)),
                    fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
        ax.scatter(z, p, color=clr['glikman11'], edgecolor='None', zorder=2, s=20, marker='p')
        return
        
    if param == 1:
        p = np.array([-25.36])
        perr = np.array([0.13])
        
    if param == 2:
        p = np.array([-3.11])
        perr = np.array([0.07])

    if param == 3:
        p = np.array([-1.3])
        perr = np.array([0.05])

    ax.errorbar(z, p, ecolor=clr['glikman11'], capsize=2,
                xerr=np.vstack((zdownerr, zuperr)),
                yerr=perr,
                fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
    ax.scatter(z, p, color=clr['glikman11'], edgecolor='None', zorder=2, s=20, marker='p')

    return

def onoue17(ax, param):

    z = np.array([6.0])
    zlow = np.array([5.5])
    zup = np.array([6.5])

    zdownerr = z-zlow
    zuperr = zup-z

    if param == 0:
        p = np.array([np.log10(4.06e-9)])

        ax.errorbar(z, p, ecolor=clr['glikman11'], capsize=2,
                    xerr=np.vstack((zdownerr, zuperr)),
                    fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
        ax.scatter(z, p, color=clr['glikman11'], edgecolor='None', zorder=2, s=20, marker='<')
        

        return
        
    if param == 1:
        p = np.array([-25.8])
        puperr = np.array([1.1])
        pdownerr = np.array([1.9])

    if param == 2:
        # This is called beta in Onoue17 and kept fixed.
        p = np.array([-2.8])

        ax.errorbar(z, p, ecolor=clr['glikman11'], capsize=2,
                    xerr=np.vstack((zdownerr, zuperr)),
                    fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
        ax.scatter(z, p, color=clr['glikman11'], edgecolor='None', zorder=2, s=20, marker='<')
        

        return
        
    if param == 3:
        # This is called alpha in Onoue17
        p = np.array([-1.63])
        puperr = np.array([1.21])
        pdownerr = np.array([1.09])
        
    ax.errorbar(z, p, ecolor=clr['glikman11'], capsize=2,
                xerr=np.vstack((zdownerr, zuperr)),
                yerr=np.vstack((pdownerr, puperr)),
                fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
    ax.scatter(z, p, color=clr['glikman11'], edgecolor='None', zorder=2, s=20, marker='<')

    return


def schulze09(ax, param):

    z = np.array([0.0])

    if param == 0:
        p = np.array([np.log10(1.55e-5)])

    if param == 1:
        p = np.array([-19.46+0.59])

    if param == 2:
        p = np.array([-2.82])

    if param == 3:
        p = np.array([-2.0])

    ax.errorbar(z, p, ecolor=clr['glikman11'], capsize=2,
                fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
    ax.scatter(z, p, color=clr['glikman11'], edgecolor='None', zorder=2, s=30, marker='>')

    return


def yang16(ax, param):

    z = np.array([5.05])
    zlow = np.array([4.7])
    zup = np.array([5.4])

    zdownerr = z-zlow
    zuperr = zup-z

    if param == 0:
        p = np.array([-9.07])
        perr = np.array([0.4])

    if param == 1:
        p = np.array([-27.32])
        perr = np.array([0.53])

    if param == 2:
        p = np.array([-3.8])
        perr = np.array([0.47])

    if param == 3:
        p = np.array([-2.14])
        perr = np.array([0.16])

    ax.errorbar(z, p, ecolor=clr['glikman11'], capsize=2,
                xerr=np.vstack((zdownerr, zuperr)),
                yerr=perr,
                fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
    ax.scatter(z, p, color=clr['glikman11'], edgecolor='None', zorder=2, s=40, marker='*')

    return

   
def mcgreer13(ax, param):

    z = np.array([4.9])
    zlow = np.array([4.7])
    zup = np.array([5.1])

    zdownerr = z-zlow
    zuperr = zup-z

    if param == 0:
        logphi = -8.94 -0.47*(z-6.0)
        p = np.array([logphi+np.log10(1.1)]) # cosmology correction by Gabor 
        puperr = np.array([0.20])
        pdownerr = np.array([0.24])

    if param == 1:
        p = np.array([-27.21+0.07]) # cosmology correction by Gabor 
        puperr = np.array([0.27])
        pdownerr = np.array([0.33])

    if param == 2:
        p = np.array([-4.0])
        ax.errorbar(z, p, ecolor=clr['glikman11'], capsize=2,
                    xerr=np.vstack((zdownerr, zuperr)),
                    fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
        ax.scatter(z, p, color=clr['glikman11'], edgecolor='None', zorder=2, s=20, marker='x')
        return
        
    if param == 3:
        p = np.array([-2.03])
        puperr = np.array([0.15])
        pdownerr = np.array([0.14])
        
    ax.errorbar(z, p, ecolor=clr['glikman11'], capsize=2,
                xerr=np.vstack((zdownerr, zuperr)),
                yerr=np.vstack((pdownerr, puperr)),
                fmt='None', zorder=2, linewidths=0.5, elinewidths=0.5)
    ax.scatter(z, p, color=clr['glikman11'], edgecolor='None', zorder=2, s=20, marker='x')

    return


def plot_phi_star(fig):

    mpl.rcParams['font.size'] = '14'

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+1)

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=4, width=1, direction='in')
    ax.tick_params('both', which='minor', length=2, width=1, direction='in')
    
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-12, -4)

    zmean, zl, zu, u, l, c = getParam(0)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color=colors[0], edgecolor='None', zorder=200, s=20)
    ax.errorbar(zmean, c, ecolor=colors[0], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=200)

    ross13_s82(ax, 0)
    masters12(ax, 0)
    jiang16(ax, 0)
    glikman11(ax, 0)
    giallongo15(ax, 0)
    akiyama18(ax, 0)
    onoue17(ax, 0)
    schulze09(ax, 0)
    yang16(ax, 0)
    mcgreer13(ax, 0)
    
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
    ax.set_ylim(-32, -17)

    zmean, zl, zu, u, l, c = getParam(1)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color=colors[1], edgecolor='None', zorder=200, s=20)
    ax.errorbar(zmean, c, ecolor=colors[1], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=200)

    ross13_s82(ax, 1)
    masters12(ax, 1)
    jiang16(ax, 1)
    glikman11(ax, 1)
    giallongo15(ax, 1)
    akiyama18(ax, 1)
    onoue17(ax, 1)
    schulze09(ax, 1)
    yang16(ax, 1)
    mcgreer13(ax, 1)
    
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
    ax.scatter(zmean, c, color=colors[2], edgecolor='None', zorder=200, s=20)
    ax.errorbar(zmean, c, ecolor=colors[2], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=200)

    ross13_s82(ax, 2)
    masters12(ax, 2)
    jiang16(ax, 2)
    glikman11(ax, 2)
    giallongo15(ax, 2)
    akiyama18(ax, 2)
    onoue17(ax, 2)
    schulze09(ax, 2)
    yang16(ax, 2)
    mcgreer13(ax, 2)

    # plt.legend(loc='upper right', fontsize=10, handlelength=2,
    #            frameon=False, framealpha=0.0, labelspacing=.1,
    #            handletextpad=-0.4, borderpad=0.1, scatterpoints=1, borderaxespad=0.3)
    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\alpha$ (bright-end slope)')
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
    ax.scatter(zmean, c, color=colors[3], edgecolor='None', zorder=200, s=20)
    ax.errorbar(zmean, c, ecolor=colors[3], capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((downerr, uperr)),
                fmt='None', zorder=200)

    ross13_s82(ax, 3)
    masters12(ax, 3)
    jiang16(ax, 3)
    glikman11(ax, 3)
    giallongo15(ax, 3)
    akiyama18(ax, 3)
    onoue17(ax, 3)
    schulze09(ax, 3)
    yang16(ax, 3)
    mcgreer13(ax, 3)
    
    ax.set_xticks((0,1,2,3,4,5,6,7))
    ax.set_ylabel(r'$\beta$ (faint-end slope)')
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

    plt.savefig('evolution_grand.pdf',bbox_inches='tight')

    mpl.rcParams['font.size'] = '22'
    
    return

summary_plot()

