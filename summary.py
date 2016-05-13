import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '14'
import matplotlib.pyplot as plt

colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a'] 
nplots_x = 2
nplots_y = 2
nplots = 4
plot_number = 0 

zlims=(2.0,6.0)
zmin, zmax = zlims
z = np.linspace(zmin, zmax, num=50)

def plot_phi_star(fig, composite):

    bf = composite.getparams(composite.samples.mean(axis=0))

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+1)
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-6.5, -5.0)

    for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
        params = composite.getparams(theta) 
        phi = composite.atz(z, params[0]) 
        ax.plot(z, phi, color=colors[0], alpha=0.02, zorder=1) 
    phi = composite.atz(z, bf[0]) 
    ax.plot(z, phi, color='k', zorder=2)

    ax.set_xticks((2,3,4,5,6))
    ax.set_ylabel(r'$\log_{10}(\phi_*/\mathrm{mag}^{-1}\mathrm{cMpc}^{-3})$')
    ax.set_xticklabels('')

    return

def plot_m_star(fig, composite):

    bf = composite.getparams(composite.samples.mean(axis=0))

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+2)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('right')
    ax.set_xlim(zmin, zmax)
    ax.set_ylim(-26.0, -20.0)

    for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
        params = composite.getparams(theta) 
        M = composite.atz(z, params[1]) 
        ax.plot(z, M, color=colors[1], alpha=0.02, zorder=3)
    M = composite.atz(z, bf[1]) 
    ax.plot(z, M, color='k', zorder=4)

    ax.set_xticks((2,3,4,5,6))
    ax.set_ylabel(r'$M_*$')
    ax.set_xticklabels('')

    return

def plot_alpha(fig, composite):

    bf = composite.getparams(composite.samples.mean(axis=0))

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+3)
    ax.set_xlim(zmin, zmax)
    #ax.set_ylim(-3.3, -2.3)

    for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
        params = composite.getparams(theta)
        alpha = composite.atz(z, params[2])
        ax.plot(z, alpha, color=colors[2], alpha=0.02, zorder=3) 
    alpha = composite.atz(z, bf[2]) 
    ax.plot(z, alpha, color='k', zorder=4)

    ax.set_xticks((2,3,4,5,6))
    ax.set_ylabel(r'$\alpha$')
    ax.set_xlabel('$z$')

    return

def plot_beta(fig, composite):

    bf = composite.getparams(composite.samples.mean(axis=0))

    ax = fig.add_subplot(nplots_x, nplots_y, plot_number+4)
    ax.yaxis.tick_right()
    ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_label_position('right')
    ax.set_xlim(zmin, zmax)
    #ax.set_ylim(-2.0, 2.0)

    if composite is not None: 
        for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
            params = composite.getparams(theta)
            beta = composite.atz(z, params[3]) 
            ax.plot(z, beta, color=colors[3], alpha=0.02, zorder=3) 
        beta = composite.atz(z, bf[3]) 
        ax.plot(z, beta, color='k', zorder=4)

    ax.set_xticks((2,3,4,5,6))
    ax.set_ylabel(r'$\beta$')
    ax.set_xlabel('$z$')

    return 

def summary_plot(composite, zlims=(2.0,6.0), dirname=''):

    fig = plt.figure(figsize=(6, 6), dpi=100)

    K = 4
    factor = 2.0           # size of one side of one panel
    lbdim = 0.5 * factor   # size of left/bottom margin
    trdim = 0.2 * factor   # size of top/right margin
    whspace = 0.05         # w/hspace size
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

    plt.savefig(dirname+'evolution.pdf',bbox_inches='tight')
    return


