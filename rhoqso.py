import numpy as np 
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt

def f(loglf, theta, m, z, fit='individual'):

    if fit == 'composite':
        return 10.0**loglf(theta, m, z)
    
    return 10.0**loglf(theta, m)

def rhoqso(loglf, theta, mlim, z, fit='individual', mbright=-35.0):

    m = np.linspace(mbright, mlim, num=1000)
    if fit == 'composite':
        farr = f(loglf, theta, m, z, fit='composite')
    else:
        farr = f(loglf, theta, m, z, fit='individual')
    
    return np.trapz(farr, m) # cMpc^-3

def get_rhoqso(lfi, mlim, z, fit='individual', mbright=-35.0):

    rindices = np.random.randint(len(lfi.samples), size=300)
    n = np.array([rhoqso(lfi.log10phi, theta, mlim, z, mbright=mbright) 
                  for theta
                  in lfi.samples[rindices]])
    u = np.percentile(n, 15.87) 
    l = np.percentile(n, 84.13)
    c = np.mean(n)
    lfi.rhoqso = [u, l, c]

    return

def draw(individuals, zlims, select=False):

    """
    Calculates and plots LyC emissivity.

    """

    fig = plt.figure(figsize=(7, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\rho(z, M_{1450} < M_\mathrm{lim})$ [cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7)

    ax.set_yscale('log')
    ax.set_ylim(1.0e-10, 1.0e-3)

    # Plot 1 
    
    mlim = -18

    if select: 
        selected = [x for x in individuals if x.z.mean() < 2.0 or x.z.mean() > 2.8]
    else:
        selected = individuals

    for x in selected:
        get_rhoqso(x, mlim, x.z.mean())
    
    c = np.array([x.rhoqso[2] for x in selected])
    u = np.array([x.rhoqso[0] for x in selected])
    l = np.array([x.rhoqso[1] for x in selected])

    rho = c
    rho_up = u - c
    rho_low = c - l 
    
    zs = np.array([x.z.mean() for x in selected])
    uz = np.array([x.zlims[0] for x in selected])
    lz = np.array([x.zlims[1] for x in selected])
    
    uzerr = uz-zs
    lzerr = zs-lz 

    ax.scatter(zs, rho, c='tomato', edgecolor='None',
               label='$M < -18$',
               s=72, zorder=4, linewidths=2) 

    ax.errorbar(zs, rho, ecolor='tomato', capsize=0, fmt='None', elinewidth=2,
                yerr=np.vstack((rho_low, rho_up)),
                xerr=np.vstack((lzerr, uzerr)), 
                zorder=3, mew=1, ms=5)

    # Plot 2 
    
    mlim = -21

    for x in selected:
        get_rhoqso(x, mlim, x.z.mean())
    
    c = np.array([x.rhoqso[2] for x in selected])
    u = np.array([x.rhoqso[0] for x in selected])
    l = np.array([x.rhoqso[1] for x in selected])

    rho = c
    rho_up = u - c
    rho_low = c - l 
    
    zs = np.array([x.z.mean() for x in selected])
    uz = np.array([x.zlims[0] for x in selected])
    lz = np.array([x.zlims[1] for x in selected])
    
    uzerr = uz-zs
    lzerr = zs-lz 

    ax.scatter(zs, rho, c='forestgreen', edgecolor='None',
               label='$M<-21$',
               s=72, zorder=4, linewidths=2) 

    ax.errorbar(zs, rho, ecolor='forestgreen', capsize=0, fmt='None',
                elinewidth=2, yerr=np.vstack((rho_low, rho_up)),
                xerr=np.vstack((lzerr, uzerr)), zorder=3, mew=1, ms=5)

    # Plot 3
    
    mlim = -24

    for x in selected:
        get_rhoqso(x, mlim, x.z.mean())
    
    c = np.array([x.rhoqso[2] for x in selected])
    u = np.array([x.rhoqso[0] for x in selected])
    l = np.array([x.rhoqso[1] for x in selected])

    rho = c
    rho_up = u - c
    rho_low = c - l 
    
    zs = np.array([x.z.mean() for x in selected])
    uz = np.array([x.zlims[0] for x in selected])
    lz = np.array([x.zlims[1] for x in selected])
    
    uzerr = uz-zs
    lzerr = zs-lz 

    ax.scatter(zs, rho, c='goldenrod', edgecolor='None',
               label='$M<-24$',
               s=72, zorder=4, linewidths=2) 

    ax.errorbar(zs, rho, ecolor='goldenrod', capsize=0, fmt='None',
                elinewidth=2, yerr=np.vstack((rho_low, rho_up)),
                xerr=np.vstack((lzerr, uzerr)), zorder=3, mew=1, ms=5)

    # Plot 4
    
    mlim = -27

    for x in selected:
        get_rhoqso(x, mlim, x.z.mean())
    
    c = np.array([x.rhoqso[2] for x in selected])
    u = np.array([x.rhoqso[0] for x in selected])
    l = np.array([x.rhoqso[1] for x in selected])

    rho = c
    rho_up = u - c
    rho_low = c - l 
    
    zs = np.array([x.z.mean() for x in selected])
    uz = np.array([x.zlims[0] for x in selected])
    lz = np.array([x.zlims[1] for x in selected])
    
    uzerr = uz-zs
    lzerr = zs-lz 

    ax.scatter(zs, rho, c='saddlebrown', edgecolor='None',
               label='$M<-27$',
               s=72, zorder=4, linewidths=2) 

    ax.errorbar(zs, rho, ecolor='saddlebrown', capsize=0, fmt='None',
                elinewidth=2, yerr=np.vstack((rho_low, rho_up)),
                xerr=np.vstack((lzerr, uzerr)), zorder=3, mew=1, ms=5)
    
    plt.legend(loc='upper left', fontsize=14, handlelength=1,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01, scatterpoints=1)
    
    plt.savefig('rhoqso.pdf',bbox_inches='tight')
    plt.close('all')

    return

def global_cumulative(ax, composite, mlim, color):

    nzs = 50 
    z = np.linspace(0, 7, nzs)
    nsample = 300
    rsample = composite.samples[np.random.randint(len(composite.samples), size=nsample)]

    bf = np.median(composite.samples, axis=0)
    r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
    ax.plot(z, r, color='k', zorder=7)

    r = np.zeros((nsample, nzs))
    for i, theta in enumerate(rsample):
        r[i] = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])

    up = np.percentile(r, 15.87, axis=0)
    down = np.percentile(r, 84.13, axis=0)
    ax.fill_between(z, down, y2=up, color=color, zorder=6, alpha=0.5)

    return


def global_differential(ax, composite, mbright, mfaint, color):

    nzs = 50 
    z = np.linspace(0, 7, nzs)
    nsample = 300
    rsample = composite.samples[np.random.randint(len(composite.samples), size=nsample)]

    bf = np.median(composite.samples, axis=0)
    r = np.array([rhoqso(composite.log10phi, bf, mfaint, x, mbright=mbright, fit='composite') for x in z])
    ax.plot(z, r, color='k', zorder=7)

    r = np.zeros((nsample, nzs))
    for i, theta in enumerate(rsample):
        r[i] = np.array([rhoqso(composite.log10phi, theta, mfaint, x, mbright=mbright, fit='composite') for x in z])

    up = np.percentile(r, 15.87, axis=0)
    down = np.percentile(r, 84.13, axis=0)

    label = '${:d}>M>{:d}$'.format(mfaint, mbright) 
    ax.fill_between(z, down, y2=up, color=color, zorder=6, alpha=0.5, label=label, linewidth=0)

    return


def global_optimum_differential(ax, composite, mbright, mfaint, color):

    nzs = 50 
    z = np.linspace(0, 7, nzs)

    bf = composite.bf.x 
    r = np.array([rhoqso(composite.log10phi, bf, mfaint, x, mbright=mbright, fit='composite') for x in z])
    ax.plot(z, r, color='k', zorder=7)

    return


def individuals_differential(ax, individuals, mbright, mfaint, color):

    # These redshift bins are labelled "bad" and are plotted differently.
    reject = [0, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    m = np.ones(len(individuals), dtype=bool)
    m[reject] = False
    minv = np.logical_not(m)

    individuals_good = [x for i, x in enumerate(individuals) if i not in set(reject)]
    individuals_bad = [x for i, x in enumerate(individuals) if i in set(reject)]
    
    for x in individuals:
        get_rhoqso(x, mfaint, x.z.mean(), mbright=mbright)
    
    c = np.array([x.rhoqso[2] for x in individuals_good])
    u = np.array([x.rhoqso[0] for x in individuals_good])
    l = np.array([x.rhoqso[1] for x in individuals_good])

    rho = c
    rho_up = u - c
    rho_low = c - l 
    
    zs = np.array([x.z.mean() for x in individuals_good])
    uz = np.array([x.zlims[0] for x in individuals_good])
    lz = np.array([x.zlims[1] for x in individuals_good])
    
    uzerr = uz-zs
    lzerr = zs-lz 

    ax.scatter(zs, rho, c=color, edgecolor='None',
               s=72, zorder=10, linewidths=2) 

    ax.errorbar(zs, rho, ecolor=color, capsize=0, fmt='None', elinewidth=2,
                yerr=np.vstack((rho_low, rho_up)),
                xerr=np.vstack((lzerr, uzerr)), 
                zorder=10, mew=1, ms=5)

    c = np.array([x.rhoqso[2] for x in individuals_bad])
    u = np.array([x.rhoqso[0] for x in individuals_bad])
    l = np.array([x.rhoqso[1] for x in individuals_bad])

    rho = c
    rho_up = u - c
    rho_low = c - l 
    
    zs = np.array([x.z.mean() for x in individuals_bad])
    uz = np.array([x.zlims[0] for x in individuals_bad])
    lz = np.array([x.zlims[1] for x in individuals_bad])
    
    uzerr = uz-zs
    lzerr = zs-lz 

    ax.errorbar(zs, rho, ecolor=color, capsize=0, fmt='None', elinewidth=1,
                yerr=np.vstack((rho_low, rho_up)),
                xerr=np.vstack((lzerr, uzerr)), 
                zorder=10, mew=1, ms=5)

    ax.scatter(zs, rho, c='#ffffff', edgecolor=color,
               s=72, zorder=10, linewidths=1) 

    
    return 


def individuals_cumulative(ax, individuals, mlim, color, label):

    # These redshift bins are labelled "bad" and are plotted differently.
    reject =[]#  [0, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    m = np.ones(len(individuals), dtype=bool)
    m[reject] = False
    minv = np.logical_not(m)

    individuals_good = [x for i, x in enumerate(individuals) if i not in set(reject)]
    individuals_bad = [x for i, x in enumerate(individuals) if i in set(reject)]
    
    for x in individuals:
        get_rhoqso(x, mlim, x.z.mean())
    
    c = np.array([x.rhoqso[2] for x in individuals_good])
    u = np.array([x.rhoqso[0] for x in individuals_good])
    l = np.array([x.rhoqso[1] for x in individuals_good])

    rho = c
    rho_up = u - c
    rho_low = c - l 
    
    zs = np.array([x.z.mean() for x in individuals_good])
    uz = np.array([x.zlims[0] for x in individuals_good])
    lz = np.array([x.zlims[1] for x in individuals_good])
    
    uzerr = uz-zs
    lzerr = zs-lz 

    ax.scatter(zs, rho, c=color, edgecolor='None',
               label=label,
               s=72, zorder=10, linewidths=2) 

    ax.errorbar(zs, rho, ecolor=color, capsize=0, fmt='None', elinewidth=2,
                yerr=np.vstack((rho_low, rho_up)),
                xerr=np.vstack((lzerr, uzerr)), 
                zorder=10, mew=1, ms=5)

    c = np.array([x.rhoqso[2] for x in individuals_bad])
    u = np.array([x.rhoqso[0] for x in individuals_bad])
    l = np.array([x.rhoqso[1] for x in individuals_bad])

    rho = c
    rho_up = u - c
    rho_low = c - l 
    
    zs = np.array([x.z.mean() for x in individuals_bad])
    uz = np.array([x.zlims[0] for x in individuals_bad])
    lz = np.array([x.zlims[1] for x in individuals_bad])
    
    uzerr = uz-zs
    lzerr = zs-lz 

    ax.errorbar(zs, rho, ecolor=color, capsize=0, fmt='None', elinewidth=1,
                yerr=np.vstack((rho_low, rho_up)),
                xerr=np.vstack((lzerr, uzerr)), 
                zorder=10, mew=1, ms=5)

    ax.scatter(zs, rho, c='#ffffff', edgecolor=color,
               s=72, zorder=10, linewidths=1) 

    
    return 
    

def draw_withGlobal_dense(composite, individuals, zlims, select=False):

    fig = plt.figure(figsize=(7, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\rho(z, M_{1450} < M_\mathrm{lim})$ [cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7)

    ax.set_yscale('log')
    ax.set_ylim(1.0e-10, 4.0e-3)

    mlim = -18
    individuals_cumulative(ax, individuals, mlim, 'tomato', '$M<-18$')
    global_cumulative(ax, composite, mlim, 'tomato')

    mlim = -19
    c='#ff7f0e'
    individuals_cumulative(ax, individuals, mlim, c, '$M<-19$')
    global_cumulative(ax, composite, mlim, c)
    
    mlim = -20
    c='#1f77b4'
    individuals_cumulative(ax, individuals, mlim, c, '$M<-20$')
    global_cumulative(ax, composite, mlim, c)
    
    mlim = -21
    individuals_cumulative(ax, individuals, mlim, 'forestgreen', '$M<-21$')
    global_cumulative(ax, composite, mlim, 'forestgreen')

    mlim = -22
    c='#9467bd'
    individuals_cumulative(ax, individuals, mlim, c, '$M<-22$')
    global_cumulative(ax, composite, mlim, c)
    
    mlim = -23
    c='#8c564b'
    individuals_cumulative(ax, individuals, mlim, c, '$M<-23$')
    global_cumulative(ax, composite, mlim, c)
    
    mlim = -24
    individuals_cumulative(ax, individuals, mlim, 'goldenrod', '$M<-24$')
    global_cumulative(ax, composite, mlim, 'goldenrod')

    mlim = -25
    c='#bcbd22'
    individuals_cumulative(ax, individuals, mlim, c, '$M<-25$')
    global_cumulative(ax, composite, mlim, c)

    mlim = -26
    c = '#7f7f7f'
    individuals_cumulative(ax, individuals, mlim, c, '$M<-26$')
    global_cumulative(ax, composite, mlim, c)
    
    mlim = -27
    c = '#17becf'
    individuals_cumulative(ax, individuals, mlim, c, '$M<-27$')
    global_cumulative(ax, composite, mlim, c)
    
    plt.legend(loc='upper left', fontsize=14, handlelength=1,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01,
               scatterpoints=1, ncol=2)
    
    plt.savefig('rhoqso_withGlobal.pdf',bbox_inches='tight')
    plt.close('all')

    return

def draw_withGlobal(composite, individuals, zlims, select=False):

    fig = plt.figure(figsize=(7, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\rho(z, M_{1450} < M_\mathrm{lim})$ [cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7)

    ax.set_yscale('log')
    ax.set_ylim(1.0e-10, 1.0e-3)

    mlim = -18
    individuals_cumulative(ax, individuals, mlim, 'tomato', '$M<-18$')
    #global_cumulative(ax, composite, mlim, 'tomato')

    mlim = -21
    individuals_cumulative(ax, individuals, mlim, 'forestgreen', '$M<-21$')
   # global_cumulative(ax, composite, mlim, 'forestgreen')

    mlim = -24
    individuals_cumulative(ax, individuals, mlim, 'goldenrod', '$M<-24$')
   # global_cumulative(ax, composite, mlim, 'goldenrod')

    mlim = -27
    c = '#17becf'
    individuals_cumulative(ax, individuals, mlim, c, '$M<-27$')
   # global_cumulative(ax, composite, mlim, c)
    
    plt.legend(loc='upper left', fontsize=14, handlelength=1,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01,
               scatterpoints=1)
    
    plt.savefig('rhoqso_withGlobal.pdf',bbox_inches='tight')
    plt.close('all')

    return


def draw_differential_dense(composite, individuals, zlims, select=False):

    fig = plt.figure(figsize=(7, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\rho(z)$ [cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7)

    ax.set_yscale('log')
    ax.set_ylim(1.0e-10, 4.0e-3)

    mf = -18
    mb = -19
    c = 'tomato'
    global_differential(ax, composite, mb, mf, c)

    mf = -19
    mb = -20
    c='#ff7f0e'
    global_differential(ax, composite, mf, mf, c)
    
    mf = -20
    mb = -21
    c='#1f77b4'
    global_differential(ax, composite, mb, mf, c)
    
    mf = -21
    mb = -22
    c = 'forestgreen'
    global_differential(ax, composite, mb, mf, c)

    mf = -22
    mb = -23
    c='#9467bd'
    global_differential(ax, composite, mb, mf, c)
    
    mf = -23
    mb = -24
    c='#8c564b'
    global_differential(ax, composite, mb, mf, c)
    
    mf = -24
    mb = -25
    c = 'goldenrod'
    global_differential(ax, composite, mb, mf, c)

    mf = -25
    mb = -26
    c='#bcbd22'
    global_differential(ax, composite, mb, mf, c)

    mf = -26
    mb = -27
    c = '#7f7f7f'
    global_differential(ax, composite, mb, mf, c)
    
    mf = -27
    mb = -28
    c = '#17becf'
    global_differential(ax, composite, mb, mf, c)
    
    plt.legend(loc='upper left', fontsize=14, handlelength=1.5,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01,
               scatterpoints=1, ncol=2)
    
    plt.savefig('rhoqso_diff.pdf',bbox_inches='tight')
    plt.close('all')

    return


def draw_differential_dense_optimum(composite, individuals, zlims, select=False):

    fig = plt.figure(figsize=(7, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\rho(z)$ [cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7)

    ax.set_yscale('log')
    ax.set_ylim(1.0e-10, 4.0e-3)

    mf = -18
    mb = -19
    c = 'tomato'
    global_optimum_differential(ax, composite, mb, mf, c)

    mf = -19
    mb = -20
    c='#ff7f0e'
    global_optimum_differential(ax, composite, mf, mf, c)
    
    mf = -20
    mb = -21
    c='#1f77b4'
    global_optimum_differential(ax, composite, mb, mf, c)
    
    mf = -21
    mb = -22
    c = 'forestgreen'
    global_optimum_differential(ax, composite, mb, mf, c)

    mf = -22
    mb = -23
    c='#9467bd'
    global_optimum_differential(ax, composite, mb, mf, c)
    
    mf = -23
    mb = -24
    c='#8c564b'
    global_optimum_differential(ax, composite, mb, mf, c)
    
    mf = -24
    mb = -25
    c = 'goldenrod'
    global_optimum_differential(ax, composite, mb, mf, c)

    mf = -25
    mb = -26
    c='#bcbd22'
    global_optimum_differential(ax, composite, mb, mf, c)

    mf = -26
    mb = -27
    c = '#7f7f7f'
    global_optimum_differential(ax, composite, mb, mf, c)
    
    mf = -27
    mb = -28
    c = '#17becf'
    global_optimum_differential(ax, composite, mb, mf, c)
    
    plt.legend(loc='upper left', fontsize=14, handlelength=1.5,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01,
               scatterpoints=1, ncol=2)
    
    plt.savefig('rhoqso_diff.pdf',bbox_inches='tight')
    plt.close('all')

    return


def draw_differential(composite, individuals, zlims, select=False):

    fig = plt.figure(figsize=(7, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\rho(z)$ [cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7)

    ax.set_yscale('log')
    ax.set_ylim(1.0e-10, 4.0e-3)

    mf = -18
    mb = -21
    c = 'tomato'
    individuals_differential(ax, individuals, mb, mf, c)
    global_differential(ax, composite, mb, mf, c)

    mf = -21
    mb = -24
    c = 'forestgreen'
    individuals_differential(ax, individuals, mb, mf, c)
    global_differential(ax, composite, mb, mf, c)

    mf = -24
    mb = -27
    c = 'goldenrod'
    individuals_differential(ax, individuals, mb, mf, c)
    global_differential(ax, composite, mb, mf, c)

    mf = -27
    mb = -30
    c = 'saddlebrown'
    individuals_differential(ax, individuals, mb, mf, c)
    global_differential(ax, composite, mb, mf, c)
    
    plt.legend(loc='upper left', fontsize=14, handlelength=1.5,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01,
               scatterpoints=1, ncol=2)
    
    plt.savefig('rhoqso_diff_ind.pdf',bbox_inches='tight')
    plt.close('all')

    return

def draw_onlyGlobal(composite):

    fig = plt.figure(figsize=(7, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\rho(z, M_{1450} < M_\mathrm{lim})$ [cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7)

    ax.set_yscale('log')
    ax.set_ylim(1.0e-10, 1.0e-3)

    # Plot 1 
    
    mlim = -18

    z = np.linspace(0, 7, 50)
    bf = np.median(composite.samples, axis=0)
    r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
    ax.plot(z, r, color='k', zorder=7)

    for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
        r = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])
        ax.plot(z, r, color='tomato', zorder=6, alpha=0.02)
    

    # Plot 2 
    
    mlim = -21

    z = np.linspace(0, 7, 50)
    bf = np.median(composite.samples, axis=0)
    r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
    ax.plot(z, r, color='k', zorder=7)

    for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
        r = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])
        ax.plot(z, r, color='forestgreen', zorder=6, alpha=0.02)
    

    # Plot 3
    
    mlim = -24

    z = np.linspace(0, 7, 50)
    bf = np.median(composite.samples, axis=0)
    r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
    ax.plot(z, r, color='k', zorder=7)

    for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
        r = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])
        ax.plot(z, r, color='goldenrod', zorder=6, alpha=0.02)

    
    # Plot 4
    
    mlim = -27

    z = np.linspace(0, 7, 50)
    bf = np.median(composite.samples, axis=0)
    r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
    ax.plot(z, r, color='k', zorder=7)

    for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
        r = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])
        ax.plot(z, r, color='saddlebrown', zorder=6, alpha=0.02)
    
    plt.legend(loc='upper left', fontsize=14, handlelength=1,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01, scatterpoints=1)
    
    plt.savefig('rhoqso_onlyGlobal.pdf',bbox_inches='tight')
    plt.close('all')

    return



