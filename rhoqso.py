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

def rhoqso(loglf, theta, mlim, z, fit='individual'):

    m = np.linspace(-35.0, mlim, num=1000)
    if fit == 'composite':
        farr = f(loglf, theta, m, z, fit='composite')
    else:
        farr = f(loglf, theta, m, z, fit='individual')
    
    return np.trapz(farr, m) # cMpc^-3

def get_rhoqso(lfi, mlim, z, fit='individual'):

    rindices = np.random.randint(len(lfi.samples), size=300)
    n = np.array([rhoqso(lfi.log10phi, theta, mlim, z) 
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

    z = np.linspace(0, 7, 50)
    bf = np.median(composite.samples, axis=0)
    r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
    ax.plot(z, r, color='k', zorder=7)

    for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
        r = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])
        ax.plot(z, r, color='tomato', zorder=6, alpha=0.02)
    

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

    z = np.linspace(0, 7, 50)
    bf = np.median(composite.samples, axis=0)
    r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
    ax.plot(z, r, color='k', zorder=7)

    for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
        r = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])
        ax.plot(z, r, color='forestgreen', zorder=6, alpha=0.02)
    

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

    z = np.linspace(0, 7, 50)
    bf = np.median(composite.samples, axis=0)
    r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
    ax.plot(z, r, color='k', zorder=7)

    for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
        r = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])
        ax.plot(z, r, color='goldenrod', zorder=6, alpha=0.02)

    
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
    
    plt.savefig('rhoqso_withGlobal.pdf',bbox_inches='tight')
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



