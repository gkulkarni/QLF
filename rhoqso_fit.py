import numpy as np
from scipy.optimize import curve_fit
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
import fit_emissivity

def luminosity(M):

    return 10.0**((51.60-M)/2.5) # ergs s^-1 Hz^-1 

def rhoqso(loglf, theta, mlim, mbright=-23.0):

    m = np.linspace(mbright, mlim, num=100)
    farr = np.array([(10.0**loglf(theta, x))*luminosity(x) for x in m])
    #farr = np.array([10.0**loglf(theta, x)*luminosity(x)*((912.0/1450.0)**0.61) for x in m])
    
    return np.trapz(farr, m) # ergs s^-1 Hz^-1 cMpc^-3

def get_rhoqso(lfi, mlim, mbright=-23):

    rindices = np.random.randint(len(lfi.samples), size=300)
    r = np.array([rhoqso(lfi.log10phi, theta, mlim, mbright=mbright)
                          for theta
                          in lfi.samples[rindices]])
    
    l = np.percentile(r, 15.87) 
    u = np.percentile(r, 84.13)
    c = np.mean(r)

    lfi.rhoqso = [u, l, c] # ergs s^-1 Hz^-1 cMpc^-3
    
def get_fit(individuals):

    # These redshift bins are labelled "bad" and are plotted differently.
    reject = [0, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    m = np.ones(len(individuals), dtype=bool)
    m[reject] = False
    minv = np.logical_not(m) 
    
    individuals_good = [x for i, x in enumerate(individuals) if i not in set(reject)]
    individuals_bad = [x for i, x in enumerate(individuals) if i in set(reject)]

    for x in individuals_good:
        get_rhoqso(x, -18.0)
        
    c = np.array([x.rhoqso[2] for x in individuals_good])
    u = np.array([x.rhoqso[0] for x in individuals_good])
    l = np.array([x.rhoqso[1] for x in individuals_good])

    print c 

    rho = c
    rho_up = u - c
    rho_low = c - l 
    sigma = u-l

    zs = np.array([x.z.mean() for x in individuals_good])
    uz = np.array([x.zlims[0] for x in individuals_good])
    lz = np.array([x.zlims[1] for x in individuals_good])
    
    uzerr = uz-zs
    lzerr = zs-lz 
    
    def func(z, a, b, c, d, e):
        r = 10.0**a * (1.0+z)**b * np.exp(-c*z) / (np.exp(d*z)+e)
        return r # erg s^-1 Hz^-1 Mpc^-3
    
    popt, pcov = curve_fit(func, zs, rho, sigma=sigma, p0=[24.6, 4.68, 0.28, 1.77, 26.3])
    print popt

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    plt.minorticks_on()
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\rho(z, M_{1450} < M_\mathrm{lim})$ [cMpc$^{-3}$]')    
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7.)

    ax.set_yscale('log')
    ax.set_ylim(1.0e23, 1.0e26)

    ax.scatter(zs, rho, c='k', edgecolor='k',
               label='This work ($M_\mathrm{1450}<-18$)',
               s=48, zorder=4, linewidths=1.5) 

    ax.errorbar(zs, rho, ecolor='k', capsize=0, fmt='None', elinewidth=1.5,
                yerr=np.vstack((rho_low, rho_up)),
                xerr=np.vstack((lzerr, uzerr)), 
                mfc='#ffffff', mec='#404040', zorder=3, mew=1,
                ms=5)
    
    z = np.linspace(0, 7)
    plt.plot(z, func(z, *popt), lw=3, c='k', label='Fit')
    
    plt.legend(loc='upper right', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01, scatterpoints=1, borderaxespad=1)

    plt.savefig('rhoqso_fit.pdf',bbox_inches='tight')
    plt.close('all')

    return

def plot(zs, uz, lz, c, u, l, c2, u2, l2):

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    plt.minorticks_on()
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    
    ax.set_ylabel(r'$\epsilon_{1450}$ [erg s$^{-1}$ Hz$^{-1}$ cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    
    ax.set_xlim(0.,20.)
    #ax.set_ylim(1.0e23, 1.0e26)
    ax.set_yscale('log')

    em = c
    em_up = u - c
    em_low = c - l

    uzerr = uz - zs
    lzerr = zs - lz 
        
    ax.errorbar(zs, em, ecolor='k', capsize=0, fmt='None', elinewidth=1.5,
                yerr=np.vstack((em_low, em_up)),
                xerr=np.vstack((lzerr, uzerr)), 
                mfc='#ffffff', mec='#404040', zorder=6, mew=1,
                ms=5)
    
    ax.scatter(zs, em, c='#ffffff', edgecolor='k',
               label='($-23<M_\mathrm{1450}<-18$)',
               s=48, zorder=6, linewidths=1.5)

    def func(z, a, b, c, d, e):
        e = 10.0**a * (1.0+z)**b * np.exp(-c*z) / (np.exp(d*z)+e)
        return e # erg s^-1 Hz^-1 Mpc^-3

    sigma = u-l 
    samples = fit_emissivity.fit(zs, em, sigma)

    nsample = 300
    rsample = samples[np.random.randint(len(samples), size=nsample)]

    z = np.linspace(0.0, 20, num=1000)
    nzs = len(z) 
    e = np.zeros((nsample, nzs))
    
    for i, theta in enumerate(rsample):
        e[i] = np.array(func(z, *theta))

    up = np.percentile(e, 15.87, axis=0)
    down = np.percentile(e, 84.13, axis=0)
    efaintfill = ax.fill_between(z, down, y2=up, color='red', zorder=5, alpha=0.6, edgecolor='None')

    downfaint = down
    upfaint = up 

    b = np.median(e, axis=0)
    efaint, = plt.plot(z, b, lw=2, c='red', zorder=5)

    medianfaint = b 

    em = c2
    em_up = u2 - c2
    em_low = c2 - l2
    
    ax.errorbar(zs, em, ecolor='k', capsize=0, fmt='None', elinewidth=1.5,
                yerr=np.vstack((em_low, em_up)),
                xerr=np.vstack((lzerr, uzerr)), 
                mfc='#ffffff', mec='#404040', zorder=6, mew=1,
                ms=5)
    
    ax.scatter(zs, em, c='k', edgecolor='k',
               label='($-30<M_\mathrm{1450}<-23$)',
               s=48, zorder=6, linewidths=1.5)

    sigma = u2-l2 
    samples = fit_emissivity.fit(zs, em, sigma)

    nsample = 300
    rsample = samples[np.random.randint(len(samples), size=nsample)]

    z = np.linspace(0.0, 20, num=1000)
    nzs = len(z) 
    e2 = np.zeros((nsample, nzs))
    
    for i, theta in enumerate(rsample):
        e2[i] = np.array(func(z, *theta))

    up = np.percentile(e2, 15.87, axis=0)
    down = np.percentile(e2, 84.13, axis=0)
    ebrightfill = ax.fill_between(z, down, y2=up, color='blue', zorder=7, alpha=0.6, edgecolor='None')

    downbright = down
    upbright = up 
    
    b = np.median(e2, axis=0)
    ebright, = plt.plot(z, b, lw=2, c='blue', zorder=7)

    medianbright = b 
    
    ridx1 = np.random.randint(nsample, size=nsample)
    ridx2 = np.random.randint(nsample, size=nsample)
    e_composite = np.zeros((nsample, nzs))

    for i in range(nsample):
        idx1 = ridx1[i]
        idx2 = ridx2[i]
        e_composite[i,:] = e[idx1,:]+e2[idx2,:]

    up = np.percentile(e_composite, 15.87, axis=0)
    down = np.percentile(e_composite, 84.13, axis=0)
    ecompositefill = ax.fill_between(z, down, y2=up, color='gold', zorder=8, alpha=0.6, edgecolor='None')

    b = np.median(e_composite, axis=0)
    ecomposite, = plt.plot(z, b, lw=2, c='gold', zorder=8)

    write_output = False
    if write_output:
        np.savez('e1450_21', z=z,
                 medianbright=medianbright,
                 downbright=downbright,
                 upbright=upbright,
                 medianfaint=medianfaint,
                 downfaint=downfaint,
                 upfaint=upfaint)

    ax.fill_between(z, downbright+downfaint, y2=(upbright+upfaint), color='green', zorder=8, alpha=0.6, edgecolor='None')
    plt.plot(z, medianbright+medianfaint, lw=2, c='green', zorder=8)

    handles, labels = ax.get_legend_handles_labels()
    handles.append((efaintfill,efaint))
    labels.append('Fit for faint qsos')

    handles.append((ebrightfill,ebright))
    labels.append('Fit for bright qsos')

    handles.append((ecompositefill,ecomposite))
    labels.append('Composite')

    plt.legend(handles, labels, loc='upper right', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01, scatterpoints=1, borderaxespad=1)

    plt.savefig('e1450.pdf',bbox_inches='tight')
    plt.close('all')

    return

def select_data(individuals):

    # These redshift bins are labelled "bad" and are plotted differently.
    reject = [0, 1, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    # Pick out the good data.
    m = np.ones(len(individuals), dtype=bool)
    m[reject] = False
    minv = np.logical_not(m) 
    individuals_good = [x for i, x in enumerate(individuals) if i not in set(reject)]
    individuals_bad = [x for i, x in enumerate(individuals) if i in set(reject)]

    return individuals_good 

def get_fit_mcmc(individuals):

    individuals_good = select_data(individuals)

    # Get the 1450A emissivity of faint qsos (-18 > M > -23).
    for x in individuals_good:
        get_rhoqso(x, -21.0)
    c = np.array([x.rhoqso[2] for x in individuals_good])
    u = np.array([x.rhoqso[0] for x in individuals_good])
    l = np.array([x.rhoqso[1] for x in individuals_good])
    z = np.array([x.z.mean() for x in individuals_good])
    uz = np.array([x.zlims[0] for x in individuals_good])
    lz = np.array([x.zlims[1] for x in individuals_good])

    # Get the 1450A emissivity of faint qsos (-23 > M > -30).
    for x in individuals_good:
        get_rhoqso(x, -23.0, mbright=-30)
    c2 = np.array([x.rhoqso[2] for x in individuals_good])
    u2 = np.array([x.rhoqso[0] for x in individuals_good])
    l2 = np.array([x.rhoqso[1] for x in individuals_good])
    
    plot(z, uz, lz, c, u, l, c2, u2, l2)

    return


