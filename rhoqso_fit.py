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

def rhoqso_912(loglf, theta, mlim, mbright=-30.0):

    m = np.linspace(mbright, mlim, num=100)
    farr = np.array([10.0**loglf(theta, x)*luminosity(x)*((912.0/1450.0)**0.61) for x in m])
    
    return np.trapz(farr, m) # ergs s^-1 Hz^-1 cMpc^-3

def get_rhoqso_912(lfi, mlim, mbright=-30):

    rindices = np.random.randint(len(lfi.samples), size=300)
    r = np.array([rhoqso_912(lfi.log10phi, theta, mlim, mbright=mbright)
                          for theta
                          in lfi.samples[rindices]])
    
    l = np.percentile(r, 15.87) 
    u = np.percentile(r, 84.13)
    c = np.mean(r)

    lfi.rhoqso = [u, l, c] # ergs s^-1 Hz^-1 cMpc^-3

    
def plot(redshifts, data, fit, filename):

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    plt.minorticks_on()
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    
    ax.set_ylabel(r'$\epsilon$ [erg s$^{-1}$ Hz$^{-1}$ cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    
    ax.set_xlim(0.,20.)
    #ax.set_ylim(1.0e23, 1.0e26)
    ax.set_yscale('log')

    c = data[0]
    u = data[1]
    l = data[2]
    
    em = c
    em_up = u - c
    em_low = c - l

    zs = redshifts[0]
    uz = redshifts[1]
    lz = redshifts[2]

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

    z = fit[0]
    up = fit[2]
    down = fit[3]
    efaintfill = ax.fill_between(z, down, y2=up, color='red', zorder=5, alpha=0.6, edgecolor='None')

    b = fit[1]
    efaint, = plt.plot(z, b, lw=2, c='red', zorder=5)

    plt.savefig(filename, bbox_inches='tight')
    plt.close('all')

    return


def plot_composite(redshifts, fit, filename):

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    plt.minorticks_on()
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    
    ax.set_ylabel(r'$\epsilon$ [erg s$^{-1}$ Hz$^{-1}$ cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    
    ax.set_xlim(0.,20.)
    #ax.set_ylim(1.0e23, 1.0e26)
    ax.set_yscale('log')

    z = fit[0]
    b1 = fit[1]
    up1 = fit[2]
    down1 = fit[3]
    b2 = fit[4]
    up2 = fit[5]
    down2 = fit[6]

    b = b1 + b2
    up = up1 + up2
    down = down1 + down2 
    
    ax.fill_between(z, down, y2=up, color='red', zorder=5, alpha=0.6, edgecolor='None')
    plt.plot(z, b, lw=2, c='red', zorder=5)
    
    plt.savefig(filename, bbox_inches='tight')
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

def get_fits(zs, uz, lz, c21f, u21f, l21f, c18f, u18f, l18f, cb, ub, lb, c18_912, u18_912, l18_912, c21_912, u21_912, l21_912):

    # Fit 1 --------------------------------------------------
    # Faint qsos -21 < M < -23 -- e1450 

    em = c21f
    em_up = u21f - c21f
    em_low = c21f - l21f

    uzerr = uz - zs
    lzerr = zs - lz 
        
    def func(z, a, b, c, d, e):
        e = 10.0**a * (1.0+z)**b * np.exp(-c*z) / (np.exp(d*z)+e)
        return e # erg s^-1 Hz^-1 Mpc^-3

    sigma = u21f-l21f 
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
    downfaint1450_21 = down
    upfaint1450_21 = up 

    b = np.median(e, axis=0)
    medianfaint1450_21 = b

    bf21 = b
    downf21 = down
    upf21 = up 

    plot((zs, uz, lz), (c21f, u21f, l21f), (z, b, up, down), 'e1450_faint21.pdf')

    # Fit 2 --------------------------------------------------
    # Faint qsos -18 < M < -23 -- e1450 

    em = c18f
    em_up = u18f - c18f
    em_low = c18f - l18f
    
    sigma = u18f-l18f
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

    downfaint1450_18 = down
    upfaint1450_18 = up 
    
    b = np.median(e2, axis=0)
    medianfaint1450_18 = b

    bf18 = b
    downf18 = down
    upf18 = up 

    plot((zs, uz, lz), (c18f, u18f, l18f), (z, b, up, down), 'e1450_faint18.pdf')

    # Fit 3 --------------------------------------------------
    # Bright qsos -23 < M < -30 -- e1450 

    em = cb
    em_up = ub - cb
    em_low = cb - lb
    
    sigma = ub-lb
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

    downbright1450 = down
    upbright1450 = up 
    
    b = np.median(e2, axis=0)
    medianbright1450 = b

    plot((zs, uz, lz), (cb, ub, lb), (z, b, up, down), 'e1450_bright.pdf')
    plot_composite((zs, uz, lz), (z, bf18, upf18, downf18, b, up, down), 'composite18.pdf')
    plot_composite((zs, uz, lz), (z, bf21, upf21, downf21, b, up, down), 'composite21.pdf')
    
    # Fit 4 --------------------------------------------------
    # Bright qsos -18 < M < -30 -- e912

    em = c18_912
    em_up = u18_912 - c18_912
    em_low = c18_912 - l18_912
    
    sigma = u18_912 - l18_912
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

    down912_18 = down
    up912_18 = up 
    
    b = np.median(e2, axis=0)
    median912_18 = b

    plot((zs, uz, lz), (c18_912, u18_912, l18_912), (z, b, up, down), 'e912_18.pdf')

    # Fit 5 --------------------------------------------------
    # Bright qsos -23 < M < -30 -- e912

    em = c21_912
    em_up = u21_912 - c21_912
    em_low = c21_912 - l21_912
    
    sigma = u21_912 - l21_912
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

    down912_21 = down
    up912_21 = up 
    
    b = np.median(e2, axis=0)
    median912_21 = b

    plot((zs, uz, lz), (c21_912, u21_912, l21_912), (z, b, up, down), 'e912_21.pdf')
    
    write_output = False
    if write_output:
        
        np.savez('e1450_18_2', z=z,
                 medianbright=medianbright1450,
                 downbright=downbright1450,
                 upbright=upbright1450,
                 medianfaint=medianfaint1450_18,
                 downfaint=downfaint1450_18,
                 upfaint=upfaint1450_18)

        np.savez('e1450_21_2', z=z,
                 medianbright=medianbright1450,
                 downbright=downbright1450,
                 upbright=upbright1450,
                 medianfaint=medianfaint1450_21,
                 downfaint=downfaint1450_21,
                 upfaint=upfaint1450_21)

        np.savez('e912_18_2', z=z,
                 median=median912_18,
                 down=down912_18,
                 up=up912_18)

        np.savez('e912_21_2', z=z,
                 median=median912_21,
                 down=down912_21,
                 up=up912_21)

    print 'done'
        
    return

def get_fit_mcmc(individuals):

    individuals_good = select_data(individuals)

    z = np.array([x.z.mean() for x in individuals_good])
    uz = np.array([x.zlims[0] for x in individuals_good])
    lz = np.array([x.zlims[1] for x in individuals_good])

    # Get the 1450A emissivity of faint qsos (-21 > M > -23).
    for x in individuals_good:
        get_rhoqso(x, -21.0, mbright=-23.0)
    c21f = np.array([x.rhoqso[2] for x in individuals_good])
    u21f = np.array([x.rhoqso[0] for x in individuals_good])
    l21f = np.array([x.rhoqso[1] for x in individuals_good])

    # Get the 1450A emissivity of bright qsos (-23 > M > -30).
    for x in individuals_good:
        get_rhoqso(x, -23.0, mbright=-30.0)
    cb = np.array([x.rhoqso[2] for x in individuals_good])
    ub = np.array([x.rhoqso[0] for x in individuals_good])
    lb = np.array([x.rhoqso[1] for x in individuals_good])

    # Get the 1450A emissivity of faint qsos (-18 > M > -23).
    for x in individuals_good:
        get_rhoqso(x, -18.0, mbright=-23.0)
    c18f = np.array([x.rhoqso[2] for x in individuals_good])
    u18f = np.array([x.rhoqso[0] for x in individuals_good])
    l18f = np.array([x.rhoqso[1] for x in individuals_good])

    # Get the 912A emissivity for M < -18 
    for x in individuals_good:
        get_rhoqso_912(x, -18.0, mbright=-30.0)
    c18_912 = np.array([x.rhoqso[2] for x in individuals_good])
    u18_912 = np.array([x.rhoqso[0] for x in individuals_good])
    l18_912 = np.array([x.rhoqso[1] for x in individuals_good])

    # Get the 912A emissivity for M < -21
    for x in individuals_good:
        get_rhoqso_912(x, -21.0, mbright=-30.0)
    c21_912 = np.array([x.rhoqso[2] for x in individuals_good])
    u21_912 = np.array([x.rhoqso[0] for x in individuals_good])
    l21_912 = np.array([x.rhoqso[1] for x in individuals_good])
    
    # plot(z, uz, lz, c, u, l, c2, u2, l2)

    get_fits(z, uz, lz, c21f, u21f, l21f, c18f, u18f, l18f, cb, ub, lb, c18_912, u18_912, l18_912, c21_912, u21_912, l21_912) 

    return


