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

    """1450A emissivity.

    """

    m = np.linspace(mbright, mlim, num=100)
    farr = np.array([(10.0**loglf(theta, x))*luminosity(x) for x in m])
    
    return np.trapz(farr, m) # ergs s^-1 Hz^-1 cMpc^-3

def get_rhoqso(lfi, mlim, mbright=-23):

    """1450A emissivity (statistics).

    """
    #np.random.seed(5)
    rindices = np.random.randint(len(lfi.samples), size=300)
    r = np.array([rhoqso(lfi.log10phi, theta, mlim, mbright=mbright)
                          for theta
                          in lfi.samples[rindices]])
    
    l = np.percentile(r, 15.87) 
    u = np.percentile(r, 84.13)
    c = np.mean(r)

    lfi.rhoqso = [u, l, c] # ergs s^-1 Hz^-1 cMpc^-3

def rhoqso_912(loglf, theta, mlim, mbright=-30.0):

    """912A emissivity.

    """

    m = np.linspace(mbright, mlim, num=100)
    farr = np.array([10.0**loglf(theta, x)*luminosity(x)*((912.0/1450.0)**0.61) for x in m])
    
    return np.trapz(farr, m) # ergs s^-1 Hz^-1 cMpc^-3

def get_rhoqso_912(lfi, mlim, mbright=-30):

    """912A emissivity (statistics).

    """

    #np.random.seed(5)
    rindices = np.random.randint(len(lfi.samples), size=300)
    r = np.array([rhoqso_912(lfi.log10phi, theta, mlim, mbright=mbright)
                          for theta
                          in lfi.samples[rindices]])
    
    l = np.percentile(r, 15.87) 
    u = np.percentile(r, 84.13)
    c = np.mean(r)

    lfi.rhoqso = [u, l, c] # ergs s^-1 Hz^-1 cMpc^-3

    
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

def get_fits(zs, uz, lz, c, u, l):

    em = c
    em_up = u - c
    em_low = c - l

    uzerr = uz - zs
    lzerr = zs - lz 
        
    def func(z, a, b, c, d, e):
        e = 10.0**a * (1.0+z)**b * np.exp(-c*z) / (np.exp(d*z)+e)
        return e # erg s^-1 Hz^-1 Mpc^-3

    sigma = u-l
    samples, params_bf = fit_emissivity.fit(zs, em, sigma)

    nsample = 300
    rsample = samples[np.random.randint(len(samples), size=nsample)]

    z = np.linspace(0.0, 20, num=1000)
    nzs = len(z) 
    e = np.zeros((nsample, nzs))
    
    for i, theta in enumerate(rsample):
        e[i] = np.array(func(z, *theta))

    up = np.percentile(e, 15.87, axis=0)
    down = np.percentile(e, 84.13, axis=0)
    b = np.median(e, axis=0)

    return z, b, up, down, params_bf

def plot_data(ax, redshifts, data, color, **kwargs):

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
        
    ax.errorbar(zs, em, ecolor=color, capsize=0,
                fmt='None', elinewidth=1.5,
                yerr=np.vstack((em_low, em_up)),
                xerr=np.vstack((lzerr, uzerr)), zorder=1)
    
    ax.scatter(zs, em, c='white', edgecolor=color,
               s=48, linewidths=1.5, zorder=2, **kwargs)
    
    
    return 

def get_fit_mcmc(individuals):

    individuals_good = select_data(individuals)

    z = np.array([x.z.mean() for x in individuals_good])
    uz = np.array([x.zlims[0] for x in individuals_good])
    lz = np.array([x.zlims[1] for x in individuals_good])

    # Get the 1450A emissivity of bright qsos (-18 > M > -30).
    for x in individuals_good:
        get_rhoqso(x, -18.0, mbright=-40.0)
    c18 = np.array([x.rhoqso[2] for x in individuals_good])
    u18 = np.array([x.rhoqso[0] for x in individuals_good])
    l18 = np.array([x.rhoqso[1] for x in individuals_good])

    print 'z=',z
    print 'e1450--18=',c18
    print 'e1450--18*factor=',(c18)*((912.0/1450.0)**0.61)
    
    z1450, b1450, up1450, down1450, p = get_fits(z, uz, lz, c18, u18, l18)
            
    write = True
    if write: 
        np.savez('e1450_18', z=z1450, median=b1450, up=up1450, down=down1450)

    #-----

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    
    plt.minorticks_on()
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    
    ax.set_ylabel(r'$\epsilon$ [erg s$^{-1}$ Hz$^{-1}$ cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    
    ax.set_xlim(0.,8.)
    ax.set_ylim(1.0e22, 1.0e26)
    ax.set_yscale('log')

    plot_data(ax, (z, uz, lz), (c18, u18, l18), 'k')
    
    ax.plot(z1450, b1450, lw=2, c='r')
    ax.fill_between(z1450, down1450, y2=up1450, color='r', zorder=-1, alpha=0.2, edgecolor='None')

    def func(z, a, b, c, d, e):
        e = 10.0**a * (1.0+z)**b * np.exp(-c*z) / (np.exp(d*z)+e)
        return e # erg s^-1 Hz^-1 Mpc^-3

    e = func(z1450, *p)
    ax.plot(z1450, e, lw=2, c='k')
    
    plt.savefig('rhoqso_fit2.pdf', bbox_inches='tight')
    plt.close('all')
    
    return

    
def print_fits(individuals):

    # See emissivity.txt and tabulate_emissivity.py for how to use
    # this information.
    
    individuals_good = individuals # Do not select for printing!

    z = np.array([x.z.mean() for x in individuals_good])
    uz = np.array([x.zlims[0] for x in individuals_good])
    lz = np.array([x.zlims[1] for x in individuals_good])

    # Get the 1450A emissivity of bright qsos (-18 > M > -30).
    for x in individuals_good:
        get_rhoqso(x, -18.0, mbright=-40.0)
    c18 = np.array([x.rhoqso[2] for x in individuals_good])
    u18 = np.array([x.rhoqso[0] for x in individuals_good])
    l18 = np.array([x.rhoqso[1] for x in individuals_good])

    em_1450 = c18
    em_up_1450 = u18 - c18
    em_low_1450 = c18 - l18

    em = em_1450*((912.0/1450.0)**0.61)
    em_up = em_up_1450*((912.0/1450.0)**0.61)
    em_low = em_low_1450*((912.0/1450.0)**0.61)
    
    for x in individuals_good:
        get_rhoqso(x, -21.0, mbright=-40.0)
    c21 = np.array([x.rhoqso[2] for x in individuals_good])
    u21 = np.array([x.rhoqso[0] for x in individuals_good])
    l21 = np.array([x.rhoqso[1] for x in individuals_good])

    em21_1450 = c21
    em21_up_1450 = u21 - c21
    em21_low_1450 = c21 - l21 
    
    em21 = em21_1450*((912.0/1450.0)**0.61)
    em21_up = em21_up_1450*((912.0/1450.0)**0.61)
    em21_low = em21_low_1450*((912.0/1450.0)**0.61)
    
    zs = z 
    uzerr = uz-zs
    lzerr = zs-lz 

    for i in range(len(zs)):
        print zs[i], lz[i], uz[i], em[i]/1.0e24, em_up[i]/1.0e24, em_low[i]/1.0e24, em_1450[i]/1.0e24, em_up_1450[i]/1.0e24, em_low_1450[i]/1.0e24, em21[i]/1.0e24, em21_up[i]/1.0e24, em21_low[i]/1.0e24, em21_1450[i]/1.0e24, em21_up_1450[i]/1.0e24, em21_low_1450[i]/1.0e24
    
    return

    
