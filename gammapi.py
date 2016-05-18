import numpy as np 
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '14'
import matplotlib.pyplot as plt

def luminosity(M):
    return 10.0**((51.60-M)/2.5) # ergs s^-1 Hz^-1 

def f(loglf, theta, M, z, fit='composite'):
    # SED power law index is from Beta's paper.
    L = luminosity(M)
    if fit=='individual':
        # If this gammapi calculation is for an individual fit (in,
        # say, fitlf.py) then loglf might not have any z argument.
        return (10.0**loglf(theta, M))*L*((912.0/1450.0)**0.61)
    return (10.0**loglf(theta, M, z))*L*((912.0/1450.0)**0.61)
        

def emissivity(loglf, theta, z, mlims, fit='composite'):
    # mlims = (lowest magnitude, brightest magnitude)
    #       = (brightest magnitude, faintest magnitude)
    m = np.linspace(mlims[0], mlims[1], num=1000)
    if fit=='individual':
        farr = f(loglf, theta, m, z, fit='individual')
    else:
        farr = f(loglf, theta, m, z)
    return np.trapz(farr, m) # erg s^-1 Hz^-1 Mpc^-3 

def Gamma_HI(loglf, theta, z, fit='composite'):

    if fit=='composite':
        fit_type = 'composite'
    else:
        fit_type = 'individual' 

    # Taken from Equation 11 of Lusso et al. 2015.
    em = emissivity(loglf, theta, z, (-30.0, -23.0), fit=fit_type)
    alpha_EUV = -1.7
    part1 = 4.6e-13 * (em/1.0e24) * ((1.0+z)/5.0)**(-2.4) / (1.5-alpha_EUV) # s^-1 

    em = emissivity(loglf, theta, z, (-23.0, -20.0), fit=fit_type)
    alpha_EUV = -0.56
    part2 = 4.6e-13 * (em/1.0e24) * ((1.0+z)/5.0)**(-2.4) / (1.5-alpha_EUV) # s^-1

    return part1+part2 

def Gamma_HI_singleslope(loglf, theta, z, fit='composite'):

    if fit=='composite':
        fit_type = 'composite'
    else:
        fit_type = 'individual' 

    # Taken from Equation 11 of Lusso et al. 2015.
    em = emissivity(loglf, theta, z, (-30.0, -20.0), fit=fit_type)
    alpha_EUV = -1.7
    return 4.6e-13 * (em/1.0e24) * ((1.0+z)/5.0)**(-2.4) / (1.5-alpha_EUV) # s^-1

def plot_gamma(composite, individuals=None, zlims=(2.0,6.5), dirname=''):

    mpl.rcParams['font.size'] = '14'
    
    zmin, zmax = zlims 
    z = np.linspace(zmin, zmax, num=50) 

    fig = plt.figure(figsize=(6, 6), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=4, width=1)
    ax.tick_params('both', which='minor', length=2, width=1)

    rindices = np.random.randint(len(composite.samples), size=900)
    for theta in composite.samples[rindices]:
        g = np.array([Gamma_HI(composite.log10phi, theta, rs) for rs in z])
        g = np.log10(g)+12.0
        ax.plot(z, g, color='#67a9cf', alpha=0.1, zorder=1)

    bf = composite.samples.mean(axis=0)
    g = np.array([Gamma_HI(composite.log10phi, theta, rs) for rs in z])
    g = np.log10(g)+12.0
    ax.plot(z, g, color='k', zorder=2)

    ax.set_ylabel(r'$\log_{10}(\Gamma_\mathrm{HI}/10^{-12} \mathrm{s}^{-1})$')
    ax.set_xlabel('$z$')
    ax.set_xlim(2.,6.5)
    ax.set_ylim(-2.,1.)
    ax.set_xticks((2,3,4,5,6))
    
    zm, gm, gm_up, gm_low = np.loadtxt('Data/BeckerBolton.dat',unpack=True) 

    ax.errorbar(zm, gm, ecolor='#d7191c', capsize=0,
                yerr=np.vstack((gm_up, abs(gm_low))),
                fmt='o', zorder=3, mfc='#d7191c', mec='#d7191c',
                mew=1, ms=5, label='Becker and Bolton 2013')

    zm, gm, gm_sigma = np.loadtxt('Data/calverley.dat',unpack=True) 
    gm += 12.0 
    ax.errorbar(zm, gm, ecolor='#fdae61', capsize=0,
                yerr=gm_sigma, fmt='o', zorder=4, mfc='#fdae61',
                mec='#fdae61', mew=1, ms=5, label='Calverley et al.~2011')

    if individuals is not None:
        c = np.array([x.gammapi[2]+12.0 for x in individuals])
        u = np.array([x.gammapi[0]+12.0 for x in individuals])
        l = np.array([x.gammapi[1]+12.0 for x in individuals])
        uyerr = u-c
        lyerr = c-l 

        zs = np.array([x.z.mean() for x in individuals])
        uz = np.array([x.z.max() for x in individuals])
        lz = np.array([x.z.min() for x in individuals])
        uzerr = uz-zs
        lzerr = zs-lz 
        
        ax.errorbar(zs, c, ecolor='#404040', capsize=0,
                    yerr=np.vstack((uyerr,lyerr)),
                    xerr=np.vstack((lzerr,uzerr)), fmt='o',
                    mfc='#ffffff', mec='#404040', zorder=3, mew=1,
                    ms=5, label='Individual Fits')

    plt.legend(loc='lower left',fontsize=12,handlelength=3,frameon=False,framealpha=0.0,
            labelspacing=.1,handletextpad=0.4,borderpad=0.2,numpoints=1)
    
    plt.savefig('gammapi.pdf',bbox_inches='tight')

    mpl.rcParams['font.size'] = '22'

    return


