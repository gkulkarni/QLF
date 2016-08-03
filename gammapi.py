import numpy as np 
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
import rtg 

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

def get_gamma_error(individuals):

    g_up = []
    g_low = []

    for x in individuals:

        z = x.z.mean()

        lowlim = np.percentile(x.samples, 15.87, axis=0)
        rate = rtg.gamma_HI(z, x.log10phi, lowlim, individual=True)
        rate = np.log10(rate)+12.0
        g_low.append(rate)

        uplim = np.percentile(x.samples, 84.13, axis=0)
        rate = rtg.gamma_HI(z, x.log10phi, uplim, individual=True)
        rate = np.log10(rate)+12.0
        g_up.append(rate)
        
    g_up = np.array(g_up)
    g_low = np.array(g_low)

    return g_up, g_low 
    

def plot_gamma(composite, individuals=None, zlims=(2.0,6.5), dirname='', fast=True, rt=False, lsa=False):

    """
    Calculates and plots HI photoionization rate. 

    rt   = True: uses the integral solution to the cosmological RT equation 
    lsa  = True: uses local source approximation 
    fast = True: reads stored values instead of calculating again 

    """

    zmin, zmax = zlims 
    z = np.linspace(zmin, zmax, num=10) 

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    if lsa:

        rindices = np.random.randint(len(composite.samples), size=900)
        for theta in composite.samples[rindices]:
            g = np.array([Gamma_HI(composite.log10phi, theta, rs) for rs in z])
            g = np.log10(g)+12.0
            ax.plot(z, g, color='#67a9cf', alpha=0.1, zorder=1)

        bf = composite.samples.mean(axis=0)
        g = np.array([Gamma_HI(composite.log10phi, bf, rs) for rs in z])
        g = np.log10(g)+12.0
        ax.plot(z, g, color='k', zorder=2)

    if rt:
        
        if fast:
            data = np.load('gammapi.npz')
            z = data['z']
            ga = data['ga']
            g_mean = np.mean(ga, axis=0)
            g_up = np.percentile(ga, 15.87, axis=0)
            g_low = np.percentile(ga, 100.0-15.87, axis=0)
            ax.fill_between(z, g_low, g_up, color='#67a9cf', alpha=0.5, edgecolor='None', zorder=1) 
            ax.plot(z, g_mean, color='k', zorder=2)
        else:
            theta = composite.samples[np.random.randint(len(composite.samples))]
            ga = np.array([rtg.gamma_HI(rs, composite.log10phi, theta) for rs in z])
            ga = np.log10(ga)+12.0
            count = 1 
            print count 
            rindices = np.random.randint(len(composite.samples), size=100)
            for theta in composite.samples[rindices]:
                g = np.array([rtg.gamma_HI(rs, composite.log10phi, theta) for rs in z])
                count += 1
                print count 
                g = np.log10(g)+12.0
                ax.plot(z, g, color='#67a9cf', alpha=0.3, zorder=1)
                ga = np.vstack((ga, g))
                
            bf = composite.samples.mean(axis=0)
            g = np.array([Gamma_HI(composite.log10phi, bf, rs) for rs in z])
            g = np.log10(g)+12.0
            ax.plot(z, g, color='k', zorder=2)

            np.savez('gammapi', z=z, ga=ga, g=g)

    ax.set_ylabel(r'$\log_{10}(\Gamma_\mathrm{HI}/10^{-12} \mathrm{s}^{-1})$')
    ax.set_xlabel('$z$')
    ax.set_xlim(2.,6.5)
    ax.set_ylim(-2.,0.5)
    ax.set_xticks((2,3,4,5,6))
    
    zm, gm, gm_up, gm_low = np.loadtxt('Data/BeckerBolton.dat',unpack=True) 

    ax.scatter(zm, gm, c='#d7191c', edgecolor='None', label='Becker and Bolton 2013', s=32)
    ax.errorbar(zm, gm, ecolor='#d7191c', capsize=5,
                yerr=np.vstack((gm_up, abs(gm_low))),
                fmt='None', zorder=3, mfc='#d7191c', mec='#d7191c',
                mew=1, ms=5)

    zm, gm, gm_sigma = np.loadtxt('Data/calverley.dat',unpack=True) 
    gm += 12.0
    ax.scatter(zm, gm, c='#99cc66', edgecolor='None', label='Calverley et al.~2011', s=32) 
    ax.errorbar(zm, gm, ecolor='#99CC66', capsize=5,
                yerr=gm_sigma, fmt='None', zorder=4, mfc='#99CC66',
                mec='#99CC66', mew=1, ms=5)

    if lsa:
        
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

    if rt:

        if individuals is not None:

            if fast:

                data = np.load('gammapi_individuals.npz')
                zs = data['zs']
                g = data['g']
                uzerr = data['uzerr']
                lzerr = data['lzerr']
                uyerr = data['uyerr']
                lyerr = data['lyerr']

                uyerr[-1]=-0.1
                uyerr[-2]=-0.1


                lyerr[-1]=-0.1
                lyerr[-2]=-0.1
                
                ax.scatter(zs, g, s=32, c='#ffffff', edgecolor='#404040', zorder=3, label='Individual fits')
                ax.errorbar(zs, g, ecolor='#404040', capsize=0,
                            yerr=np.vstack((uyerr,lyerr)),
                            xerr=np.vstack((lzerr,uzerr)),
                            fmt='None',zorder=2)

            else:
                    
                zs = np.array([x.z.mean() for x in individuals])
                uz = np.array([x.z.max() for x in individuals])
                lz = np.array([x.z.min() for x in individuals])
                uzerr = uz-zs
                lzerr = zs-lz 
                print zs

                g = []
                for x in individuals:
                    z = x.z.mean()
                    bf = x.samples.mean(axis=0)
                    rate = rtg.gamma_HI(z, x.log10phi, bf, individual=True)
                    rate = np.log10(rate)+12.0
                    g.append(rate)

                g = np.array(g)
                ax.scatter(zs, g, s=32, c='#ffffff', edgecolor='#404040', zorder=3, label='Individual fits')

                g_up, g_low = get_gamma_error(individuals)
                uyerr = g_up-g
                lyerr = g-g_low
                ax.errorbar(zs, g, ecolor='#404040', capsize=0,
                            yerr=np.vstack((uyerr,lyerr)),
                            xerr=np.vstack((lzerr,uzerr)),
                            fmt='None',zorder=2)

                np.savez('gammapi_individuals', zs=zs, g=g, uzerr=uzerr, lzerr=lzerr, uyerr=uyerr, lyerr=lyerr)
        
    plt.legend(loc='lower left', fontsize=14, handlelength=1,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01, scatterpoints=1)
    
    plt.savefig('gammapi.pdf',bbox_inches='tight')
    plt.close('all')

    return

