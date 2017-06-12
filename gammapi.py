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

def get_emissivity(lfi, z):

    rindices = np.random.randint(len(lfi.samples), size=300)
    e = np.array([emissivity(lfi.log10phi, theta, z, (-30.0, -20.0), fit='individual')
                          for theta
                          in lfi.samples[rindices]])
    u = np.percentile(e, 15.87) 
    l = np.percentile(e, 84.13)
    c = np.mean(e)
    lfi.emissivity = [u, l, c]

    return 

def Gamma_HI(loglf, theta, z, fit='composite'):

    if fit=='composite':
        fit_type = 'composite'
    else:
        fit_type = 'individual' 

    # Taken from Equation 11 of Lusso et al. 2015.
    em = emissivity(loglf, theta, z, (-30.0, -23.0), fit=fit_type)
    alpha_EUV = -1.7
    part1 = 4.6e-13 * (em/1.0e24) * ((1.0+z)/5.0)**(-2.4) / (1.5-alpha_EUV) # s^-1 

    em = emissivity(loglf, theta, z, (-23.0, -18.0), fit=fit_type)
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

def draw(individuals, zlims):

    """
    Calculates and plots HI photoionization rate. 

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\Gamma_\mathrm{HI}~[10^{-12} \mathrm{s}^{-1}]$')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7)

    
    ax.set_ylim(1.0e-2,10)
    ax.set_yscale('log')
    # ax.set_xticks((2,3,4,5,6))

    locs = (1.0e-2, 1.0e-1, 1.0, 10.0)
    labels = ('0.01', '0.1', '1', '10')
    plt.yticks(locs, labels)

    zm, gm, gm_up, gm_low = np.loadtxt('Data/BeckerBolton.dat',unpack=True)
    
    gml = 10.0**gm
    gml_up = 10.0**(gm+gm_up)-10.0**gm
    gml_low = 10.0**gm - 10.0**(gm-np.abs(gm_low))

    ax.scatter(zm, gml, c='#d7191c', edgecolor='None', label='Becker and Bolton 2013', s=64)
    ax.errorbar(zm, gml, ecolor='#d7191c', capsize=5, elinewidth=2, capthick=2,
                yerr=np.vstack((gml_low, gml_up)),
                fmt='None', zorder=1, mfc='#d7191c', mec='#d7191c',
                mew=1, ms=5)

    zm, gm, gm_sigma = np.loadtxt('Data/calverley.dat',unpack=True) 
    gm += 12.0

    gml = 10.0**gm
    gml_up = 10.0**(gm+gm_sigma)-10.0**gm
    gml_low = 10.0**gm - 10.0**(gm-gm_sigma)
    
    ax.scatter(zm, gml, c='#99cc66', edgecolor='None', label='Calverley et al.~2011', s=64) 
    ax.errorbar(zm, gml, ecolor='#99CC66', capsize=5, elinewidth=2, capthick=2,
                yerr=np.vstack((gml_low, gml_up)), fmt='None', zorder=1, mfc='#99CC66',
                mec='#99CC66', mew=1, ms=5)

    c = np.array([x.gammapi[2]+12.0 for x in individuals])
    u = np.array([x.gammapi[0]+12.0 for x in individuals])
    l = np.array([x.gammapi[1]+12.0 for x in individuals])

    gml = 10.0**c
    gml_up = 10.0**u-10.0**c
    gml_low = 10.0**c - 10.0**l
    
    zs = np.array([x.z.mean() for x in individuals])
    uz = np.array([x.z.max() for x in individuals])
    lz = np.array([x.z.min() for x in individuals])


    uz = np.array([x[0] for x in zlims])
    lz = np.array([x[1] for x in zlims])
    
    uzerr = uz-zs
    lzerr = zs-lz 

    ax.scatter(zs, gml, c='#ffffff', edgecolor='k',
               label='Individual fits ($M<-18$, local source approximation)',
               s=44, zorder=4, linewidths=2) 
    ax.errorbar(zs, gml, ecolor='k', capsize=0, fmt='None', elinewidth=2,
                yerr=np.vstack((gml_low,gml_up)),
                xerr=np.vstack((lzerr,uzerr)), 
                mfc='#ffffff', mec='#404040', zorder=3, mew=1,
                ms=5)

    plt.legend(loc='lower left', fontsize=14, handlelength=1,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01, scatterpoints=1)
    
    plt.savefig('gammapi.pdf',bbox_inches='tight')
    plt.close('all')

    return

def emissivity_MH15(z):

    # Madau and Haardt 2015 Equation (1) 
    
    loge = 25.15*np.exp(-0.0026*z) - 1.5*np.exp(-1.3*z)

    return 10.0**loge # erg s^-1 Hz^-1 Mpc^-3

def emissivity_HM12(z):

    # Haardt and Madau 2012 Equation (37) 
    
    e = 10.0**24.6 * (1.0+z)**4.68 * np.exp(-0.28*z) / (np.exp(1.77*z)+26.3)

    return e # erg s^-1 Hz^-1 Mpc^-3

def emissivity_Manti17(z):

    # Manti et al. 2017 (MNRAS 466 1160) Equation (9) 

    loge = 23.59 + 0.55*z - 0.062*z**2 + 0.0047*z**3 - 0.0012*z**4
    
    return 10.0**loge # erg s^-1 Hz^-1 Mpc^-3


def draw_emissivity(all_individuals, zlims, composite=None, select=False):

    """
    Calculates and plots LyC emissivity.

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\epsilon_{912}$ [erg s$^{-1}$ Hz$^{-1}$ cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7.)

    ax.set_yscale('log')
    ax.set_ylim(1.0e23, 1.0e26)

    if select: 
        individuals = [x for x in all_individuals if x.z.mean() < 2.0 or x.z.mean() > 2.8]
    else:
        individuals = all_individuals
    

    for x in individuals:
        get_emissivity(x, x.z.mean())
    
    c = np.array([x.emissivity[2] for x in individuals])
    u = np.array([x.emissivity[0] for x in individuals])
    l = np.array([x.emissivity[1] for x in individuals])

    em = c
    em_up = u - c
    em_low = c - l 
    
    zs = np.array([x.z.mean() for x in individuals])
    uz = np.array([x.zlims[0] for x in individuals])
    lz = np.array([x.zlims[1] for x in individuals])
    
    uzerr = uz-zs
    lzerr = zs-lz 

    ax.scatter(zs, em, c='#ffffff', edgecolor='k',
               label='Individual fits ($M<-20$)',
               s=48, zorder=4, linewidths=1.5) 

    ax.errorbar(zs, em, ecolor='k', capsize=0, fmt='None', elinewidth=1.5,
                yerr=np.vstack((em_low, em_up)),
                xerr=np.vstack((lzerr, uzerr)), 
                mfc='#ffffff', mec='#404040', zorder=3, mew=1,
                ms=5)

    zg, eg, zg_lerr, zg_uerr, eg_lerr, eg_uerr = np.loadtxt('Data_new/giallongo15_emissivity.txt', unpack=True)
    
    eg *= 1.0e24
    eg_lerr *= 1.0e24
    eg_uerr *= 1.0e24
    ax.scatter(zg, eg, c='tomato', edgecolor='None',
               label='Giallongo et al.\ 2015 ($M<-18$)',
               s=72, zorder=4)

    ax.errorbar(zg, eg, ecolor='tomato', capsize=0, fmt='None', elinewidth=2,
                xerr=np.vstack((zg_lerr, zg_uerr)),
                yerr=np.vstack((eg_lerr, eg_uerr)), 
                zorder=3, mew=1)
                
    z = np.linspace(0, 7)
    e_MH15 = emissivity_MH15(z)
    ax.plot(z, e_MH15, lw=2, c='forestgreen', label='Madau and Haardt 2015')

    e_HM12 = emissivity_HM12(z)
    ax.plot(z, e_HM12, lw=2, c='dodgerblue', label='Haardt and Madau 2012')

    e_M17 = emissivity_Manti17(z)
    ax.plot(z, e_M17, lw=2, c='brown', label='Manti et al.\ 2017 ($M<-19$)')

    if composite is not None:
        zc = np.linspace(0, 7, 200)
        bf = np.median(composite.samples, axis=0)
        e = [emissivity(composite.log10phi, bf, x, (-30.0, -20.0)) for x in zc]
        for theta in composite.samples[np.random.randint(len(composite.samples),
                                                         size=300)]:
            e = [emissivity(composite.log10phi, theta, x, (-30.0, -20.0)) for x in zc]
            ax.plot(zc, e, c='goldenrod', alpha=0.1)
        ax.plot(zc, e, c='goldenrod', lw=2, label='Global model ($M<-20$)')
        ax.plot(zc, e, c='k', lw=2) 

    
    plt.legend(loc='upper right', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.01, scatterpoints=1)
    
    plt.savefig('emissivity.pdf',bbox_inches='tight')
    plt.close('all')

    return


