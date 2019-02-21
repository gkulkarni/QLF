import numpy as np
import emcee
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '16'
import matplotlib.pyplot as plt
from astropy.stats import knuth_bin_width  as kbw
from astropy.stats import poisson_conf_interval as pci

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


def cumVol(selmap, Mlim, zrange):

    v = 0.0
    for i in xrange(selmap.m.size):
        if (selmap.m[i] < Mlim):
            if (selmap.z[i] >= zrange[0]) and (selmap.z[i] < zrange[1]):
                    v += selmap.volarr[i]*selmap.p[i]

    return v
    
    
def binVol(selmap, mrange, zrange):

    """

    Calculate volume in an M-z bin for *one* selmap.

    """

    v = 0.0
    for i in xrange(selmap.m.size):
        if (selmap.m[i] >= mrange[0]) and (selmap.m[i] < mrange[1]):
            if (selmap.z[i] >= zrange[0]) and (selmap.z[i] < zrange[1]):
                if selmap.sid == 7: # Giallongo 
                    v += selmap.volarr[i]*selmap.p[i]*selmap.dm[i]
                else:
                    v += selmap.volarr[i]*selmap.p[i]*selmap.dm_array[i]

    return v


def totBinVol(lf, m, mbins, selmaps):

    """

    Given magnitude bins mbins and a list of selection maps
    selmaps, compute the volume for an object with magnitude m.

    """

    idx = np.searchsorted(mbins, m)
    mlow = mbins[idx-1]
    mhigh = mbins[idx]
    mrange = (mlow, mhigh)

    v = np.array([binVol(x, mrange, lf.zlims) for x in selmaps])
    total_vol = v.sum() 

    return total_vol


def get_rhoqso(lf, sid, Mlim):

    m = lf.M1450[lf.sid==sid]
    selmaps = [x for x in lf.maps if x.sid == sid]
    n = np.size(m[m < Mlim])
    v = np.array([cumVol(x, Mlim, lf.zlims) for x in selmaps])

    return n/np.sum(v)


def get_lf(lf, sid, bins):
    
    # Bin data.  This is only for visualisation and to compare
    # with reported binned values.  
    m = lf.M1450[lf.sid==sid]

    selmaps = [x for x in lf.maps if x.sid == sid]

    v1 = np.array([totBinVol(lf, x, bins, selmaps) for x in m])

    v1_nonzero = v1[np.where(v1>0.0)]
    m = m[np.where(v1>0.0)]

    h = np.histogram(m, bins=bins, weights=1.0/(v1_nonzero))

    nums = h[0]
    mags = (h[1][:-1] + h[1][1:])*0.5
    dmags = np.diff(h[1])*0.5

    left = mags - h[1][:-1]
    right = h[1][1:] - mags

    phi = nums
    logphi = np.log10(phi) # cMpc^-3 mag^-1

    # Calculate errorbars on our binned LF.  These have been estimated
    # using Equations 1 and 2 of Gehrels 1986 (ApJ 303 336), as
    # implemented in astropy.stats.poisson_conf_interval.  The
    # interval='frequentist-confidence' option to that astropy function is
    # exactly equal to the Gehrels formulas, although the documentation
    # does not say so.
    n = np.histogram(m, bins=bins)[0]
    nlims = pci(n,interval='frequentist-confidence')
    nlims *= phi/n 
    uperr = np.log10(nlims[1]) - logphi 
    downerr = logphi - np.log10(nlims[0])

    return mags, left, right, logphi, uperr, downerr


def rhoqso2(lfs, mag_threshold):

    zs = []
    rhos = []

    for x in lfs:
        z = (x.zlims[0]+x.zlims[1])/2
        sids = np.unique(x.sid)

        rhos_thiszbin = []
        
        for sid in sids:
            rho = get_rhoqso(x, sid, mag_threshold)
            rhos_thiszbin.append(rho)
            
        rhos.append(np.mean(rhos_thiszbin))
        zs.append(z)
            
    return zs, rhos 
            
            
def rhoqso_old(lfs, mag_threshold, bins):

    zs = []
    rhos = [] 

    for x in lfs:

        z = (x.zlims[0]+x.zlims[1])/2
        
        sids = np.unique(x.sid)

        rhos_thiszbin = []
        
        for sid in sids: 
            mags, left, right, logphi, uperr, downerr = get_lf(x, sid, bins)

            dm = left + right

            phidm = 10.0**logphi * dm
            intphidm = np.cumsum(phidm)

            # for d in zip(dm, mags, logphi, phidm, intphidm):
            #     print '{:.2f}  {:.2f}  {:.2f}  {:.2e}  {:.2e}'.format(*d)
            
            if (mag_threshold > np.min(mags)) and (mag_threshold < np.max(mags)):
                assert(np.all(np.diff(mags) > 0))

                rho = np.interp(mag_threshold, mags, intphidm)
                
                rhos_thiszbin.append(rho)
                
                # print 'Mlim = {:.2f}  z = {:.2f}  rho = {:.3e}'.format(mag_threshold, z, rho)
                
        rhos.append(np.mean(rhos_thiszbin))
        zs.append(z)
        
    return zs, rhos 


def rhoqso3(lfs, mag_threshold, bins):

    zs = []
    rhos = [] 
    uerr = []
    lerr = []
    
    for x in lfs:

        z = (x.zlims[0]+x.zlims[1])/2
        sids = np.unique(x.sid)

        logphis = []
        for sid in sids: 
            mags, left, right, logphi, uperr, downerr = get_lf(x, sid, bins)
            logphis.append(10.0**logphi)
        logphi = np.sum(logphis, axis=0)

        dm = left + right
        phidm = logphi * dm 
        intphidm = np.cumsum(phidm)

        assert(np.all(np.diff(mags) > 0))
        rho = np.interp(mag_threshold, mags, intphidm)
        nqso = np.size(x.M1450[x.M1450 < mag_threshold])

        if nqso > 0: 
            nlims = pci(nqso,interval='frequentist-confidence')
            nlims *= rho/nqso
            uperr = nlims[1] - rho
            downerr = rho - nlims[0]
        else:
            uperr = 0.0
            downerr = 0.0
            
        if np.max(mags[logphi > 0.0]) < mag_threshold:
            # not reaching threshold
            rho = 0.0
            uperr = 0.0
            downerr = 0.0
                
        rhos.append(rho)
        zs.append(z)
        uerr.append(uperr)
        lerr.append(downerr)

    return zs, rhos, uerr, lerr


def global_cumulative(ax, composite, mlim, color, **kwargs):

    nzs = 500
    z = np.linspace(0, 7, nzs)
    nsample = 300
    rsample = composite.samples[np.random.randint(len(composite.samples), size=nsample)]

    # bf = np.median(composite.samples, axis=0)
    # r = np.array([rhoqso(composite.log10phi, bf, mlim, x, fit='composite') for x in z])
    # ax.plot(z, r, color='k', zorder=7)

    r = np.zeros((nsample, nzs))
    for i, theta in enumerate(rsample):
        r[i] = np.array([rhoqso(composite.log10phi, theta, mlim, x, fit='composite') for x in z])

    up = np.percentile(r, 15.87, axis=0)
    down = np.percentile(r, 84.13, axis=0)
    f = ax.fill_between(z, down, y2=up, color=color, zorder=6, alpha=0.7, **kwargs)

    c = np.median(r, axis=0)

    if color == 'forestgreen':
        p, = ax.plot(z, c, color=color, zorder=7)
    
    if color == 'peru': 
        p, = ax.plot(z, c, color='brown', zorder=7)

    if color == 'grey': 
        p, = ax.plot(z, c, color='k', zorder=7)
        
    return f, p 


def draw_withGlobal_multiple(c1, c2, c3, individuals, select=False):

    fig = plt.figure(figsize=(7, 11), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\rho(z, M_{1450} < M_\mathrm{lim})$ [cMpc$^{-3}$]')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7)

    ax.set_yscale('log')
    ax.set_ylim(1.0e-10, 1.0e-3)

    mlim = -21
    zs, rhos, uperr, downerr = rhoqso3(individuals, mlim, bins=np.arange(-30.9, -17.3, 0.6))
    ax.scatter(zs, rhos, c='forestgreen', edgecolor='None', s=42, zorder=10, linewidths=2, label='$M_{1450}<-21$') 
    ax.errorbar(zs, rhos, ecolor='forestgreen', capsize=0, fmt='None', elinewidth=1, 
                yerr=np.vstack((downerr, uperr)),
                #xerr=np.vstack((lzerr, uzerr)), 
                zorder=10, mew=1, ms=5) 
    m1f, m1 = global_cumulative(ax, c1, mlim, 'grey')
    m2f, m2 = global_cumulative(ax, c2, mlim, 'forestgreen')
    m3f, m3 = global_cumulative(ax, c3, mlim, 'peru')
    
    mlim = -23
    zs, rhos, uperr, downerr = rhoqso3(individuals, mlim, bins=np.arange(-30.9, -17.3, 0.1))
    ax.scatter(zs, rhos, c='red', edgecolor='None', s=42, zorder=10, linewidths=2, label='$M_{1450}<-23$') 
    ax.errorbar(zs, rhos, ecolor='red', capsize=0, fmt='None', elinewidth=1, 
                yerr=np.vstack((downerr, uperr)),
                #xerr=np.vstack((lzerr, uzerr)), 
                zorder=10, mew=1, ms=5)
    m1f, m1 = global_cumulative(ax, c1, mlim, 'grey')
    m2f, m2 = global_cumulative(ax, c2, mlim, 'forestgreen')
    m3f, m3 = global_cumulative(ax, c3, mlim, 'peru')    

    mlim = -24
    zs, rhos, uperr, downerr = rhoqso3(individuals, mlim, bins=np.arange(-30.9, -17.3, 0.1))
    ax.scatter(zs, rhos, c='goldenrod', edgecolor='None', s=42, zorder=10, linewidths=2, label='$M_{1450}<-24$') 
    ax.errorbar(zs, rhos, ecolor='goldenrod', capsize=0, fmt='None', elinewidth=1,
                yerr=np.vstack((downerr, uperr)),
                #xerr=np.vstack((lzerr, uzerr)), 
                zorder=10, mew=1, ms=5)
    m1f, m1 = global_cumulative(ax, c1, mlim, 'grey')
    m2f, m2 = global_cumulative(ax, c2, mlim, 'forestgreen')
    m3f, m3 = global_cumulative(ax, c3, mlim, 'peru')    

    mlim = -25
    zs, rhos, uperr, downerr = rhoqso3(individuals, mlim, bins=np.arange(-30.9, -17.3, 0.1))
    ax.scatter(zs, rhos, c='peru', edgecolor='None', s=42, zorder=10, linewidths=2, label='$M_{1450}<-25$') 
    ax.errorbar(zs, rhos, ecolor='peru', capsize=0, fmt='None', elinewidth=1,
                yerr=np.vstack((downerr, uperr)),
                #xerr=np.vstack((lzerr, uzerr)), 
                zorder=10, mew=1, ms=5)
    m1f, m1 = global_cumulative(ax, c1, mlim, 'grey')
    m2f, m2 = global_cumulative(ax, c2, mlim, 'forestgreen')
    m3f, m3 = global_cumulative(ax, c3, mlim, 'peru')    
    
    mlim = -26
    zs, rhos, uperr, downerr = rhoqso3(individuals, mlim, bins=np.arange(-30.9, -17.3, 0.1))
    ax.scatter(zs, rhos, c='tomato', edgecolor='None', s=42, zorder=10, linewidths=2, label='$M_{1450}<-26$') 
    ax.errorbar(zs, rhos, ecolor='tomato', capsize=0, fmt='None', elinewidth=1,
                yerr=np.vstack((downerr, uperr)),
                #xerr=np.vstack((lzerr, uzerr)), 
                zorder=10, mew=1, ms=5)
    m1f, m1 = global_cumulative(ax, c1, mlim, 'grey')
    m2f, m2 = global_cumulative(ax, c2, mlim, 'forestgreen')
    m3f, m3 = global_cumulative(ax, c3, mlim, 'peru')    
    
    mlim = -27
    zs, rhos, uperr, downerr = rhoqso3(individuals, mlim, bins=np.arange(-30.9, -17.3, 0.1))
    ax.scatter(zs, rhos, c='saddlebrown', edgecolor='None', s=42, zorder=10, linewidths=2, label='$M_{1450}<-27$') 
    ax.errorbar(zs, rhos, ecolor='saddlebrown', capsize=0, fmt='None', elinewidth=1,
                yerr=np.vstack((downerr, uperr)),
                #xerr=np.vstack((lzerr, uzerr)), 
                zorder=10, mew=1, ms=5)
    m1f, m1 = global_cumulative(ax, c1, mlim, 'grey')
    m2f, m2 = global_cumulative(ax, c2, mlim, 'forestgreen')
    m3f, m3 = global_cumulative(ax, c3, mlim, 'peru')

    plt.legend(loc='upper left', fontsize=14, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.3, borderpad=0.1,
               scatterpoints=1)
    
    plt.savefig('rhoqso_data.pdf',bbox_inches='tight')
    plt.close('all')

    return
