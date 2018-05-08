import numpy as np
import emcee
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
from astropy.stats import knuth_bin_width  as kbw
from astropy.stats import poisson_conf_interval as pci
from scipy.stats import binned_statistic as bs
import cosmolopy.distance as cd
cosmo = {'omega_M_0':0.3,
         'omega_lambda_0':0.7,
         'omega_k_0':0.0,
         'h':0.70}


"""Makes LF plots at particular redshifts.  Shows data with
individual and composite models.  This is similar to the draw()
function in individual.py but is more flexible.

"""

def lfsample(theta, n, mlims):

    """
    Return n qso magnitudes between mlims[0] and mlims[1] when the LF
    is described by parameters theta.

    """

    mmin = mlims[0]
    mmax = mlims[1]

    def lnprob(x, theta):

        if x < mmax and x > mmin: 
            mag = x 
            log10phi_star, M_star, alpha, beta = theta 
            phi = 10.0**log10phi_star / (10.0**(0.4*(alpha+1)*(mag-M_star)) +
                                         10.0**(0.4*(beta+1)*(mag-M_star)))
            return np.log(phi)
        else:
            return -np.inf 
    
    ndim = 1 
    nwalkers = 250
    dm = np.abs(mmin-mmax)
    p0 = (np.random.rand(ndim*nwalkers)*dm + mmin).reshape((nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[theta])
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 1000)
    sample = sampler.flatchain[:,0]

    return np.random.choice(sample, n)
    
def plot_posterior_sample_lfs(lf, ax, maglims, **kwargs):

    nmags = 100
    mags = np.linspace(*maglims, num=nmags)
    nsample = 1000
    rsample = lf.samples[np.random.randint(len(lf.samples), size=nsample)]
    phi = np.zeros((nsample, nmags))
    
    for i, theta in enumerate(rsample):
        phi[i] = lf.log10phi(theta, mags)

    up = np.percentile(phi, 15.87, axis=0)
    down = np.percentile(phi, 84.13, axis=0)
    f = ax.fill_between(mags, down, y2=up, color='#ffbf00', alpha=0.7)
    #f = ax.fill_between(mags, down, y2=up, color='grey', alpha=0.7)

    return f

def plot_bestfit_lf(lf, ax, mags, **kwargs):

    bf = np.median(lf.samples, axis=0)
    phi_fit = lf.log10phi(bf, mags)
    # ax.plot(mags, phi_fit, lw=1.5, c='k', zorder=kwargs['zorder'], label=kwargs['label'])
    bf, = ax.plot(mags, phi_fit, lw=1.5, c='#ffbf00', zorder=kwargs['zorder'])
    #bf, = ax.plot(mags, phi_fit, lw=1.5, c='k', zorder=kwargs['zorder'])

    return bf 

def binVol(self, selmap, mrange, zrange):

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
                    v += selmap.volarr[i]*selmap.p[i]*selmap.dm

    return v


def binVol_all(self, selmap, mrange, zrange):

    """

    Calculate volume in an M-z bin for *one* selmap.

    """

    v = 0.0
    for i in xrange(selmap.m_all.size):
        if (selmap.m_all[i] >= mrange[0]) and (selmap.m_all[i] < mrange[1]):
            if (selmap.z_all[i] >= zrange[0]) and (selmap.z_all[i] < zrange[1]):
                if selmap.sid == 7: # Giallongo 
                    v += selmap.volarr_all[i]*selmap.p_all[i]*selmap.dm[i]
                else:
                    v += selmap.volarr_all[i]*selmap.p_all[i]*selmap.dm

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

    v = np.array([binVol(lf, x, mrange, lf.zlims) for x in selmaps])
    total_vol = v.sum() 

    return total_vol


def totBinVol_all(lf, m, mbins, selmaps):

    """

    Given magnitude bins mbins and a list of selection maps
    selmaps, compute the volume for an object with magnitude m.

    """

    idx = np.searchsorted(mbins, m)
    mlow = mbins[idx-1]
    mhigh = mbins[idx]
    mrange = (mlow, mhigh)

    v = np.array([binVol_all(lf, x, mrange, lf.zlims) for x in selmaps])
    total_vol = v.sum() 

    return total_vol


def get_lf(lf, sid, z_plot):
    
    # Bin data.  This is only for visualisation and to compare
    # with reported binned values.  
    m = lf.M1450[lf.sid==sid]

    selmaps = [x for x in lf.maps if x.sid == sid]

    if sid==6:
        # Glikman's sample needs wider bins.
        bins = np.array([-26.0, -25.0, -24.0, -23.0, -22.0, -21])
    elif sid == 7:
        bins = np.array([-23.5, -21.5, -20.5, -19.5, -18.5])
    elif sid == 10 or sid == 18:
        bins = np.arange(-30.9, -17.3, 1.8)
    else:
        bins = np.arange(-30.9, -17.3, 0.6)

    v1 = np.array([totBinVol_all(lf, x, bins, selmaps) for x in m])

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

    # print 'sid=', sid 
    # print 'mags=', mags
    # print 'nums=', nums
    # print 'total=', np.sum(nums)

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


def get_lf_all(lf, sid, z_plot):

    # Bin data.  This is only for visualisation and to compare
    # with reported binned values.  

    m = lf.M1450_all[lf.sid_all==sid]

    selmaps = [x for x in lf.maps if x.sid == sid]

    if sid==6:
        # Glikman's sample needs wider bins.
        bins = np.array([-26.0, -25.0, -24.0, -23.0, -22.0, -21])
    elif sid == 7:
        bins = np.array([-23.5, -21.5, -20.5, -19.5, -18.5])
    elif sid == 10 or sid == 18:
        bins = np.arange(-30.9, -17.3, 1.8)
    else:
        bins = np.arange(-30.9, -17.3, 0.6)

    v1 = np.array([totBinVol_all(lf, x, bins, selmaps) for x in m])

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


def get_lf_sample(lf, sid, z_plot):

    # Bin data.  This is only for visualisation and to compare
    # with reported binned values.  

    m = lf.M1450[lf.sid==sid]
    n = m.size
    mlims = (m.min(), m.max())
    theta = np.median(lf.samples, axis=0)
    m = lfsample(theta, n, mlims)

    selmaps = [x for x in lf.maps if x.sid == sid]

    if sid==6:
        # Glikman's sample needs wider bins.
        bins = np.array([-26.0, -25.0, -24.0, -23.0, -22.0, -21])
    elif sid == 7:
        bins = np.array([-23.5, -21.5, -20.5, -19.5, -18.5])
    elif sid == 10:
        bins = np.arange(-30.9, -17.3, 1.8)
    else:
        bins = np.arange(-30.9, -17.3, 0.6)

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


def plot_giallongo_z5p75(lf, ax, mags):

    M_star_giallongo = -23.4
    log10phi_star_giallongo = -5.8
    beta = -1.66 # Giallongo et al. call this -beta
    alpha = -3.35 # Giallongo et al. call this -gamma

    p = (log10phi_star_giallongo, M_star_giallongo, alpha, beta)
    
    phi_fit = lf.log10phi(p, mags)
    ax.plot(mags, phi_fit, lw=2, c='r', zorder=100, label=r'Giallongo et al.\ 2015 fit at $z=5.75$', dashes=[7,2])

    return 


def plot_giallongo_z4p75(lf, ax, mags):

    M_star_giallongo = -23.6
    log10phi_star_giallongo = -5.7
    beta = -1.81 # Giallongo et al. call this -beta
    alpha = -3.14 # Giallongo et al. call this -gamma

    p = (log10phi_star_giallongo, M_star_giallongo, alpha, beta)
    
    phi_fit = lf.log10phi(p, mags)
    ax.plot(mags, phi_fit, lw=2, c='r', zorder=100, label=r'Giallongo et al.\ 2015 fit at $z=4.75$', dashes=[7,2])

    return 


def plot_giallongo_z4p25(lf, ax, mags):

    M_star_giallongo = -23.2
    log10phi_star_giallongo = -5.2
    beta = -1.52 # Giallongo et al. call this -beta
    alpha = -3.13 # Giallongo et al. call this -gamma

    p = (log10phi_star_giallongo, M_star_giallongo, alpha, beta)
    
    phi_fit = lf.log10phi(p, mags)
    ax.plot(mags, phi_fit, lw=2, c='r', zorder=100, label=r'Giallongo et al.\ 2015 fit at $z=4.25$', dashes=[7,2])

    return 


def render(ax, lf, composite=None, showMockSample=False, show_individual_fit=True, c2=None, c3=None):

    """

    Plot data, best fit LF, and posterior LFs.

    """

    z_plot = lf.z.mean() 

    if show_individual_fit: 
        mag_plot = np.linspace(-32.0, -16.0, num=200) 
        indf = plot_posterior_sample_lfs(lf, ax, (-32.0, -16.0), lw=1,
                                       c='#ffbf00', alpha=0.1, zorder=2) 
        # plot_bestfit_lf(lf, ax, mag_plot, lw=2,
        #                      c='#ffbf00', zorder=3, label='This work')

        indbf = plot_bestfit_lf(lf, ax, mag_plot, lw=2,
                             c='#ffbf00', zorder=3)
        
        # if z_plot < 4.5:
        #     plot_giallongo_z4p25(lf, ax, mag_plot)

        # if z_plot < 5.5 and z_plot > 4.7:
        #     plot_giallongo_z4p75(lf, ax, mag_plot)

        # if z_plot > 5.5:
        #     plot_giallongo_z5p75(lf, ax, mag_plot)
            

    if composite is not None:

        nmags = 200 
        mags = np.linspace(-32.0, -16.0, num=nmags)
        bf = np.median(composite.samples, axis=0)
        nsample = 1000
        rsample = composite.samples[np.random.randint(len(composite.samples), size=nsample)]
        phi = np.zeros((nsample, nmags))
        
        for i, theta in enumerate(rsample):
            phi[i] = composite.log10phi(theta, mags, z_plot)

        up = np.percentile(phi, 15.87, axis=0)
        down = np.percentile(phi, 84.13, axis=0)
        c1f = ax.fill_between(mags, down, y2=up, color='grey', zorder=6, alpha=0.7)

        p = np.median(phi, axis=0)
        c1bf, = ax.plot(mags, p, color='k', zorder=6, lw=1)


    if c2 is not None: 
        nmags = 200 
        mags = np.linspace(-32.0, -16.0, num=nmags)
        bf = np.median(c2.samples, axis=0)
        nsample = 1000
        rsample = c2.samples[np.random.randint(len(c2.samples), size=nsample)]
        phi = np.zeros((nsample, nmags))
        
        for i, theta in enumerate(rsample):
            phi[i] = c2.log10phi(theta, mags, z_plot)

        up = np.percentile(phi, 15.87, axis=0)
        down = np.percentile(phi, 84.13, axis=0)
        c2f = ax.fill_between(mags, down, y2=up, color='forestgreen', zorder=4, alpha=0.7)

        p = np.median(phi, axis=0)
        c2bf, = ax.plot(mags, p, color='forestgreen', zorder=4, lw=1)


    if c3 is not None: 
        nmags = 200 
        mags = np.linspace(-32.0, -16.0, num=nmags)
        bf = np.median(c3.samples, axis=0)
        nsample = 1000
        rsample = c3.samples[np.random.randint(len(c3.samples), size=nsample)]
        phi = np.zeros((nsample, nmags))
        
        for i, theta in enumerate(rsample):
            phi[i] = c3.log10phi(theta, mags, z_plot)

        up = np.percentile(phi, 15.87, axis=0)
        down = np.percentile(phi, 84.13, axis=0)
        c3f = ax.fill_between(mags, down, y2=up, color='peru', zorder=5, alpha=0.7)

        p = np.median(phi, axis=0)
        c3bf, = ax.plot(mags, p, color='brown', zorder=5, lw=1)
        
        
    cs = { 1 : '#1f77b4', # "blue"
           6 : '#17becf', # "cyan"
           7 : '#9467bd', # "purple"
           8 : '#8c564b', # "brown"
           9 : 'r', 
           10 : '#ff7f0e', # "orange"
           11 : '#7f7f7f', # "grey"
           13 : '#d62728', # "red"
           15 : '#2ca02c', # "green"
           17 : '#bcbd22', # "yellow"
           18 : '#e377c2' # "pink"
    }
    
    def dsl(i):
        for x in lf.maps:
            if x.sid == i:
                return x.label
        return

    sids = np.unique(lf.sid)
    
    bad_data_set = False
    if bad_data_set:
        for i in sids: 
            mags, left, right, logphi, uperr, downerr = get_lf(lf, i, z_plot)
            print mags[logphi>-100.0]
            print logphi[logphi>-100.0]
            ax.errorbar(mags, logphi, ecolor=cs[i], capsize=0,
                        xerr=np.vstack((left, right)), 
                        yerr=np.vstack((uperr, downerr)),
                        fmt='None', zorder=4)
            ax.scatter(mags, logphi, c='#ffffff', edgecolor=cs[i], zorder=4, s=16, label=dsl(i)+' (rejected bins)')
        return

    for i in sids[::-1]:

        mags, left, right, logphi, uperr, downerr = get_lf(lf, i, z_plot)

        print mags[logphi>-100.0]
        print logphi[logphi>-100.0]
        ax.scatter(mags, logphi, c=cs[i], edgecolor='None', zorder=4, s=20, label=dsl(i))
        ax.errorbar(mags, logphi, ecolor=cs[i], capsize=0,
                    xerr=np.vstack((left, right)), 
                    yerr=np.vstack((uperr, downerr)),
                    fmt='None', zorder=4)

        if i == 8:
            # No need to plot rejected bins for McGreer's data because
            # they were rejected for overlap with Yang, not due to
            # incompleteness.
            continue 
        
        mags_all, left_all, right_all, logphi_all, uperr_all, downerr_all = get_lf_all(lf, i, z_plot)
        print mags_all[logphi_all!=logphi]
        print logphi_all[logphi_all!=logphi]

        select = (logphi_all!=logphi)
        mags_all = mags_all[select]
        left_all = left_all[select]
        right_all = right_all[select]
        logphi_all = logphi_all[select]
        uperr_all  = uperr_all[select]
        downerr_all = downerr_all[select]

        if mags_all.any(): 
            ax.errorbar(mags_all, logphi_all, ecolor=cs[i], capsize=0,
                        xerr=np.vstack((left_all, right_all)), 
                        yerr=np.vstack((uperr_all, downerr_all)),
                        fmt='None', zorder=4)
            ax.scatter(mags_all, logphi_all, c='#ffffff', edgecolor=cs[i],
                       zorder=4, s=16, label=dsl(i)+' (rejected bin)')

    if showMockSample:
        for i in sids:
            mags, left, right, logphi, uperr, downerr = get_lf_sample(lf, i, z_plot)
            ax.scatter(mags, logphi, c='k', edgecolor='None', zorder=4, s=16, label=dsl(i))
            ax.errorbar(mags, logphi, ecolor='k', capsize=0,
                        xerr=np.vstack((left, right)), 
                        yerr=np.vstack((uperr, downerr)),
                        fmt='None', zorder=4)
        
    return # (indf, indbf), (c1f, c1bf), (c2f, c2bf), (c3f, c3bf) 

def draw(lf, composite=None, dirname='', showMockSample=False, show_individual_fit=True):

    """

    Plot data, best fit LF, and posterior LFs.

    """

    z_plot = lf.z.mean() 
    
    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)

    render(ax, lf, composite=composite, showMockSample=showMockSample,
           show_individual_fit=show_individual_fit)

    ax.set_xlim(-17.0, -31.0)
    ax.set_ylim(-12.0, -4.0)
    ax.set_xticks(np.arange(-31,-16, 2))

    ax.set_xlabel(r'$M_{1450}$')
    ax.set_ylabel(r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$')

    legend_title = r'$\langle z\rangle={0:.3f}$'.format(z_plot)
    plt.legend(loc='lower left', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.2, scatterpoints=1,
               title=legend_title)

    plottitle = r'${:g}\leq z<{:g}$'.format(lf.zlims[0], lf.zlims[1]) 
    plt.title(plottitle, size='medium', y=1.01)

    plotfile = dirname+'lf_z{0:.3f}.pdf'.format(z_plot)
    plt.savefig(plotfile, bbox_inches='tight')

    plt.close('all') 

    return 
    
