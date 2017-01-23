import numpy as np
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

def plot_posterior_sample_lfs(lf, ax, mags, **kwargs):

    random_thetas = lf.samples[np.random.randint(len(lf.samples), size=1000)]
    for theta in random_thetas:
        phi_fit = lf.log10phi(theta, mags)
        ax.plot(mags, phi_fit, **kwargs)

    return

def plot_bestfit_lf(lf, ax, mags, **kwargs):

    phi_fit = lf.log10phi(lf.bf.x, mags)
    ax.plot(mags, phi_fit, **kwargs)
    ax.plot(mags, phi_fit, lw=1, c='k', zorder=kwargs['zorder'])

    return

def binVol(lf, selmap, mrange, zrange):

    """

    Calculate volume in an M-z bin for *one* selmap.

    """

    v = 0.0
    for i in xrange(selmap.m.size):
        if (selmap.m[i] >= mrange[0]) and (selmap.m[i] < mrange[1]):
            if (selmap.z[i] >= zrange[0]) and (selmap.z[i] < zrange[1]):
                v += selmap.volarr[i]*selmap.p[i]*selmap.dm

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


def get_lf(lf, sid, z_plot):

    # Bin data.  This is only for visualisation and to compare
    # with reported binned values.  

    z = lf.z[lf.sid==sid]
    m = lf.M1450[lf.sid==sid]
    p = lf.p[lf.sid==sid]

    selmaps = [x for x in lf.maps if x.sid == sid]

    if sid==6:
        # Glikman's sample needs wider bins.
        bins = np.array([-26.0, -25.0, -24.0, -23.0, -22.0, -21])
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

def render(ax, lf):

    """

    Plot data, best fit LF, and posterior LFs.

    """

    z_plot = lf.z.mean() 
    
    mag_plot = np.linspace(-32.0, -16.0, num=200) 
    plot_posterior_sample_lfs(lf, ax, mag_plot, lw=1,
                                   c='#ffbf00', alpha=0.1, zorder=2) 
    plot_bestfit_lf(lf, ax, mag_plot, lw=2,
                         c='#ffbf00', zorder=3)

    cs = {13: 'r', 15:'g', 1:'b', 17:'m', 8:'c', 6:'#ff7f0e',
          7:'#8c564b', 18:'#7f7f7f', 10:'#17becf', 11:'r'}

    def dsl(i):
        for x in lf.maps:
            if x.sid == i:
                return x.label
        return

    sids = np.unique(lf.sid)
    for i in sids:
        mags, left, right, logphi, uperr, downerr = get_lf(lf, i, z_plot)
        ax.scatter(mags, logphi, c=cs[i], edgecolor='None', zorder=4, s=16, label=dsl(i))
        ax.errorbar(mags, logphi, ecolor=cs[i], capsize=0,
                    xerr=np.vstack((left, right)), 
                    yerr=np.vstack((uperr, downerr)),
                    fmt='None', zorder=4)

    return 

def draw(lf, composite=None, dirname=''):

    """

    Plot data, best fit LF, and posterior LFs.

    """

    z_plot = lf.z.mean() 
    
    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)

    render(ax, lf)

    ax.set_xlim(-17.0, -31.0)
    ax.set_ylim(-12.0, -5.0)
    ax.set_xticks(np.arange(-31,-16, 2))

    ax.set_xlabel(r'$M_{1450}$')
    ax.set_ylabel(r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$')

    legend_title = r'$\langle z\rangle={0:.3f}$'.format(z_plot)
    plt.legend(loc='lower left', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.2, scatterpoints=1, title=legend_title)

    plottitle = r'${:g}\leq z<{:g}$'.format(lf.zlims[0], lf.zlims[1]) 
    plt.title(plottitle, size='medium', y=1.01)

    plotfile = dirname+'lf_z{0:.3f}.pdf'.format(z_plot)
    plt.savefig(plotfile, bbox_inches='tight')

    plt.close('all') 

    return 
    
