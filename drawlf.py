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

"""Makes LF plots at particular redshifts.  Shows data with
individual and composite models.  This is similar to the draw()
function in individual.py but is more flexible.

"""

def get_lf(lf, z_plot):

    # Bin data.  This is only for visualisation and to compare
    # with reported binned values.  The number of bins (nbins) is
    # estimated by Knuth's rule (astropy.stats.knuth_bin_width).

    m = lf.M1450[lf.p!=0.0]
    p = lf.p[lf.p!=0.0]
    sid = lf.sid[lf.p!=0.0]
    v = np.array([lf.quasar_volume(s) for s in sid])

    x =  kbw(m)
    print x, 3*x, 4*x
    nbins = int(np.ptp(m)/(3.3*kbw(m)))
    h = np.histogram(m,bins=nbins,weights=1.0/(p*v))
    nums = h[0]
    mags = (h[1][:-1] + h[1][1:])*0.5
    dmags = np.diff(h[1])*0.5

    phi = nums/np.diff(h[1])
    logphi = np.log10(phi) # cMpc^-3 mag^-1

    # Calculate errorbars on our binned LF.  These have been estimated
    # using Equations 1 and 2 of Gehrels 1986 (ApJ 303 336), as
    # implemented in astropy.stats.poisson_conf_interval.  The
    # interval='frequentist-confidence' option to that astropy function is
    # exactly equal to the Gehrels formulas, although the documentation
    # does not say so.
    n = np.histogram(lf.M1450,bins=nbins)[0]
    nlims = pci(n,interval='frequentist-confidence')
    nlims *= phi/n 
    uperr = np.log10(nlims[1]) - logphi 
    downerr = logphi - np.log10(nlims[0])

    return mags, dmags, logphi, uperr, downerr

def plot_composite_lf(ax, composite, mag_plot, z_plot):

    # Plot posterior samples from composit fit
    for theta in composite.samples[np.random.randint(len(composite.samples), size=900)]:
        phi_fit = composite.log10phi(theta, mag_plot, z_plot)
        ax.plot(mag_plot, phi_fit, lw=1, c='#00c0ff', alpha=0.1)

    # Plot best fit LF from composit fit 
    phi_fit = composite.log10phi(composite.bf.x, mag_plot, z_plot)
    ax.plot(mag_plot, phi_fit, lw=2, c='#00c0ff',
            label='Evolving double power law model')
    ax.plot(mag_plot, phi_fit, lw=1, c='k')

    return 
    
def draw(lf, z_plot, composite=None, dirname='', plotlit=False):
    """
    Plot data, best fit LF, and posterior LFs.

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)

    mag_plot = np.linspace(-30.0,-22.0,num=100) 
    lf.plot_posterior_sample_lfs(ax, mag_plot, lw=1,
                                   c='#ffbf00', alpha=0.1, zorder=2)
    label = 'Double power law at $z={0:.3f}$'.format(z_plot)
    lf.plot_bestfit_lf(ax, mag_plot, lw=2,
                         c='#ffbf00', label=label, zorder=3)

    if composite is not None:
        plot_composite_lf(ax, composite, mag_plot, z_plot)

    mags, dmags, logphi, uperr, downerr = get_lf(lf, z_plot)
    ax.scatter(mags, logphi, c='r', edgecolor='None', zorder=4)
    ax.errorbar(mags, logphi, ecolor='r', capsize=0,
                xerr=dmags,
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)

    if plotlit: 
        lf.plot_literature(ax, z_plot)

    ax.set_xlim(-29.0, -22.0)
    ax.set_ylim(-12.0, -5.0) 

    ax.set_xlabel(r'$M_{1450}$')
    ax.set_ylabel(r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$')

    plt.legend(loc='lower right', fontsize=14, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.2, scatterpoints=1)

    plottitle = r'$\langle z\rangle={0:.3f}$'.format(z_plot)
    plt.title(plottitle, size='medium', y=1.01)

    plotfile = dirname+'lf_z{0:.3f}.pdf'.format(z_plot)
    plt.savefig(plotfile, bbox_inches='tight')

    plt.close('all') 

    return 
