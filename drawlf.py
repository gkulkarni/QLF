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

def volume(z, area, cosmo=cosmo):

    omega = (area/41253.0)*4.0*np.pi # str
    volperstr = cd.diff_comoving_volume(z,**cosmo) # cMpc^3 str^-1 dz^-1

    return omega*volperstr # cMpc^3 dz^-1

def get_lf(lf, z_plot, sids):

    sample_selection = np.in1d(lf.sid, sids)
    m = lf.M1450[sample_selection]
    p = lf.p[sample_selection]
    sid = lf.sid[sample_selection]
    
    m = m[p!=0.0]
    p = p[p!=0.0]
    sid = sid[p!=0.0]

    # NDWFS
    area = 1.71 # deg^2
    dz = 0.02 
    dm = 0.05
    zsel, msel, psel = np.loadtxt('Data/glikman11_selfunc_ndwfs.dat', usecols=(1,2,3), unpack=True)
    vol = volume(zsel, area)*dz
    
    def volm(mag, magsel, probsel, volsel):
        # Below, np.isclose() selects the tile to which this magnitude
        # belongs at each redshift.  We then select the tiles for
        # which selection probability is non-zero (probsel[t]>0.0) and
        # sum the volume of all those tiles.
        t = np.isclose(magsel, mag, rtol=0.0, atol=dm*0.5)
        return np.sum(volsel[t][probsel[t]>0.0], dtype=np.float64)
    
    v1 = np.array([volm(x, msel, psel, vol) for x in m[:12]])

    # DLS
    area = 2.05 # deg^2
    dz = 0.02 
    dm = 0.05
    zsel, msel, psel = np.loadtxt('Data/glikman11_selfunc_dls.dat', usecols=(1,2,3), unpack=True)
    vol = volume(zsel, area)*dz

    v2 = np.array([volm(x, msel, psel, vol) for x in m[12:]])

    v = np.concatenate((v1, v2))     
    
    bins = [-26.0, -25.0, -24.0, -23.0, -22.0, -21]
    h = np.histogram(m[1:],bins=bins,weights=1.0/(p[1:]*v[1:]))
    
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
    n = np.histogram(m,bins=bins)[0]

    nlims = pci(n,interval='frequentist-confidence')
    nlims *= phi/n 
    uperr = np.log10(nlims[1]) - logphi 
    downerr = logphi - np.log10(nlims[0])

    mags, b, c = bs(m, m, bins=bins)

    print mags
    print b
    print c

    print np.unique(c, return_counts=True)
    print m[c==1]
    print m[c==1].mean()

    left = mags - b[:-1]
    right = b[1:] - mags

    return mags, left, right, logphi, uperr, downerr

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

    mag_plot = np.linspace(-30.0,-20.0,num=100) 
    lf.plot_posterior_sample_lfs(ax, mag_plot, lw=1,
                                   c='#ffbf00', alpha=0.1, zorder=2)
    label = 'Double power law at $z={0:.3f}$'.format(z_plot)
    lf.plot_bestfit_lf(ax, mag_plot, lw=2,
                         c='#ffbf00', label=label, zorder=3)

    if composite is not None:
        plot_composite_lf(ax, composite, mag_plot, z_plot)

    mags, left, right, logphi, uperr, downerr = get_lf(lf, z_plot, [6, 15])

    ax.scatter(mags, logphi, c='r', edgecolor='None', zorder=4, label='my binning')
    ax.errorbar(mags, logphi, ecolor='r', capsize=0,
                xerr=np.vstack((left, right)),
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)

    if plotlit: 
        lf.plot_literature(ax, z_plot)

    f_glikman = 'Data/glikman11.dat'
    (M1450_glikman, M1450_glikman_lerr, M1450_glikman_uerr, phi_glikman,
     phi_glikman_uerr, phi_glikman_lerr) = np.loadtxt(f_glikman, unpack=True)

    phi_glikman *= 1.0e-8
    phi_glikman_lerr *= 1.0e-8
    phi_glikman_uerr *= 1.0e-8

    phi_glikman_llogerr = np.log10(phi_glikman)-np.log10(phi_glikman-phi_glikman_lerr)
    phi_glikman_ulogerr = np.log10(phi_glikman+phi_glikman_uerr)-np.log10(phi_glikman) 
    phi_glikman = np.log10(phi_glikman) 

    ax.scatter(M1450_glikman[4:], phi_glikman[4:], c='b', label='Glikman et al.\ 2011', edgecolor='None',zorder=303) 
    ax.errorbar(M1450_glikman[4:], phi_glikman[4:], ecolor='b', capsize=0,
                xerr=np.vstack((M1450_glikman_lerr[4:], M1450_glikman_uerr[4:])),
                yerr=np.vstack((phi_glikman_llogerr[4:], phi_glikman_ulogerr[4:])), fmt=None,zorder=304)
    
    ax.set_xlim(-29.0, -20.0)
    ax.set_ylim(-12.0, -4.0) 

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
