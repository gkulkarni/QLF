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

    # DR7
    area = 6248.0 # deg^2
    dz = 0.05 
    dm = 0.1
    zsel, msel, psel = np.loadtxt('Data/mcgreer13_dr7selfunc.dat', usecols=(1,2,3), unpack=True)
    vol = volume(zsel, area)*dz
    
    # def volm(m, msel, psel, vsel):

    #     total_vol = 0.0
    #     bincount = 0
    #     bincount_pnonzero = 0

    #     dm = abs(msel[1]-msel[0])

    #     for i in xrange(msel.size):
    #         if (m >= msel[i]-dm) and (m < msel[i]+dm):
    #             bincount += 1 
    #             if psel[i] > 0.0:
    #                 bincount_pnonzero += 1 
    #                 # total_vol += vsel[i]*psel[i]
    #                 total_vol += vsel[i] # Should this be vsel[i]*dm?

    #     return total_vol

    def volm(m, msel, psel, vsel):

        total_vol = 0.0

        bins = np.array([-28.23, -27.73, -27.23, -26.73, -26.23, -25.73])
        idx = np.searchsorted(bins, m, side='right')
        mlow = bins[idx-1]
        mhigh = bins[idx]

        dm = abs(msel[1]-msel[0])
        n = int(abs(mhigh-mlow)/dm)+1

        for i in xrange(msel.size):
            if (mlow >= msel[i]-dm) and (mlow < msel[i]+dm):
                for j in range(n):
                    try: 
                        total_vol += vsel[i+j]*psel[i+j]*dm
                    except(IndexError):
                        total_vol += 0.0 

        return total_vol

    v1 = np.array([volm(x, msel, psel, vol) for x in m])

    bins = [-28.23, -27.73, -27.23, -26.73, -26.23, -25.73]
    # h = np.histogram(m,bins=bins,weights=1.0/(p*v1))
    h = np.histogram(m,bins=bins,weights=1.0/v1)
    print m[v1==0.0]
    print p[v1==0.0]
    # print volm(-26.73, msel, psel, vol, pd=True)
    print '----'
    
    # # Stripe 82 
    # area = 235.0 # deg^2
    # dz = 0.05
    # dm = 0.1
    # zsel, msel, psel = np.loadtxt('Data/mcgreer13_s82selfunc.dat', usecols=(1,2,3), unpack=True)
    # vol = volume(zsel, area)*dz

    # mask = np.ones(len(M1450), dtype=bool)
    # mask[dr7] = False 
    # v2 = np.array([volm(x, msel, psel, vol) for x in m[mask]])

    # v = np.where(mask, v2, v1)
    
    # bins = [-26.0, -25.0, -24.0, -23.0, -22.0, -21]
    # h = np.histogram(m[1:],bins=bins,weights=1.0/(p[1:]*v[1:]))
    
    nums = h[0]
    mags = (h[1][:-1] + h[1][1:])*0.5
    dmags = np.diff(h[1])*0.5

    left = mags - h[1][:-1]
    right = h[1][1:] - mags
    
    phi = nums/np.diff(h[1])
    logphi = np.log10(phi) # cMpc^-3 mag^-1

    # Calculate errorbars on our binned LF.  These have been estimated
    # using Equations 1 and 2 of Gehrels 1986 (ApJ 303 336), as
    # implemented in astropy.stats.poisson_conf_interval.  The
    # interval='frequentist-confidence' option to that astropy function is
    # exactly equal to the Gehrels formulas, although the documentation
    # does not say so.
    n = np.histogram(m,bins=bins)[0]

    print n 

    nlims = pci(n,interval='frequentist-confidence')
    nlims *= phi/n 
    uperr = np.log10(nlims[1]) - logphi 
    downerr = logphi - np.log10(nlims[0])

    print mags
    print logphi 

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
    
def draw(lf, z_plot, composite=None, dirname=''):
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

    mags, left, right, logphi, uperr, downerr = get_lf(lf, z_plot, [16,])

    ax.scatter(mags, logphi, c='r', edgecolor='None', zorder=4, label='my binning')
    ax.errorbar(mags, logphi, ecolor='r', capsize=0,
                xerr=np.vstack((left, right)),
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=4)

    f = 'Data/allqlfs.dat'
    (sid, M1450, M1450_lerr, M1450_uerr, phi, phi_uerr, phi_lerr) = np.loadtxt(f, usecols=(1,6,7,8,9,10,11), unpack=True)
    M1450 = M1450[sid==8] # Select McGreer's sample (number 8) 
    M1450_lerr = M1450_lerr[sid==8]
    M1450_uerr = M1450_uerr[sid==8]
    phi = phi[sid==8]
    phi_uerr = phi_uerr[sid==8]
    phi_lerr = phi_lerr[sid==8]

    print phi
    print M1450

    dr7 = [0, 1, 2, 4, 6]

    ax.scatter(M1450[dr7], phi[dr7], c='b', label='McGreer et al.\ 2013 SDSS DR7', edgecolor='None', zorder=303) 
    ax.errorbar(M1450[dr7], phi[dr7], ecolor='b', capsize=0,
                xerr=np.vstack((M1450_lerr[dr7], M1450_uerr[dr7])),
                yerr=np.vstack((phi_lerr[dr7], phi_uerr[dr7])), fmt=None,zorder=304)

    # mask = np.ones(len(M1450), dtype=bool)
    # mask[dr7] = False 
    # ax.scatter(M1450[mask], phi[mask], c='darkgreen', label='McGreer et al.\ 2013 Stripe 82', edgecolor='None', zorder=303) 
    # ax.errorbar(M1450[mask], phi[mask], ecolor='darkgreen', capsize=0,
    #             xerr=np.vstack((M1450_lerr[mask], M1450_uerr[mask])),
    #             yerr=np.vstack((phi_lerr[mask], phi_uerr[mask])), fmt=None,zorder=304)
    
    ax.set_xlim(-29.0, -21.0)
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
