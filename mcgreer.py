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
         'h':0.7}

def volume(z, area, cosmo=cosmo):

    omega = (area/41253.0)*4.0*np.pi # str
    volperstr = cd.diff_comoving_volume(z,**cosmo) # cMpc^3 str^-1 dz^-1

    return omega*volperstr # cMpc^3 dz^-1

def binvol(m, zrange, bins, msel, psel, vsel, zsel):

    """
    
    Calculate volume in the i'th bin.

    """
    
    total_vol = 0.0

    idx = -1 
    for i in range(len(bins)):
        if (m > bins[i]) and (m < bins[i+1]):
            idx = i 
    
    idx = np.searchsorted(bins, m)
    
    mlow = bins[idx-1]
    mhigh = bins[idx]

    dm = 0.1
    n = int(abs(mhigh-mlow)/dm)

    for i in xrange(msel.size):
        if (msel[i] >= mlow) and (msel[i] < mhigh):
            if (zsel[i] >= zrange[0]) and (zsel[i] < zrange[1]):
                total_vol += vsel[i]*psel[i]*dm

    return total_vol


def get_lf(zrange, bins):

    z, m, p = np.loadtxt('Data/mcgreer13_dr7sample2.dat', usecols=(1, 2, 3), unpack=True)
    select = ((z>=zrange[0]) & (z<zrange[1]))
    m = m[select]
    p = p[select]

    area = 6222.0 # deg^2
    dz = 0.05 
    dm = 0.1
    zsel, msel, psel = np.loadtxt('Data/mcgreer13_dr7selfunc2.dat', usecols=(1, 2, 3), unpack=True)
    vol = volume(zsel, area)*dz

    psel[(zsel < zrange[0]) | (zsel >= zrange[1])] = 0.0


    v1 = np.array([binvol(x, zrange, bins, msel, psel, vol, zsel) for x in m])
    print v1.size

    v1_nonzero = v1[np.where(v1>0.0)]
    m = m[np.where(v1>0.0)]
    print v1_nonzero.size
    
    h = np.histogram(m,bins=bins,weights=1.0/(v1_nonzero))

    nums = h[0]
    mags = (h[1][:-1] + h[1][1:])*0.5
    dmags = np.diff(h[1])*0.5

    left = mags - h[1][:-1]
    right = h[1][1:] - mags
    
    phi = nums
    logphi = np.log10(phi) # cMpc^-3 mag^-1

    n = np.histogram(m,bins=bins)[0]
    print n 

    nlims = pci(n,interval='frequentist-confidence')
    nlims *= phi/n 
    uperr = np.log10(nlims[1]) - logphi 
    downerr = logphi - np.log10(nlims[0])

    return mags, left, right, logphi, uperr, downerr

def get_lf_s82(zrange, bins):

    z, m, p = np.loadtxt('Data/mcgreer13_s82sample2.dat', usecols=(1, 2, 3), unpack=True)
    select = ((z>=zrange[0]) & (z<zrange[1]))
    m = m[select]
    p = p[select]

    area = 235.0 # deg^2
    dz = 0.05 
    dm = 0.1
    zsel, msel, psel = np.loadtxt('Data/mcgreer13_s82selfunc2.dat', usecols=(1, 2, 3), unpack=True)
    vol = volume(zsel, area)*dz

    psel[(zsel < zrange[0]) | (zsel >= zrange[1])] = 0.0

    v1 = np.array([binvol(x, zrange, bins, msel, psel, vol, zsel) for x in m])

    print v1.size

    v1_nonzero = v1[np.where(v1>0.0)]
    m = m[np.where(v1>0.0)]

    print v1_nonzero.size
    
    h = np.histogram(m,bins=bins,weights=1.0/(v1_nonzero))

    nums = h[0]
    mags = (h[1][:-1] + h[1][1:])*0.5
    dmags = np.diff(h[1])*0.5

    left = mags - h[1][:-1]
    right = h[1][1:] - mags
    
    phi = nums
    logphi = np.log10(phi) # cMpc^-3 mag^-1

    n = np.histogram(m,bins=bins)[0]

    nlims = pci(n,interval='frequentist-confidence')
    nlims *= phi/n 
    uperr = np.log10(nlims[1]) - logphi 
    downerr = logphi - np.log10(nlims[0])

    return mags, left, right, logphi, uperr, downerr


fig = plt.figure(figsize=(6, 6), dpi=100)
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel(r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$')
ax.set_xlabel('$M_{1450}$')

ax.tick_params('both', which='major', length=7, width=1)
ax.tick_params('both', which='minor', length=3, width=1)
ax.tick_params('x', which='major', pad=6)

# Plot binned LF from literature
f = 'Data/mcgreer_dr7_published.dat'
(sid, M1450, M1450_lerr, M1450_uerr,
 phi, phi_err) = np.loadtxt(f, unpack=True)
d = phi_err * 1.0e-9
x = 10.0**phi
phi_lerr = np.log10(x)-np.log10(x-d)
phi_uerr = np.log10(x+d)-np.log10(x)
phi_err = np.log10(phi_err * 1.0e-9)

ax.errorbar(M1450, phi, ecolor='b', capsize=0,
            xerr=np.vstack((M1450_lerr, M1450_uerr)),
            yerr=np.vstack((phi_lerr, phi_uerr)), fmt='none', zorder=303)
ax.scatter(M1450, phi, c='#ffffff', label='McGreer et al.\ 2013 SDSS DR7',
           edgecolor='b', zorder=304, s=32)

# Plot binned LF from literature
f = 'Data/mcgreer_s82_published.dat'
(sid, M1450, M1450_lerr, M1450_uerr,
 phi, phi_err) = np.loadtxt(f, unpack=True)
d = phi_err * 1.0e-9
x = 10.0**phi
phi_lerr = np.log10(x)-np.log10(x-d)
phi_uerr = np.log10(x+d)-np.log10(x)
phi_err = np.log10(phi_err * 1.0e-9)

ax.errorbar(M1450, phi, ecolor='r', capsize=0,
            xerr=np.vstack((M1450_lerr, M1450_uerr)),
            yerr=np.vstack((phi_lerr, phi_uerr)), fmt='none', zorder=305)
ax.scatter(M1450, phi, c='#ffffff', label='McGreer et al.\ 2013 SDSS Stripe 82',
           edgecolor='r', zorder=306, s=32)

zrange = (4.7, 5.1)
bins = np.arange(-29.3, -24.0, 0.5)
mags, left, right, logphi, uperr, downerr = get_lf(zrange, bins)

ax.scatter(mags, logphi, c='g', edgecolor='None', zorder=306, label='Our binning (SDSS DR7)', s=35)
ax.errorbar(mags, logphi, ecolor='g', capsize=0,
            xerr=np.vstack((left, right)),
            yerr=np.vstack((uperr, downerr)),
            fmt='None',zorder=306)

zrange = (4.7, 5.1)
bins = np.arange(-27.275, -23., 0.55)
mags, left, right, logphi, uperr, downerr = get_lf_s82(zrange, bins)

ax.scatter(mags, logphi, c='maroon', edgecolor='None', zorder=306, label='Our binning (SDSS Stripe 82)', s=35)
ax.errorbar(mags, logphi, ecolor='maroon', capsize=0,
            xerr=np.vstack((left, right)),
            yerr=np.vstack((uperr, downerr)),
            fmt='None',zorder=306)

ax.set_xlim(-29.0, -23.0)
ax.set_ylim(-11.0, -6.0) 

plt.legend(loc='upper left', fontsize=10, handlelength=3, frameon=False, framealpha=0.0,
           labelspacing=.1, handletextpad=-0.4, borderpad=0.2, scatterpoints=1)

plt.savefig('mcgreer.pdf', bbox_inches='tight')

