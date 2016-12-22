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

    dm = 0.05
    n = int(abs(mhigh-mlow)/dm)

    for i in xrange(msel.size):
        if (msel[i] >= mlow) and (msel[i] < mhigh):
            if (zsel[i] >= zrange[0]) and (zsel[i] < zrange[1]):
                total_vol += vsel[i]*psel[i]*dm

    return total_vol


def get_lf(zrange, bins, old=True):

    if old:
        z, m, p = np.loadtxt('Data/glikman11qso.dat', usecols=(1, 2, 3), unpack=True)
    else:
        z, m, p = np.loadtxt('Data/glikman11debug.dat', usecols=(1, 2, 3), unpack=True)
    select = ((z>=zrange[0]) & (z<zrange[1]))
    m = m[select]
    p = p[select]

    area = 1.71 # deg^2
    dz = 0.02
    dm = 0.05
    if old: 
        zsel, msel, psel = np.loadtxt('Data/glikman11_selfunc_ndwfs_old.dat', usecols=(1,2,3), unpack=True)
    else:
        zsel, msel, psel = np.loadtxt('Data/glikman11_selfunc_ndwfs.dat', usecols=(1,2,3), unpack=True)
    vol = volume(zsel, area)*dz

    psel[(zsel < zrange[0]) | (zsel >= zrange[1])] = 0.0

    # m[:12] because only those qsos are from NDWFS
    v1 = np.array([binvol(x, zrange, bins, msel, psel, vol, zsel) for x in m[:12]])
    
    area = 2.05 # deg^2
    if old:
        zsel, msel, psel = np.loadtxt('Data/glikman11_selfunc_dls_old.dat', usecols=(1, 2, 3), unpack=True)
    else:
        zsel, msel, psel = np.loadtxt('Data/glikman11_selfunc_dls.dat', usecols=(1, 2, 3), unpack=True)
    vol = volume(zsel, area)*dz

    psel[(zsel < zrange[0]) | (zsel >= zrange[1])] = 0.0

    # m[12:] because only those qsos are from DLS
    v2 = np.array([binvol(x, zrange, bins, msel, psel, vol, zsel) for x in m[12:]])

    v = np.concatenate((v1,v2))
    print v.size
    v_nonzero = v[np.where(v>0.0)]
    print v_nonzero.size
    m = m[np.where(v>0.0)]
    
    h = np.histogram(m,bins=bins,weights=1.0/(v_nonzero))

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

    mags, b, c = bs(m, m, bins=bins)
    left = mags - b[:-1]
    right = b[1:] - mags
    
    return mags, left, right, logphi, uperr, downerr


fig = plt.figure(figsize=(6, 6), dpi=100)
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel(r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$')
ax.set_xlabel('$M_{1450}$')

ax.tick_params('both', which='major', length=7, width=1)
ax.tick_params('both', which='minor', length=3, width=1)
ax.tick_params('x', which='major', pad=6)

# Plot binned LF from literature
f = 'Data/glikman11.dat'
(M1450_glikman, M1450_glikman_lerr, M1450_glikman_uerr, phi_glikman,
 phi_glikman_uerr, phi_glikman_lerr) = np.loadtxt(f, unpack=True)

phi_glikman *= 1.0e-8
phi_glikman_lerr *= 1.0e-8
phi_glikman_uerr *= 1.0e-8

phi_glikman_llogerr = np.log10(phi_glikman)-np.log10(phi_glikman-phi_glikman_lerr)
phi_glikman_ulogerr = np.log10(phi_glikman+phi_glikman_uerr)-np.log10(phi_glikman) 
phi_glikman = np.log10(phi_glikman) 

ax.errorbar(M1450_glikman[4:], phi_glikman[4:], ecolor='r', capsize=0,
            xerr=np.vstack((M1450_glikman_lerr[4:], M1450_glikman_uerr[4:])),
            yerr=np.vstack((phi_glikman_llogerr[4:], phi_glikman_ulogerr[4:])), fmt='none', zorder=305)
ax.scatter(M1450_glikman[4:], phi_glikman[4:], c='#ffffff', label='Glikman et al.\ 2011 NDWFS+DLS', edgecolor='r', zorder=306, s=32)

zrange = (3.5, 5.2)
bins = np.array([-26.0, -25.0, -24.0, -23.0, -22.0, -21])
mags, left, right, logphi, uperr, downerr = get_lf(zrange, bins)

ax.scatter(mags, logphi, c='g', edgecolor='None', zorder=306, label='my binning', s=35)
ax.errorbar(mags, logphi, ecolor='g', capsize=0,
            xerr=np.vstack((left, right)),
            yerr=np.vstack((uperr, downerr)),
            fmt='None',zorder=306)

mags, left, right, logphi, uperr, downerr = get_lf(zrange, bins, old=False)

ax.scatter(mags, logphi, c='maroon', edgecolor='None', zorder=306, label='my binning (with Gabor\'s modified data of 18 October 2016)', s=35)
ax.errorbar(mags, logphi, ecolor='maroon', capsize=0,
            xerr=np.vstack((left, right)),
            yerr=np.vstack((uperr, downerr)),
            fmt='None',zorder=306)

ax.set_xlim(-28.0, -20.0)
ax.set_ylim(-10.0, -4.0)
plt.xticks(np.arange(-28,-19,2))

plt.legend(loc='lower right', fontsize=10, handlelength=3, frameon=False, framealpha=0.0,
           labelspacing=.1, handletextpad=-0.4, borderpad=0.6, scatterpoints=1)

plt.savefig('glikman.pdf', bbox_inches='tight')

