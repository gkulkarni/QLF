import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '16'
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

def binvol(m, zrange, bins, msel, psel, vsel, zsel, totv):

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

    z, m, p = np.loadtxt('Data/r13miz2_sample.dat', usecols=(1, 2, 3), unpack=True)
    select = ((z>=zrange[0]) & (z<zrange[1]))
    m = m[select]
    p = p[select]

    area = 2236.0 # deg^2
    dz = 0.05 
    dm = 0.1
    zsel, msel, psel = np.loadtxt('Data/r13miz2_selfunc.dat', usecols=(1,2,3), unpack=True)
    vol = volume(zsel, area)*dz

    psel[(zsel < zrange[0]) | (zsel >= zrange[1])] = 0.0

    zv = np.linspace(zrange[0], zrange[1], 50)
    dzv = np.diff(zv)[0]
    v = volume(zv, area)*dzv
    totvol = np.sum(v)

    v1 = np.array([binvol(x, zrange, bins, msel, psel, vol, zsel, totvol) for x in m])

    v1_nonzero = v1[np.where(v1>0.0)]
    m = m[np.where(v1>0.0)]

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


def ross(ax, zrange, yticklabels=False, xticklabels=False, nofirstylabel=True, nolastxlabel=True, plotmybins=False, bins=None, legend=False):

    ax.tick_params('both', which='major', length=4, width=1)
    
    datafile = 'Data/ross13_qlf.dat'
    with open(datafile, 'r') as f: 
        z, m, phi, phi_err = np.loadtxt(f, usecols=(0,2,4,5), unpack=True)

    sel = ((z >= zrange[0]) & (z < zrange[1]))
    
    m = m[sel]
    dm = 0.15*np.ones(m.size)

    phi = phi[sel]

    phi_err = (phi_err[sel])*1.0e-9
    phiu = 10.0**phi + phi_err
    phil = 10.0**phi - phi_err
    phi_uerr = np.log10(phiu)-phi
    phi_lerr = phi-np.log10(phil)
    
    ax.scatter(m, phi, c='#ffffff', s=30, label='Ross et al.\ 2013',
               edgecolor='r', zorder=304)
    ax.errorbar(m, phi, ecolor='r', capsize=0, 
                yerr=np.vstack((phi_lerr, phi_uerr)), fmt='None', zorder=303)

    if plotmybins:
        mags, left, right, logphi, uperr, downerr = get_lf(zrange, bins)
        
        ax.scatter(mags, logphi, c='g', edgecolor='None', zorder=1, label='my binning', s=35)
        ax.errorbar(mags, logphi, ecolor='g', capsize=0,
                    yerr=np.vstack((uperr, downerr)),
                    fmt='None',zorder=1)

    ax.set_xlim(-23, -31)
    ax.set_ylim(-9, -5)
    ax.set_xticks(np.arange(-31,-22, 2))

    if not yticklabels:
        ax.set_yticklabels('')

    if not xticklabels:
        ax.set_xticklabels('')

    if nofirstylabel:
        ax.get_yticklabels()[0].set_visible(False)

    if nolastxlabel:
        ax.get_xticklabels()[0].set_visible(False)

    plt.text(0.04, 0.03, r'${:g}\leq z<{:g}$'.format(zrange[0], zrange[1]),
             transform=ax.transAxes, fontsize=12)

    if legend:
        plt.legend(loc='upper right', fontsize=10, handlelength=3, frameon=False, framealpha=0.0,
                   labelspacing=.1, handletextpad=0.4, borderpad=0.2, scatterpoints=1)

        
    return 

nplots_x = 3
nplots_y = 3
nplots = nplots_x * nplots_y

plot_number = 0

nx = nplots_x
ny = nplots_y 
factor_x = 2.5
factor_y = 3.3
ldim = 0.4*factor_x
bdim = 0.25*factor_y
rdim = 0.1*factor_x
tdim = 0.15*factor_y
wspace = 0.
hspace = 0. 

plotdim_x = factor_x*nx + (nx-1)*wspace
plotdim_y = factor_y*ny + (ny-1)*hspace

hdim = plotdim_x + ldim + rdim 
vdim = plotdim_y + tdim + bdim 

fig = plt.figure(figsize=(hdim, vdim), dpi=100)

l = ldim/hdim
b = bdim/vdim
r = (ldim + plotdim_x)/hdim
t = (bdim + plotdim_y)/vdim 
fig.subplots_adjust(left=l, bottom=b, right=r, top=t, wspace=wspace/hdim,
                    hspace=hspace/vdim)

zs = {1:(2.2,2.3), 2:(2.3,2.4), 3:(2.4,2.5), 4:(2.5,2.6), 5:(2.6,2.7),
      6:(2.7,2.8), 7:(2.8,3.0), 8:(3,3.25), 9:(3.25,3.5)}

bins = {7:np.arange(-29.1, -24.1, 0.3), 8:np.arange(-29.1, -23.6, 0.3)}

for i in range(nplots):

    ax = fig.add_subplot(nplots_y, nplots_x, i+1)
    
    if i in set([0,3]):
        ross(ax, zs[i+1], yticklabels=True, plotmybins=True, bins=bins[8])
    elif i==7:
        ross(ax, zs[i+1], xticklabels=True, plotmybins=True, bins=bins[7])
    elif i==8:
        ross(ax, zs[i+1], xticklabels=True, nolastxlabel=False, plotmybins=True, bins=bins[8])
    elif i == 6:
        ross(ax, zs[i+1], yticklabels=True, xticklabels=True, nofirstylabel=False, plotmybins=True, bins=bins[8])
    elif i==2:
        ross(ax, zs[i+1], plotmybins=True, bins=bins[8], legend=True)
    else:
        ross(ax, zs[i+1], plotmybins=True, bins=bins[8])

fig.text(0.5, 0.02, r'$M_i [z=2]$', transform=fig.transFigure,
         horizontalalignment='center', verticalalignment='center')

fig.text(0.03, 0.5, r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$',
         transform=fig.transFigure, horizontalalignment='center',
         verticalalignment='center', rotation='vertical')

fig.text(0.5, 0.97, r'BOSS DR9 colour-selected',
         transform=fig.transFigure, horizontalalignment='center')

plt.savefig('boss.pdf')



