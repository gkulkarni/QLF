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

    z, m, p = np.loadtxt('Data/richards06_sample.dat', usecols=(0, 2, 3), unpack=True)
    select = ((z>=zrange[0]) & (z<zrange[1]))
    m = m[select]
    p = p[select]

    area = 1622.0 # deg^2
    dz = 0.05 
    dm = 0.1
    zsel, msel, psel = np.loadtxt('Data/r06miz2_selfunc.dat', usecols=(1, 2, 3), unpack=True)
    vol = volume(zsel, area)*dz

    psel[(zsel < zrange[0]) | (zsel >= zrange[1])] = 0.0

    v1 = np.array([binvol(x, zrange, bins, msel, psel, vol, zsel) for x in m])

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


def richards(ax, zrange, yticklabels=False, xticklabels=False, nofirstylabel=True,
             nolastxlabel=True, nofirstxlabel=False, plotmybins=False, bins=None, legend=False):

    ax.tick_params('both', which='major', length=4, width=1)
    
    datafile = 'Data/richards06_qlf.dat'
    with open(datafile, 'r') as f: 
        z, m, phi, phi_err = np.loadtxt(f, usecols=(0,1,2,3), unpack=True)

    sel = ((z >= zrange[0]) & (z < zrange[1]))
    
    m = m[sel]
    dm = 0.15*np.ones(m.size)

    phi = phi[sel]

    phi_err = (phi_err[sel])*1.0e-9
    phiu = 10.0**phi + phi_err
    phil = 10.0**phi - phi_err
    phil = np.where(phil < 0.0, 1.0e-15, phil)
    
    phi_uerr = np.log10(phiu)-phi
    phi_lerr = phi-np.log10(phil)
    
    ax.scatter(m, phi, c='#ffffff', s=30, label='Richards et al.\ 2006',
               edgecolor='r', zorder=304)
    ax.errorbar(m, phi, ecolor='r', capsize=0, 
                yerr=np.vstack((phi_lerr, phi_uerr)), fmt='None', zorder=303)

    if plotmybins:
        mags, left, right, logphi, uperr, downerr = get_lf(zrange, bins)
        
        ax.scatter(mags, logphi, c='g', edgecolor='None', zorder=1, label='Our binning', s=35)
        ax.errorbar(mags, logphi, ecolor='g', capsize=0,
                    yerr=np.vstack((uperr, downerr)),
                    fmt='None',zorder=1)
    
    ax.set_xlim(-22, -31)
    ax.set_ylim(-10, -5)
    ax.set_xticks(np.arange(-31,-20, 2))
    ax.set_yticks(np.arange(-10, -4))

    if not yticklabels:
        ax.set_yticklabels('')

    if not xticklabels:
        ax.set_xticklabels('')

    if nofirstylabel:
        ax.get_yticklabels()[0].set_visible(False)

    if nolastxlabel:
        ax.get_xticklabels()[0].set_visible(False)

    if nofirstxlabel:
        ax.get_xticklabels()[5].set_visible(False)

    plt.text(0.04, 0.05, r'${:g}\leq z<{:g}$'.format(zrange[0], zrange[1]),
             transform=ax.transAxes, fontsize=12)

    if legend:
        plt.legend(loc='upper left', fontsize=10, handlelength=3, frameon=False, framealpha=0.0,
                   labelspacing=.1, handletextpad=-0.4, borderpad=0.2, scatterpoints=1)

    return

nplots_x = 3
nplots_y = 4
nplots = nplots_x * nplots_y

plot_number = 0

nx = nplots_x
ny = nplots_y 
factor_x = 2.5
factor_y = 2.5
ldim = 0.4*factor_x
bdim = 0.25*factor_y
rdim = 0.1*factor_x
tdim = 0.18*factor_y
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

zs = {1:(0.3,0.68), 2:(0.68,1.06), 3:(1.06,1.44), 4:(1.44,1.82), 5:(1.82,2.2),
      6:(2.2,2.6), 7:(2.6,3), 8:(3,3.5), 9:(3.5,4), 10:(4,4.5), 11:(4.5,5)}

bins = np.arange(-30.9, -22.3, 0.3)  

for i in range(nplots-1):

    ax = fig.add_subplot(nplots_y, nplots_x, i+1)

    if i in set([0,3,6]):
        richards(ax, zs[i+1], yticklabels=True, plotmybins=True, bins=bins)
    elif i==8:
        richards(ax, zs[i+1], xticklabels=True, nolastxlabel=False, nofirstxlabel=True, plotmybins=True, bins=bins)
    elif i==9:
        richards(ax, zs[i+1], yticklabels=True, xticklabels=True, nofirstylabel=False, plotmybins=True, bins=bins, legend=True)
    elif i==10:
        richards(ax, zs[i+1], xticklabels=True, nolastxlabel=False, plotmybins=True, bins=bins)
    else:
        richards(ax, zs[i+1], plotmybins=True, bins=bins)
        
fig.text(0.5, 0.01, r'$M_i [z=2]$', transform=fig.transFigure,
         horizontalalignment='center', verticalalignment='center')

fig.text(0.05, 0.5, r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$',
         transform=fig.transFigure, horizontalalignment='center',
         verticalalignment='center', rotation='vertical')

fig.text(0.5, 0.97, r'SDSS DR3',
         transform=fig.transFigure, horizontalalignment='center')

plt.savefig('sdss.pdf')



