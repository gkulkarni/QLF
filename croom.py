"""
Compare 2SLAQ binned LF to Croom et al. 2009.

"""
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
from drawlf import render
from individual import lf
import drawlf
mpl.rcParams['font.size'] = '16'

case = 'M1450_worseck'
# case = 'Mgz2_unprocessed'
# case = 'Mgz2_zsuccess'
# case = 'Mgz2_zsuccess_coverage'
# case = 'Mgz2_coverage'

if case == 'Mgz2_coverage':

    qlumfiles = ['Data_new/dr7z2p2_sample.dat',
                 'croom_mgz2_photometric_coverage/croom09sgp_sample_Mgz2.dat',
                 'croom_mgz2_photometric_coverage/croom09ngp_sample_Mgz2.dat']
    
    selnfiles = [('Data_new/dr7z2p2_selfunc.dat',
                  0.1, 0.05, 6248.0, 13,
                  r'SDSS DR7 Richards et al.\ 2006'),

                 ('croom_mgz2_photometric_coverage/croom09_selfunc_photometric_hostcorrected_coverage_sgp.dat',
                  0.3, 0.05, 64.2, 15,
                  r'2SLAQ Croom et al.\ 2009'),

                 ('croom_mgz2_photometric_coverage/croom09_selfunc_photometric_hostcorrected_coverage_ngp.dat',
                  0.3, 0.05, 127.7, 15,
                  r'2SLAQ Croom et al.\ 2009')]
    
elif case == 'Mgz2_zsuccess_coverage':

    qlumfiles = ['Data_new/dr7z2p2_sample.dat',
                 'croom_mgz2_photometric_zsuccess_coverage/croom09sgp_sample_Mgz2.dat',
                 'croom_mgz2_photometric_zsuccess_coverage/croom09ngp_sample_Mgz2.dat']
    
    selnfiles = [('Data_new/dr7z2p2_selfunc.dat',
                  0.1, 0.05, 6248.0, 13,
                  r'SDSS DR7 Richards et al.\ 2006'),

                 ('croom_mgz2_photometric_zsuccess_coverage/croom09_selfunc_photometric_hostcorrected_zsuccess_coverage_sgp.dat',
                  0.3, 0.05, 64.2, 15,
                  r'2SLAQ Croom et al.\ 2009'),

                 ('croom_mgz2_photometric_zsuccess_coverage/croom09_selfunc_photometric_hostcorrected_zsuccess_coverage_ngp.dat',
                  0.3, 0.05, 127.7, 15,
                  r'2SLAQ Croom et al.\ 2009')]

elif case == 'Mgz2_zsuccess':

    qlumfiles = ['Data_new/dr7z2p2_sample.dat',
                 'croom_mgz2_photometric_zsuccess/croom09sgp_sample_Mgz2.dat',
                 'croom_mgz2_photometric_zsuccess/croom09ngp_sample_Mgz2.dat']
    
    selnfiles = [('Data_new/dr7z2p2_selfunc.dat',
                  0.1, 0.05, 6248.0, 13,
                  r'SDSS DR7 Richards et al.\ 2006'),

                 ('croom_mgz2_photometric_zsuccess/croom09_selfunc_photometric_hostcorrected_zsuccess.dat',
                  0.3, 0.05, 64.2, 15,
                  r'2SLAQ Croom et al.\ 2009'),

                 ('croom_mgz2_photometric_zsuccess/croom09_selfunc_photometric_hostcorrected_zsuccess.dat',
                  0.3, 0.05, 127.7, 15,
                  r'2SLAQ Croom et al.\ 2009')]

elif case == 'Mgz2_unprocessed':

    qlumfiles = ['Data_new/dr7z2p2_sample.dat',
                 'croom_mgz2_photometric/croom09sgp_sample_Mgz2.dat',
                 'croom_mgz2_photometric/croom09ngp_sample_Mgz2.dat']

    selnfiles = [('Data_new/dr7z2p2_selfunc.dat',
                  0.1, 0.05, 6248.0, 13,
                  r'SDSS DR7 Richards et al.\ 2006'),

                 ('croom_mgz2_photometric/croom09_selfunc_photometric_hostcorrected.dat',
                  0.3, 0.05, 64.2, 15,
                  r'2SLAQ Croom et al.\ 2009'),

                 ('croom_mgz2_photometric/croom09_selfunc_photometric_hostcorrected.dat',
                  0.3, 0.05, 127.7, 15,
                  r'2SLAQ Croom et al.\ 2009')]
    
else:
    qlumfiles = ['Data_new/croom09sgp_sample_test.dat',
                 'Data_new/croom09ngp_sample_test.dat']

    selnfiles = [('croom09sgp_selfunc_withdmdz.dat',
                  0.3, 0.05, 64.2, 15,
                  r'2SLAQ Croom et al.\ 2009'),

                 ('croom09ngp_selfunc_withdmdz.dat',
                  0.3, 0.05, 127.7, 15,
                  r'2SLAQ Croom et al.\ 2009')]
    

def our_lf(zrange):

    lfi = lf(quasar_files=qlumfiles, selection_maps=selnfiles, zlims=zrange)

    return lfi


def log10phi(theta, mag):

    log10phi_star, M_star, alpha, beta = theta 

    phi = 10.0**log10phi_star / (10.0**(0.4*(alpha+1)*(mag-M_star)) +
                                 10.0**(0.4*(beta+1)*(mag-M_star)))
    return np.log10(phi)


def croom(i, ax, zrange, yticklabels=False, xticklabels=False, nofirstylabel=True,
          nolastxlabel=True, plotmybins=True, bins=None, legend=False):

    ax.tick_params('both', which='major', length=4, width=1)

    # Plot Croom's public LF 
    datafile = 'Data/croom09_qlf.dat'
    with open(datafile, 'r') as f: 
        m, z, phi, phi_err_low, phi_err_up = np.loadtxt(f, usecols=(0,1,3,4,5), unpack=True)
    sel = ((z >= zrange[0]) & (z < zrange[1]))
    m = m[sel]
    phi = phi[sel]
    phi_lerr = -phi_err_low[sel] # Values in table have minus sign
    phi_uerr = phi_err_up[sel]
    dm = 0.15*np.ones(m.size)
    ax.scatter(m, phi, c='#ffffff', s=30, label='Croom et al.\ 2009',
               edgecolor='r', zorder=2)
    ax.errorbar(m, phi, ecolor='r', capsize=0, 
                yerr=np.vstack((phi_lerr, phi_uerr)), fmt='None', zorder=1)

    if plotmybins:
        lfi = our_lf(zrange)

        print np.unique(lfi.sid)
        print 'nqso =', lfi.sid[lfi.sid==15].size
        print 'nqso all =', lfi.sid_all[lfi.sid_all==15].size
        print 'faintest qso: ', np.max(lfi.M1450_all[lfi.sid_all==15])

        # Plot 2SLAQ LF 
        sid_croom = 15
        z_plot = (zrange[0]+zrange[1])/2
        if case == 'M1450_worseck':
            lfbins = drawlf.get_lf(lfi, sid_croom, z_plot, special='croom_comparison')
        else:
            lfbins = drawlf.get_lf(lfi, sid_croom, z_plot, special='croom_comparison_Mgz2')
        mags, left, right, logphi, uperr, downerr = lfbins
        if case == 'M1450_worseck':
            # Convert from M1450 to Mg(z=2) by using Equation B8 of Ross
            # et al. 2013.
            mags = mags - 1.23
        ax.scatter(mags, logphi, c='g', edgecolor='None',
                   zorder=3, label='Our binning (2SLAQ)', s=35)
        ax.errorbar(mags, logphi, ecolor='g', capsize=0,
                    yerr=np.vstack((uperr, downerr)),
                    fmt='None',zorder=2)

        # # Plot SDSS DR7 LF
        # sid_sdss = 13
        # lfbins = drawlf.get_lf(lfi, sid_sdss, z_plot, special='croom_comparison')
        # mags, left, right, logphi, uperr, downerr = lfbins
        # # Convert from M1450 to Mg(z=2) by using Equation B8 of Ross
        # # et al. 2013.
        # mags = mags - 1.23
        # ax.scatter(mags, logphi, c='b', edgecolor='None',
        #            zorder=1, label='Our binning (SDSS)', s=35)
        # ax.errorbar(mags, logphi, ecolor='b', capsize=0,
        #             yerr=np.vstack((uperr, downerr)),
        #             fmt='None',zorder=1)

    ax.set_xlim(-19, -31)
    ax.set_ylim(-11, -4)
    ax.set_xticks(np.arange(-31,-17, 4))
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
        plt.legend(loc='upper right', fontsize=10, handlelength=3,
                   frameon=False, framealpha=0.0, labelspacing=.1,
                   handletextpad=0., borderpad=0.2, scatterpoints=1)

    return 

nplots_x = 3
nplots_y = 2
nplots = nplots_x * nplots_y
nx = nplots_x
ny = nplots_y 
factor_x = 2.5
factor_y = 3.3
ldim = 0.35*factor_x
bdim = 0.2*factor_y
rdim = 0.1*factor_x
tdim = 0.1*factor_y
wspace = 0.0
hspace = 0.0 
plotdim_x = factor_x*nx + (nx-1)*wspace
plotdim_y = factor_y*ny + (ny-1)*hspace
hdim = plotdim_x + ldim + rdim 
vdim = plotdim_y + tdim + bdim 

fig = plt.figure(figsize=(hdim, vdim), dpi=100)
l = ldim/hdim
b = bdim/vdim
r = (ldim + plotdim_x)/hdim
t = (bdim + plotdim_y)/vdim 
fig.subplots_adjust(left=l, bottom=b, right=r, top=t,
                    wspace=wspace/hdim,
                    hspace=hspace/vdim)

zs = {1:(0.4, 0.68), 2:(0.68, 1.06), 3:(1.06, 1.44),
      4:(1.44, 1.82), 5:(1.82,2.2), 6:(2.2,2.6)}

ax = fig.add_subplot(nplots_y, nplots_x, 1)
croom(1, ax, zs[1], yticklabels=True, legend=True)

ax = fig.add_subplot(nplots_y, nplots_x, 2)
croom(2, ax, zs[2])

ax = fig.add_subplot(nplots_y, nplots_x, 3)
croom(3, ax, zs[3])

ax = fig.add_subplot(nplots_y, nplots_x, 4)
croom(4, ax, zs[4], yticklabels=True, nofirstylabel=False, xticklabels=True)

ax = fig.add_subplot(nplots_y, nplots_x, 5)
croom(5, ax, zs[5], xticklabels=True)

ax = fig.add_subplot(nplots_y, nplots_x, 6)
croom(6, ax, zs[6], xticklabels=True, plotmybins=False, nolastxlabel=False)

fig.text(0.5, 0.02, r'$M_g [z=2]$', transform=fig.transFigure,
         horizontalalignment='center', verticalalignment='center')

fig.text(0.03, 0.5,
         r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$',
         transform=fig.transFigure, horizontalalignment='center',
         verticalalignment='center', rotation='vertical')

fig.text(0.5, 0.97,
         r"Croom (accounting for non-uniform tile sizes)",
         transform=fig.transFigure, horizontalalignment='center')

plt.savefig('croom.pdf')



