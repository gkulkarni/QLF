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
    qlumfiles = ['Data_new/dr7z2p2_sample.dat',
                 'Data_new/croom09sgp_sample.dat',
                 'Data_new/croom09ngp_sample.dat']

    selnfiles = [('Selmaps_with_tiles/dr7z2p2_selfunc.dat',
                  6248.0, 13,
                  r'SDSS DR7 Richards et al.\ 2006'),
                 
                 ('Selmaps_with_tiles/croom09sgp_selfunc.dat',
                  64.2, 15,
                  r'2SLAQ Croom et al.\ 2009'),

                 ('Selmaps_with_tiles/croom09ngp_selfunc.dat',
                  127.7, 15,
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
    ax.scatter(m, phi, c='#ffffff', s=30, label='Croom09 (2SLAQ + SDSS DR3)$^1$',
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

        # Plot SDSS DR7 LF
        sid_sdss = 13
        lfbins = drawlf.get_lf(lfi, sid_sdss, z_plot, special='croom_comparison')
        mags, left, right, logphi, uperr, downerr = lfbins
        # Convert from M1450 to Mg(z=2) by using Equation B8 of Ross
        # et al. 2013.
        mags = mags - 1.23
        ax.scatter(mags, logphi, c='b', edgecolor='None',
                   zorder=2, label='Our binning (SDSS DR7)', s=35)
        ax.errorbar(mags, logphi, ecolor='b', capsize=0,
                    yerr=np.vstack((uperr, downerr)),
                    fmt='None',zorder=2)
        
    print 'i=', i

    if i == 1:
        mags = np.linspace(-16, -31)

        p = [-5.944, -24.059, -3.028, -1.490]
        plt.plot(mags, log10phi(p, mags), c='r', lw=2)

        p = [-6.467, -23.570, -3.424, -1.610]
        plt.plot(mags-1.23, log10phi(p, mags), c='g', lw=2)

    # p = [-6.426, -23.499, -3.382, -1.577]
    # plt.plot(mags-1.23, log10phi(p, mags), c='peru', lw=2, dashes=[7,2])

    #     p = [-6.341, -23.398, -3.429, -1.430]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='k', lw=2)

    #     p = [-6.467, -23.569, -3.427, -1.610]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='m', lw=2)

    #     p = [-6.048, -22.797, -2.984, -1.425]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='brown', lw=2)

    #     p = [-6.244, -23.215, -3.254, -1.470]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='dodgerblue', lw=2)
        
    if i == 2:

        mags = np.linspace(-16, -31)

        p = [-5.557, -24.203, -3.059, -1.234]
        c, = plt.plot(mags, log10phi(p, mags), c='r', lw=2)

        p = [ -6.634, -24.774, -3.798, -1.850]
        o, = plt.plot(mags-1.23, log10phi(p, mags), c='g', lw=2)

    #     p = [-6.360, -24.354, -3.481, -1.637]
    #     c21, = plt.plot(mags-1.23, log10phi(p, mags), c='peru', lw=2, dashes=[7,2])

    #     p = [-6.671, -24.846, -3.967, -1.746]
    #     c22, = plt.plot(mags-1.23, log10phi(p, mags), c='k', lw=2)

    #     p = [-6.637, -24.778, -3.801, -1.851]
    #     c23, = plt.plot(mags-1.23, log10phi(p, mags), c='m', lw=2)

    #     p = [-6.398, -24.412, -3.474, -1.761]
    #     c25, = plt.plot(mags-1.23, log10phi(p, mags), c='brown', lw=2)

    #     p = [-6.270, -24.162, -3.239, -1.599]
    #     c26, = plt.plot(mags-1.23, log10phi(p, mags), c='dodgerblue', lw=2)
        
    if i == 3:
        mags = np.linspace(-16, -31)

    #     p = [-6.176, -24.886, -3.635, -1.570]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='peru', lw=2, dashes=[7,2])

    #     p = [-6.312, -25.094, -3.804, -1.644]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='k', lw=2)

    #     p = [-6.428, -25.257, -3.935, -1.828]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='m', lw=2)

        p = [-5.525, -24.953, -3.115, -1.071]
        plt.plot(mags, log10phi(p, mags), c='r', lw=2)

        p = [-6.428, -25.257, -3.932, -1.828]
        plt.plot(mags-1.23, log10phi(p, mags), c='g', lw=2) 
        
    #     p = [-6.315 , -25.078, -3.806, -1.787]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='brown', lw=2)

    #     p = [-5.803, -24.205, -3.188, -1.313]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='dodgerblue', lw=2)
        
    if i == 4:
        mags = np.linspace(-16, -31)
    #     p = [-6.347, -25.546, -3.961, -1.668]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='peru', lw=2, dashes=[7,2])

    #     p = [-6.459, -25.699, -4.120, -1.726]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='k', lw=2)

    #     p = [-6.500, -25.753, -4.170, -1.833]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='m', lw=2)

        p = [-6.136, -26.411, -3.653, -1.620]
        plt.plot(mags, log10phi(p, mags), c='r', lw=2)

        p = [-6.499, -25.751, -4.170, -1.832]
        plt.plot(mags-1.23, log10phi(p, mags), c='g', lw=2)

    #     p = [-6.433, -25.667, -3.937, -1.807]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='brown', lw=2)

    #     p = [-6.115, -25.161, -3.464, -1.541]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='dodgerblue', lw=2)
        
    if i == 5:
        mags = np.linspace(-16, -31)

        p = [-5.795, -26.321, -3.612, -1.274]
        plt.plot(mags, log10phi(p, mags), c='r', lw=2)

        p = [-6.592, -26.138, -4.004, -1.905]
        plt.plot(mags-1.23, log10phi(p, mags), c='g', lw=2)
        
    #     p = [-6.348, -25.793, -3.709, -1.704]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='peru', lw=2, dashes=[7,2])

    #     p = [-6.348, -25.794, -3.707, -1.704]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='k', lw=2)

    #     p = [-6.592, -26.137, -4.003, -1.904]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='m', lw=2)

    #     p = [-6.098, -25.429, -3.473, -1.614]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='brown', lw=2)

    #     p = [-5.658, -24.649, -3.100, -1.094]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='dodgerblue', lw=2)

        
    # if i == 7:
    #     mags = np.linspace(-19, -31)
    #    # p = [-5.341, -23.341, -2.773, -0.890]
    #    # print log10phi(p, mags)
    #     #plt.plot(mags, log10phi(p, mags), c='k')
    #     p = [-5.557, -24.203, -3.059, -1.234]
    #     plt.plot(mags, log10phi(p, mags), c='k', lw=2)
    #     #p = [-5.667, -24.479, -3.004, -1.280]
    #     #plt.plot(mags, log10phi(p, mags), c='k')
    #     #p = [-5.553, -24.561, -3.056, -1.075]
    #     #plt.plot(mags, log10phi(p, mags), c='k')

    #     #p = [-6.683, -24.385, -3.722, -1.939]
    #     #plt.plot(mags-1.23, log10phi(p, mags), c='c', lw=2)
    #     p = [-6.567, -24.737, -3.805, -1.936]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='c', lw=2)

    #     #p = [-6.503, -24.162, -3.680, -1.789]
    #     #plt.plot(mags-1.23, log10phi(p, mags), c='m', lw=2)
    #     p = [-6.465, -24.619, -3.799, -1.785]
    #     plt.plot(mags-1.23, log10phi(p, mags), c='m', lw=2)



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
        plt.legend(loc='upper right', fontsize=8, handlelength=3,
                   frameon=False, framealpha=0.0, labelspacing=.1,
                   handletextpad=0., borderpad=0.2, scatterpoints=1)

    if i == 2:
        handles = [c, o]
        labels = ['Croom09 fit$^2$', 'Our fit' ]
        l1 = plt.legend(handles, labels, loc='upper right', fontsize=8, handlelength=3,
                        frameon=False, framealpha=0.0, labelspacing=.1, ncol=1,
                        handletextpad=0.4, borderpad=0.5)



    return 

nplots_x = 3
nplots_y = 2
nplots = nplots_x * nplots_y
nx = nplots_x
ny = nplots_y 
factor_x = 2.5
factor_y = 3.3
ldim = 0.35*factor_x
bdim = 0.4*factor_y
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

fig.text(0.5, 0.1, r'$M_g [z=2]$', transform=fig.transFigure,
         horizontalalignment='center', verticalalignment='center')

fig.text(0.03, 0.52,
         r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$',
         transform=fig.transFigure, horizontalalignment='center',
         verticalalignment='center', rotation='vertical')

fig.text(0.1, 0.07, 
         r'$^1$ Using SDSS DR3 instead of DR7 for our fits does not change them significantly.',
         transform=fig.transFigure, horizontalalignment='left',
         verticalalignment='center', fontsize=10)

fig.text(0.1, 0.05, 
         r'$^2$ Croom09 provide fits for narrower redshifts bins. Fit shown here corresponds to the bin that is closest to the center',
         transform=fig.transFigure, horizontalalignment='left',
         verticalalignment='center', fontsize=10)

fig.text(0.112, 0.03, 
         r' of the redshift range in each panel.',
         transform=fig.transFigure, horizontalalignment='left',
         verticalalignment='center', fontsize=10)

# fig.text(0.5, 0.97,
#          r"Croom (accounting for non-uniform tile sizes)",
#          transform=fig.transFigure, horizontalalignment='center')

plt.savefig('croom.pdf')



