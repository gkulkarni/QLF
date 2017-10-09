import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '14'
import matplotlib.pyplot as plt
from drawlf import render 

def plot(lf, ax, composite=None, yticklabels=False, xticklabels=False,
         nofirstylabel=True, nolastxlabel=True, nofirstxlabel=False, legend=False):

    """

    Plot data, best fit LF, and posterior LFs.

    """

    z_plot = lf.z.mean()
    print z_plot

    render(ax, lf, composite=composite)

    ax.set_xlim(-18.0, -30.0)
    ax.set_ylim(-12.0, -4.0)

    print ax.get_ylim()

    ax.set_xticks(np.arange(-30, -16, 4))
    ax.set_yticks(np.arange(-12, -3, 2))

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=3, width=1)
    ax.tick_params('both', which='minor', length=1.5, width=1)
    
    if not yticklabels:
        ax.set_yticklabels('')

    if not xticklabels:
        ax.set_xticklabels('')

    if nofirstylabel:
        ax.get_yticklabels()[0].set_visible(False)

    if nolastxlabel:
        ax.get_xticklabels()[0].set_visible(False)

    if nofirstxlabel:
        ax.get_xticklabels()[-1].set_visible(False)

    label = r'${:g}\leq z<{:g}$'.format(lf.zlims[0], lf.zlims[1])
    plt.text(0.03, 0.05, label, horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, fontsize='10')

    num = r'{:d} ({:d}) quasars'.format(lf.M1450.size, lf.M1450_all.size)
    plt.text(0.03, 0.12, num, horizontalalignment='left',
             verticalalignment='center', transform=ax.transAxes, fontsize='10')

    return 
    
def draw(lfs, composite=None):

    nplots_x = 5
    nplots_y = 5
    nplots = nplots_x * nplots_y

    plot_number = 0

    nx = nplots_x
    ny = nplots_y 
    factor_x = 2.
    factor_y = 2.
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

    for i in range(nplots):

        ax = fig.add_subplot(nplots_y, nplots_x, i+1)
        print 'plotting', i

        idx_offset=0
        
        if i in set([0,5,10,15]):
            plot(lfs[i+idx_offset], ax, composite=composite, yticklabels=True)
        elif i in set([21,22,23]):
            plot(lfs[i+idx_offset], ax, composite=composite, xticklabels=True)
        elif i == 20:
            plot(lfs[i+idx_offset], ax, composite=composite, yticklabels=True, xticklabels=True, nofirstylabel=False)
        elif i == 24:
            plot(lfs[i+idx_offset], ax, composite=composite, xticklabels=True, nolastxlabel=False)
        else:
            plot(lfs[i+idx_offset], ax, composite=composite)
        
    fig.text(0.5, 0.01, r'$M_{1450}$', transform=fig.transFigure,
             horizontalalignment='center', verticalalignment='center')

    fig.text(0.03, 0.5, r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$',
             transform=fig.transFigure, horizontalalignment='center',
             verticalalignment='center', rotation='vertical')

    plt.savefig('mosaic_small.pdf')

    plt.close('all')

