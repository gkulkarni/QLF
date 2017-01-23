import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '16'
import matplotlib.pyplot as plt
from drawlf import render 

def plot(lf, ax, yticklabels=False, xticklabels=False,
         nofirstylabel=True, nolastxlabel=True, legend=True):

    """

    Plot data, best fit LF, and posterior LFs.

    """

    z_plot = lf.z.mean()
    print z_plot
    
    render(ax, lf)

    ax.set_xlim(-18.0, -30.0)
    ax.set_ylim(-12.0, -5.0)
    # ax.set_xticks(np.arange(-31,-16, 2))
    ax.set_xticks(np.arange(-30, -16, 3))

    if not yticklabels:
        ax.set_yticklabels('')

    if not xticklabels:
        ax.set_xticklabels('')

    if nofirstylabel:
        ax.get_yticklabels()[0].set_visible(False)

    if nolastxlabel:
        ax.get_xticklabels()[0].set_visible(False)

    if legend:
        legend_title = r'${:g}\leq z<{:g}$'.format(lf.zlims[0], lf.zlims[1]) 
        l = plt.legend(loc='lower left', fontsize=8, handlelength=3,
                   frameon=False, framealpha=0.0, labelspacing=.1,
                   handletextpad=-0.5, borderpad=0.2, scatterpoints=1,
                   title=legend_title)
        plt.setp(l.get_title(),fontsize=10)
        
    return 
    
def draw(lfs):

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

    for i in range(nplots):

        ax = fig.add_subplot(nplots_y, nplots_x, i+1)
        print 'plotting', i

        idx_offset=9
        
        if i in set([0,3]):
            plot(lfs[i+idx_offset], ax, yticklabels=True)
        elif i == 7:
            plot(lfs[i+idx_offset], ax, xticklabels=True)
        elif i == 8:
            plot(lfs[i+idx_offset], ax, xticklabels=True, nolastxlabel=False)
        elif i == 6:
            plot(lfs[i+idx_offset], ax, yticklabels=True, xticklabels=True, nofirstylabel=False)
        elif i == 2:
            plot(lfs[i+idx_offset], ax)
        else:
            plot(lfs[i+idx_offset], ax)
        
    fig.text(0.5, 0.02, r'$M_{1450}$', transform=fig.transFigure,
             horizontalalignment='center', verticalalignment='center')

    fig.text(0.03, 0.5, r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$',
             transform=fig.transFigure, horizontalalignment='center',
             verticalalignment='center', rotation='vertical')

    plt.savefig('mosaic.pdf')

    plt.close('all')

