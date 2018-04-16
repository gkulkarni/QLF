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
         nofirstylabel=True, nolastxlabel=True, nofirstxlabel=False, legend=False, c2=None, c3=None):

    """

    Plot data, best fit LF, and posterior LFs.

    """

    z_plot = lf.z.mean()
    print z_plot

    ind, co1, co2, co3 = render(ax, lf, composite=composite, c2=c2, c3=c3)

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

    return ind, co1, co2, co3
    
def draw(lfs, composite=None, c2=None, c3=None):

    nplots_x = 5
    nplots_y = 3
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



    ax = fig.add_subplot(nplots_y, nplots_x, 1)
    plot(lfs[0], ax, composite=composite, yticklabels=True, c2=c2, c3=c3)
    ax = fig.add_subplot(nplots_y, nplots_x, 2)
    plot(lfs[1], ax, composite=composite, c2=c2, c3=c3)
    ax = fig.add_subplot(nplots_y, nplots_x, 3)
    plot(lfs[2], ax, composite=composite, c2=c2, c3=c3)
    ax = fig.add_subplot(nplots_y, nplots_x, 4)
    plot(lfs[3], ax, composite=composite, c2=c2, c3=c3)
    ax = fig.add_subplot(nplots_y, nplots_x, 5)
    plot(lfs[4], ax, composite=composite, c2=c2, c3=c3)
    ax = fig.add_subplot(nplots_y, nplots_x, 6)
    plot(lfs[5], ax, composite=composite, yticklabels=True, c2=c2, c3=c3)
    ax = fig.add_subplot(nplots_y, nplots_x, 7)
    plot(lfs[6], ax, composite=composite, xticklabels=True, c2=c2, c3=c3, nofirstxlabel=True)
    ax = fig.add_subplot(nplots_y, nplots_x, 8)
    plot(lfs[7], ax, composite=composite, xticklabels=True, c2=c2, c3=c3)
    ax = fig.add_subplot(nplots_y, nplots_x, 9)
    plot(lfs[8], ax, composite=composite, xticklabels=True, c2=c2, c3=c3)
    ax = fig.add_subplot(nplots_y, nplots_x, 10)
    plot(lfs[9], ax, composite=composite, xticklabels=True, c2=c2, c3=c3, nolastxlabel=False)
    ax = fig.add_subplot(nplots_y, nplots_x, 11)
    ind, co1, co2, co3 = plot(lfs[10], ax, composite=composite, yticklabels=True, c2=c2, c3=c3, xticklabels=True, nolastxlabel=False, nofirstylabel=False)

    # if i in set([0,5,10,15]):
    #     plot(lfs[i+idx_offset], ax, composite=composite, yticklabels=True, c2=c2, c3=c3)
    # elif i in set([7,8,9]):
    #     plot(lfs[i+idx_offset], ax, composite=composite, xticklabels=True, c2=c2, c3=c3)
    # elif i == 20:
    #     plot(lfs[i+idx_offset], ax, composite=composite, yticklabels=True, xticklabels=True, nofirstylabel=False, c2=c2, c3=c3)
    # elif i == 24:
    #     plot(lfs[i+idx_offset], ax, composite=composite, xticklabels=True, nolastxlabel=False, c2=c2, c3=c3)
    # else:
    #     ind, co1, co2, co3 = plot(lfs[i+idx_offset], ax, composite=composite, c2=c2, c3=c3)

    fig.text(0.5, 0.31, r'$M_{1450}$', transform=fig.transFigure,
             horizontalalignment='center', verticalalignment='center')

    fig.text(0.03, 0.5, r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$',
             transform=fig.transFigure, horizontalalignment='center',
             verticalalignment='center', rotation='vertical')

    handles, labels = [], []
    handles.append(co1)
    labels.append('Model 1')

    handles.append(co2)
    labels.append('Model 2')

    handles.append(co3)
    labels.append('Model 3')
    
    handles.append(ind)
    labels.append('Double power law in redshift bin')

    plt.legend(handles, labels, loc='upper left', fontsize=14, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.3, borderpad=0.01,
               scatterpoints=1, ncol=1, bbox_to_anchor=[1.0,0.4])
    
    plt.savefig('mosaic_small.pdf')

    plt.close('all')

