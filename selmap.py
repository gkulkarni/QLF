import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import interp2d, bisplrep, bisplev
from scipy.interpolate import LinearNDInterpolator as linInterp
import sys

def plot_selmap(z, m, p, title='', filename='selmap.pdf',
                show_qsos=False, qso_file='None'):     

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)
    ax.tick_params('x', which='major', pad=6)

    s = plt.scatter(z, m, s=100, c=p, vmin=0.0, vmax=1.0,
                    edgecolor='none', marker='s',
                    rasterized=True, cmap=cm.jet)

    if show_qsos:
        zq, mq = np.loadtxt(qso_file, usecols=(1,2), unpack=True)
        plt.scatter(zq, mq, s=10, c='#ffffff', edgecolor='k')

    plt.xlabel('$z$')
    plt.ylabel('$M_{1450}$')
    plt.title(title, y='1.01')

    plt.xlim(0.0, 2.5) 
    plt.ylim(-28, -14)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", "5%", pad="3%")
    cb = plt.colorbar(s, cax=cax)
    cb.set_label(r'selection probability', labelpad=20)
    cb.solids.set_edgecolor("face")

    plt.savefig(filename, bbox_inches='tight')


# map_file = 'Data_new/croom09ngp_selfunc.dat'
# qso_file = 'Data_new/croom09ngp_sample.dat'
# out_map_file = 'croom09ngp_selfunc_interpolated_linear.dat'

METHOD = 'linear'
map_file = 'Data_new/' + sys.argv[1]
out_map_file = 'smooth_maps/' + sys.argv[1]
# zmin = np.float(sys.argv[2])
# zmax = np.float(sys.argv[3])
# Mbright = np.float(sys.argv[4])
# Mfaint = np.float(sys.argv[5])

with open(map_file, 'r') as f:
    z, m, p = np.loadtxt(f, usecols=(1,2,3), unpack=True)

zmin = np.min(z) - 0.5
if zmin < 0.0:
    zmin = 0.0 
zmax = np.max(z) + 0.5

Mbright = np.min(m) - 1.0 
Mfaint = np.max(m) + 1.0

Mbright = -16
Mfaint = -32 

print 'Using zmin = {:g} and zmax = {:g}'.format(zmin, zmax)
print 'Using Mbright = {:g} and Mfaint = {:g}'.format(Mbright, Mfaint) 

# plot_selmap(z, m, p, title='2SLAQ NGP', filename='selmap_ngp.pdf',
#             show_qsos=False, qso_file=qso_file)

if METHOD == 'spline': 
    tck = bisplrep(z, m, p)
    # znew = np.linspace(0, 3.5, num=500)
    # mnew = np.linspace(-30, -16, num=500)

    znew = np.linspace(zmin, zmax, num=500)
    mnew = np.linspace(Mbright, Mfaint, num=500)
                         
    pnew = bisplev(znew, mnew, tck)
    zp, mp = np.meshgrid(znew, mnew, indexing='ij')

if METHOD == 'linear':
    points = np.vstack((z,m)).T
    f = linInterp(points, p)
    
    # znew = np.linspace(0, 3.5, num=500)
    # mnew = np.linspace(-30, -16, num=500)

    znew = np.linspace(zmin, zmax, num=500)
    mnew = np.linspace(Mbright, Mfaint, num=500)
    
    zp, mp = np.meshgrid(znew, mnew, indexing='ij')
    points_new = np.vstack((zp.flatten(), mp.flatten())).T
    pnew = f(points_new)

# plot_selmap(zp, mp, pnew, title='2SLAQ NGP (interpolated)',
#             filename='selmap_ngp_interpolated.pdf',
#             show_qsos=False, qso_file=qso_file)

WRITE_SELMAP = True
if WRITE_SELMAP:
    with open(out_map_file, 'w') as fl:
        for i, x in enumerate(zip(zp.flatten(), mp.flatten(), pnew.flatten())):
            if x[2]==x[2]: 
                fl.write('{:d}  {:.2f}  {:.2f}  {:.6f}\n'.format(i, x[0],
                                                                 x[1], x[2]))
            else:
                fl.write('{:d}  {:.2f}  {:.2f}  {:.6f}\n'.format(i, x[0],
                                                                 x[1], 0.0))                                                                 

# if METHOD == 'linear': 
#     zq, mq, pq = np.loadtxt(qso_file, usecols=(1,2,3), unpack=True)
#     points_q = np.vstack((zq, mq)).T
#     pq_new = f(points_q)

#     fig = plt.figure(figsize=(7, 7), dpi=100)
#     ax = fig.add_subplot(1, 1, 1)
#     ax.tick_params('both', which='major', length=7, width=1)
#     ax.tick_params('both', which='minor', length=3, width=1)
#     ax.tick_params('x', which='major', pad=6)

#     plt.scatter(pq, pq_new, s=10, c='#ffffff', edgecolor='k')
#     p = np.linspace(0,1)
#     plt.plot(p, p, lw=2, c='tomato')

#     plt.xlabel('$p$')
#     plt.ylabel('$p_\mathrm{new}$')

#     plt.xlim(0.0, 1.0)
#     plt.ylim(0.0, 1.0)

#     plt.savefig('compare_p.pdf', bbox_inches='tight')
