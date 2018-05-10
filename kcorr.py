import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
import sys

fig = plt.figure(figsize=(7, 7/1.68), dpi=100)
ax = fig.add_subplot(1, 1, 1)
ax.set_ylabel(r'$K_{m,1450}$')
ax.set_xlabel('$z$')

plt.minorticks_on()
ax.tick_params('both', which='major', length=7, width=1)
ax.tick_params('both', which='minor', length=3, width=1)
ax.tick_params('x', which='major', pad=6)

z, k = np.loadtxt('Data_new/kcorrg_l15.dat', usecols=(1,2), unpack=True)
plt.plot(z, k, c='b', lw=2) 

z, k = np.loadtxt('Data_new/kcorrg_t02.dat', usecols=(1,2), unpack=True)
plt.plot(z, k, c='b', lw=2, dashes=[7,2]) 

z, k = np.loadtxt('Data_new/kcorrg_v01.dat', usecols=(1,2), unpack=True)
plt.plot(z, k, c='b', lw=2, dashes=[2,2])

plt.text(0.35, -0.75, '$m=g$', fontsize=12)

#-----

z, k = np.loadtxt('Data_new/kcorri_l15.dat', usecols=(1,2), unpack=True)
plt.plot(z, k, c='r', lw=2) 

z, k = np.loadtxt('Data_new/kcorri_t02.dat', usecols=(1,2), unpack=True)
plt.plot(z, k, c='r', lw=2, dashes=[7,2]) 

z, k = np.loadtxt('Data_new/kcorri_v01.dat', usecols=(1,2), unpack=True)
plt.plot(z, k, c='r', lw=2, dashes=[2,2])

plt.text(2.0, -1.8, '$m=i$', fontsize=12)

#-----

z, k = np.loadtxt('Data_new/kcorrz_l15.dat', usecols=(1,2), unpack=True)
plt.plot(z, k, c='k', lw=2, label='Lusso et al.\ 2015') 

z, k = np.loadtxt('Data_new/kcorrz_t02.dat', usecols=(1,2), unpack=True)
plt.plot(z, k, c='k', lw=2, dashes=[7,2], label='Telfer et al.\ 2002') 

z, k = np.loadtxt('Data_new/kcorrz_v01.dat', usecols=(1,2), unpack=True)
plt.plot(z, k, c='k', lw=2, dashes=[2,2], label='Vanden Berk et al.\ 2001')

plt.text(4.7, -2.0, '$m=z_\mathrm{AB}$', fontsize=12)

plt.ylim(-2.5,-0.5)
plt.xlim(0,5.5)
# plt.xticks(np.arange(2,6.5,1))

handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, labels, loc='upper right', fontsize=14, handlelength=3,
           frameon=False, framealpha=0.0, labelspacing=.1,
           handletextpad=0.4, borderpad=0.5)

plt.savefig('kcorr.pdf', bbox_inches='tight')

