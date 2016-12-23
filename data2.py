import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt

"""

Histogram of all quasar data used in this study.

"""


def getqlums(lumfile):

    """Read quasar luminosities."""

    with open(lumfile,'r') as f: 
        z, mag, p = np.loadtxt(lumfile, usecols=(1,2,3), unpack=True)
        
    return z, mag, p

class sample:

    def __init__(self, sample_data_files, color='None', label='None'):

        for f in sample_data_files: 
            z, m, p = getqlums(f)
            try:
                self.z = np.append(self.z, z)
            except(AttributeError):
                self.z = z
        
        self.color = color
        self.label = label

        return 
    
def plot_data(data):

    fig = plt.figure(figsize=(14, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)
    ax.set_yscale('log')

    bin_width = 0.1

    for d in data:
        nbins = int(np.ptp(d.z)/bin_width)+1
        plt.hist(d.z, bins=nbins, color=d.color, 
                 histtype='stepfilled', ec='none', label=d.label)
        
    ax.set_xlabel(r'redshift')
    ax.set_ylabel(r'Number of quasars')

    plt.ylim(7e-1, 1.0e4)
    plt.xlim(0., 8.)

    plt.legend(loc='upper right', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.2,markerscale=.5)

    plt.savefig('qsos.pdf', bbox_inches='tight')

    return

# seaborn.color_palette('husl', 8).as_hex()
cs = [u'#f77189',
      u'#ce9032',
      u'#97a431',
      u'#32b166',
      u'#36ada4',
      u'#39a7d0',
      u'#a48cf4',
      u'#f561dd'] 

data = []

f = ['Data_new/dr7z2p2_sample.dat']
l = r'$0.065<z<2.2$ SDSS DR7 with Richards et al.\ (2006) selection function'
s = sample(f, color=cs[0], label=l)
data.append(s)

f = ['Data_new/croom09sgp_sample.dat',
     'Data_new/croom09ngp_sample.dat']
l = r'$0.4<z<2.6$ 2SLAQ NGP+SGP Croom et al.\ (2009a, 2009b)'
s = sample(f, color=cs[1], label=l)
data.append(s)

f = ['Data_new/dr7z3p7_sample.dat']
l = r'$3.7<z<4.7$ SDSS DR7 with Richards et al.\ (2006) selection function'
s = sample(f, color=cs[2], label=l)
# Use only up to z = 4.7 to avoid overlap with McGreer and Yang 
s.z = s.z[s.z<4.7]
data.append(s)

f = ['Data_new/bossdr9color.dat']
l = r'$2.2<z<3.5$ BOSS DR9 colour-selected Ross et al.\ (2013)'
s = sample(f, color=cs[3], label=l)
data.append(s)

f = ['Data/glikman11qso.dat']
l = r'$3.7<z<5.2$ NDWFS+DLS Glikman et al.\ (2011)'
s = sample(f, color=cs[4], label=l)
data.append(s)


plot_data(data)

