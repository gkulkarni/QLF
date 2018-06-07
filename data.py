import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
from random import shuffle
import sys 

"""

Histogram of all quasar data used in this study.

"""

def getqlums(lumfile):

    """Read quasar luminosities."""

    with open(lumfile,'r') as f: 
        z, mag, p = np.loadtxt(lumfile, usecols=(1,2,3), unpack=True)
        
    return z, mag, p

class sample:

    def __init__(self, sample_data_files, color='None', label=None):

        for f in sample_data_files: 
            z, m, p = getqlums(f)
            try:
                self.z = np.append(self.z, z)
                self.m = np.append(self.m, m)
            except(AttributeError):
                self.z = z
                self.m = m 
        
        self.color = color

        if label is not None: 
            self.label = label

        return 

def plot_data(data):

    fig = plt.figure(figsize=(14, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    plt.minorticks_on()
    ax.tick_params('both', which='major', length=7, width=1, direction='in')
    ax.tick_params('both', which='minor', length=3, width=1, direction='in')
    ax.set_yscale('log')

    bin_width = 0.1
    bins = np.arange(0.0, 7.0, bin_width)

    # Colours were obtained from Matplotlib 2.0 colour palette
    # (mpl.rcParams['axes.prop_cycle']), combined with some colours
    # from seaborn.color_palette('husl', 16).as_hex() and
    # colorbrewer2.org.
    cs = [u'#1f77b4', u'#ff7f0e', u'#2ca02c', u'#d62728', u'#9467bd',
          u'#8c564b', u'#e377c2', u'#7f7f7f', u'#bcbd22', u'#17becf',
          u'#77ab31']#, u'#d8b365']#, u'#cc7af4', u'#f66bad']

    d = [x.z for x in data]
    l = [x.label for x in data]
    plt.hist(d, histtype='bar', stacked=True, rwidth=1.0,
             ec='None', bins=bins, color=cs, label=l, lw=0.0)

    # Three qsos added by hand 
    z = 7.085
    zlim = (z-bin_width/2.0, z+bin_width/2.0)
    nbins = 1
    name = r'UKIDSS Mortlock et al.\ (2011)'
    n, bins, patches = plt.hist(z, bins=nbins, range=zlim, color=u'#beaed4', 
                                histtype='stepfilled', ec='None',
                                label=name)

    z = 6.530 
    zlim = (z-bin_width/2.0, z+bin_width/2.0)
    nbins = 1
    name = r'UKIDSS Venemans et al.\ (2015)'
    n, bins, patches = plt.hist(z, bins=nbins, range=zlim, color=u'#f0027f', 
                                histtype='stepfilled', ec='None',
                                label=name)

    z = 7.54 
    zlim = (z-bin_width/2.0, z+bin_width/2.0)
    nbins = 1
    name = r'ALLWISE+UKIDSS+DECaLS Ba\~nados et al.\ (2018)'
    n, bins, patches = plt.hist(z, bins=nbins, range=zlim, color=u'#bf5b17', 
                                histtype='stepfilled', ec='None',
                                label=name)
    
    ax.set_xlabel(r'redshift')
    ax.set_ylabel(r'Number of AGN')

    plt.ylim(7e-1, 1.0e4)
    plt.xlim(0., 8.)

    plt.legend(loc='upper right', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.2,markerscale=.5)

    plt.savefig('qsos4.pdf', bbox_inches='tight')

    return

data = []

f = ['Data_new/dr7z2p2_sample.dat',
     'Data_new/dr7z3p7_sample.dat']
l = r'SDSS DR7 Schneider et al.\ (2010)'
s = sample(f, label=l)
s.z = s.z[s.z<4.7]
data.append(s)

f = ['Data_new/croom09sgp_sample.dat',
     'Data_new/croom09ngp_sample.dat']
l = r'2SLAQ Croom et al.\ (2009)'
s = sample(f, label=l)
s.z = s.z[s.z<2.2]
data.append(s)

f = ['Data_new/bossdr9color.dat']
l = r'BOSS DR9 colour-selected Ross et al.\ (2013)'
s = sample(f, label=l)
data.append(s)

f = ['Data_new/yang16_sample.dat']
l = r'SDSS+Wise Yang et al.\ (2016)'
s = sample(f, label=l)
data.append(s)

f = ['Data_new/mcgreer13_dr7sample.dat',
     'Data_new/mcgreer13_dr7extend.dat']
l = r'SDSS DR7 McGreer et al.\ (2013)'
s = sample(f, label=l)
s.z = s.z[s.m>-26.73]
data.append(s)

f = ['Data_new/mcgreer13_s82sample.dat',
     'Data_new/mcgreer13_s82extend.dat']
l = r'SDSS Stripe 82 McGreer et al.\ (2013)'
s = sample(f, label=l)
s.z = s.z[s.m>-26.73]
data.append(s)

f = ['Data/glikman11qso.dat']
l = r'NDWFS+DLS Glikman et al.\ (2011)'
s = sample(f, label=l)
data.append(s)

f = ['Data_new/giallongo15_sample.dat']
l = r'CANDELS GOODS-S Giallongo et al.\ (2015)'
s = sample(f, label=l)
data.append(s)

f = ['Data_new/jiang16main_sample.dat', 'Data_new/jiang16overlap_sample.dat', 'Data_new/jiang16s82_sample.dat']
l = r'SDSS Jiang et al.\ (2016)'
s = sample(f, label=l)
data.append(s)

# f = []
# l = r'SDSS Overlap Jiang et al.\ (2016)'
# s = sample(f, label=l)
# data.append(s)

# f = []
# l = r'SDSS Stripe 82 Jiang et al.\ (2016)'
# s = sample(f, label=l)
# data.append(s)

f = ['Data_new/willott10_cfhqsdeepsample.dat',
     'Data_new/willott10_cfhqsvwsample.dat']
l = r'CFHQS Willott et al.\ (2010)'
s = sample(f, label=l)
data.append(s)

# f = []
# l = r'CFHQS Very Wide Willott et al.\ (2010)'
# s = sample(f, label=l)
# data.append(s)

f = ['Data_new/kashikawa15_sample.dat']
l = r'Subaru High-$z$ Quasar Survey Kashikawa et al.\ (2010)'
s = sample(f, label=l)
data.append(s)

sum = 0
for x in data: 
    sum += x.z.size
print 'Total number of AGN:', sum+3 # 3 qsos added by hand above.
    
plot_data(data)

