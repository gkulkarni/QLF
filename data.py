import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
from random import shuffle

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
    bins = np.arange(0.0, 7.0, bin_width)

    # seaborn.color_palette('husl', 16).as_hex()
    cs = [u'#f77189', u'#f7754f', u'#dc8932', u'#c39532', u'#ae9d31', u'#97a431',
          u'#77ab31', u'#31b33e', u'#33b07a', u'#35ae93', u'#36ada4', u'#37abb4',
          u'#38a9c5', u'#3aa5df', u'#6e9bf4', u'#a48cf4', u'#cc7af4', u'#f45cf2',
          u'#f565cc', u'#f66bad']

    shuffle(cs)
    print cs

    cs = [u'#a48cf4', u'#f7754f', u'#3aa5df', u'#f565cc', u'#6e9bf4',
          u'#31b33e', u'#33b07a', u'#dc8932', u'#f77189', u'#cc7af4',
          u'#ae9d31', u'#77ab31', u'#35ae93', u'#97a431', u'#f45cf2',
          u'#37abb4', u'#36ada4']#, u'#f66bad', u'#38a9c5', u'#c39532']

    d = [x.z for x in data]
    print len(d), len(cs)
    l = [x.label for x in data]
    plt.hist(d, histtype='bar', stacked=True, rwidth=1.0, ec='None', bins=bins, color=cs, label=l, lw=0.0)
    
    z = 7.085
    zlim = (z-bin_width/2.0, z+bin_width/2.0)
    nbins = 1
    n, bins, patches = plt.hist(z, bins=nbins, range=zlim, color=u'#f77189', 
                                histtype='stepfilled', ec='None', label='UKIDSS Mortlock et al.\ (2011)')
            
    ax.set_xlabel(r'redshift')
    ax.set_ylabel(r'Number of quasars')

    plt.ylim(7e-1, 5.0e4)
    plt.xlim(0., 8.)

    plt.legend(loc='upper right', fontsize=10, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.2,markerscale=.5)

    plt.savefig('qsos4.pdf', bbox_inches='tight')

    return

# seaborn.color_palette('husl', 16).as_hex()
cs = [u'#f77189', u'#f7754f', u'#dc8932', u'#c39532', u'#ae9d31', u'#97a431',
      u'#77ab31', u'#31b33e', u'#33b07a', u'#35ae93', u'#36ada4', u'#37abb4',
      u'#38a9c5', u'#3aa5df', u'#6e9bf4', u'#a48cf4', u'#cc7af4', u'#f45cf2',
      u'#f565cc', u'#f66bad']

shuffle(cs)

data = []

f = ['Data_new/dr7z2p2_sample.dat']
l = r'SDSS DR7 with Richards et al.\ (2006) selection function'
s = sample(f, color=cs[0], label=l)
data.append(s)

f = ['Data_new/croom09sgp_sample.dat',
     'Data_new/croom09ngp_sample.dat']
l = r'2SLAQ NGP+SGP Croom et al.\ (2009a, 2009b)'
s = sample(f, color=cs[1], label=l)
data.append(s)

f = ['Data_new/dr7z3p7_sample.dat']
l = r'SDSS DR7 with Richards et al.\ (2006) selection function'
s = sample(f, color=cs[0], label=l)
# Use only up to z = 4.7 to avoid overlap with McGreer and Yang 
s.z = s.z[s.z<4.7]
data.append(s)

f = ['Data_new/bossdr9color.dat']
l = r'BOSS DR9 colour-selected Ross et al.\ (2013)'
s = sample(f, color=cs[2], label=l)
data.append(s)

f = ['Data_new/yang16_sample.dat']
l = r'SDSS+Wise Yang et al.\ (2016)'
s = sample(f, color=cs[3], label=l)
data.append(s)

f = ['Data_new/mcgreer13_dr7sample.dat']
l = r'SDSS DR7 McGreer et al.\ (2013)'
s = sample(f, color=cs[4], label=l)
data.append(s)

f = ['Data_new/mcgreer13_s82sample.dat']
l = r'SDSS Stripe 82 McGreer et al.\ (2013)'
s = sample(f, color=cs[5], label=l)
data.append(s)

f = ['Data_new/mcgreer13_dr7extend.dat']
l = r'SDSS DR7 extended McGreer et al.\ (2013)'
s = sample(f, color=cs[6], label=l)
data.append(s)

f = ['Data_new/mcgreer13_s82extend.dat']
l = r'SDSS Stripe 82 extended McGreer et al.\ (2013)'
s = sample(f, color=cs[7], label=l)
data.append(s)

f = ['Data/glikman11qso.dat']
l = r'NDWFS+DLS Glikman et al.\ (2011)'
s = sample(f, color=cs[8], label=l)
data.append(s)

f = ['Data_new/giallongo15_sample.dat']
l = r'CANDELS GOODS-S Giallongo et al.\ (2015)'
s = sample(f, color=cs[9], label=l)
data.append(s)

f = ['Data_new/jiang16main_sample.dat']
l = r'SDSS Main Jiang et al.\ (2016)'
s = sample(f, color=cs[10], label=l)
data.append(s)

f = ['Data_new/jiang16overlap_sample.dat']
l = r'SDSS Overlap Jiang et al.\ (2016)'
s = sample(f, color=cs[11], label=l)
data.append(s)

f = ['Data_new/jiang16s82_sample.dat']
l = r'SDSS Stripe 82 Jiang et al.\ (2016)'
s = sample(f, color=cs[12], label=l)
data.append(s)

f = ['Data_new/willott10_cfhqsdeepsample.dat']
l = r'CFHQS Deep Willott et al.\ (2010)'
s = sample(f, color=cs[13], label=l)
data.append(s)

f = ['Data_new/willott10_cfhqsvwsample.dat']
l = r'CFHQS Very Wide Willott et al.\ (2010)'
s = sample(f, color=cs[14], label=l)
data.append(s)

f = ['Data_new/kashikawa15_sample.dat']
l = r'Subaru High-$z$ Quasar Survey Kashikawa et al.\ (2010)'
s = sample(f, color=cs[15], label=l)
data.append(s)

plot_data(data)

