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

def plot_data(quasar_files):

    fig = plt.figure(figsize=(14, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)
    ax.set_yscale('log')

    # bin width = kbw(FULL quasar sample)
    bw = 2*0.037433628318583878

    n = len(quasar_files)

    cs = [(0.40000000596046448, 0.7607843279838562, 0.64705884456634521),
          (0.98131487965583808, 0.55538641635109398, 0.38740485135246722),
          (0.55432528607985565, 0.62711267120697922, 0.79595541393055635),
          (0.90311419262605563, 0.54185316071790801, 0.76495195557089413),
          (0.65371782148585622, 0.84708959004458262, 0.32827375098770734),
          (0.9986312957370983, 0.85096502233954041, 0.18488274134841617),
          (0.89573241682613591, 0.76784315109252932, 0.58182240093455595),
          (0.70196080207824707, 0.70196080207824707, 0.70196080207824707),
          (0.40000000596046448, 0.7607843279838562, 0.64705884456634521),
          (0.98131487965583808, 0.55538641635109398, 0.38740485135246722),
          (0.55432528607985565, 0.62711267120697922, 0.79595541393055635)]

    labels=['BOSS DR9 color-selected (Ross et al.\ 2013)',
            'SDSS DR7 with Richards et al.\ (2006) selection',
            'Glikman et al.\ (2011)',
            'SDSS DR7 with McGreer et al.\ (2013) selection',
            'Stripe 82 McGreer et al.\ (2013)',
            'Extended Stripe 82 McGreer et al.\ (2013)',
            'CFHQS Very Wide Survey Willott et al. (2010)',
            'SDSS Fan et al.\ (2006)',
            'SDSS Deep Jiang et al.\ (2009)',
            'SDSS Deep Jiang et al.\ (2008)',
            'CFHQS Deep Survey Willott et al. (2010)']
    
    for i, datafile in enumerate(quasar_files):
        z, m, p = getqlums(datafile)
        nbins = int(np.ptp(z)/bw)+1
        if z.size == 1:
            zlim = (z-bw/2.0, z+bw/2.0)
            n, bins, patches = plt.hist(z, bins=nbins, range=zlim, color=cs[i],
                                        histtype='stepfilled', ec='none', label=labels[i])
        else:
            n, bins, patches = plt.hist(z, bins=nbins, color=cs[i],
                                        histtype='stepfilled', ec='none', label=labels[i])
        
        
    ax.set_xlabel(r'redshift')
    ax.set_ylabel(r'Number of quasars')

    plt.ylim(7e-1, 1.0e4)

    plt.legend(loc='upper right', fontsize=14, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.2)

    plt.savefig('qsos.pdf', bbox_inches='tight')

    return

qlumfiles = ['Data/bossdr9color.dat',
             'Data/dr7z3p7.dat',
             'Data/glikman11qso.dat',
             'Data/mcgreer13_dr7sample.dat',
             'Data/mcgreer13_s82sample.dat',
             'Data/mcgreer13_s82extend.dat',
             'Data/willott10_cfhqsvwsample.dat',
             'Data/fan06_sample.dat',
             'Data/jiang09_sample.dat',
             'Data/jiang08_sample.dat',
             'Data/willott10_cfhqsdeepsample.dat'
             ]

plot_data(qlumfiles)


