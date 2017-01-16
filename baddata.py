import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt

def getqlums(lumfile):

    """Read quasar luminosities."""

    with open(lumfile,'r') as f: 
        z, mag, p = np.loadtxt(lumfile, usecols=(1,2,3), unpack=True)
        
    return z, mag, p

def plot_sample(sid, **kwargs): 

    qlf_file = 'Data/allqlfs.dat'
    sample_id, z_bin, z_min, z_max, nqso = np.loadtxt(qlf_file, usecols=(1,2,3,4,12), unpack=True)

    nqso = nqso[sample_id==sid]
    z_bin = z_bin[sample_id==sid]
    z_low = z_min[sample_id==sid]
    z_high = z_max[sample_id==sid]

    z_bin = np.sort(z_bin)
    sortargs = np.argsort(z_bin)
    z_low = z_low[sortargs]
    z_high = z_high[sortargs]
    nqso = nqso[sortargs]
    
    a, b = np.unique(z_bin, return_index=True)

    if len(a) == 1:
        nqso = np.array([np.sum(nqso)])
        z_bin = a
        z_low = np.unique(z_low)
        z_high = np.unique(z_high)
    else:
        nqso = np.array([np.sum(x) for x in np.split(nqso, b[1:])])
        z_bin = a
        z_low = np.unique(z_low)
        z_high = np.unique(z_high)
       # raise ValueError('more than one z bins')

    plt.bar(z_low, nqso, width=z_high-z_low, **kwargs)

    return


def plot_data(quasar_files):

    fig = plt.figure(figsize=(14, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=3, width=1)
    ax.set_yscale('log')

    # bin width = kbw(FULL quasar sample)
    bw = 2*0.037433628318583878

    n = len(quasar_files)

    cs = [(0.0, 0.10980392156862745, 0.4980392156862745),
          (0.00392156862745098, 0.4588235294117647, 0.09019607843137255),
          (0.5490196078431373, 0.03529411764705882, 0.0),
          (0.4627450980392157, 0.0, 0.6313725490196078),
          (0.7215686274509804, 0.5254901960784314, 0.043137254901960784),
          (0.0, 0.38823529411764707, 0.4549019607843137),
          (0.0, 0.10980392156862745, 0.4980392156862745),
          (0.00392156862745098, 0.4588235294117647, 0.09019607843137255),
          (0.5490196078431373, 0.03529411764705882, 0.0),
          (0.4627450980392157, 0.0, 0.6313725490196078)]
    
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

    plot_sample(2, color=cs[5], edgecolor=cs[5], label='BOSS DR9 variability-selected Ross et al.\ 2013 (no selection map)')
    plot_sample(3, color=cs[0], edgecolor=cs[0], label='BOSS+MMT Palanque-Delabrouille et al.\ 2013 (only give $M_{g, z=2}$)')
    plot_sample(4, color=cs[1], edgecolor=cs[1], label='SWIRE Siana et al.\ 2008 (high spectroscopic incompleteness')
    plot_sample(5, color=cs[2], edgecolor=cs[2], label='COSMOS Masters et al. 2012 (photometric redshifts)')
    plot_sample(12, color=cs[3], edgecolor=cs[3], label='VVDS Type 1 QLF Bongiorno et al. 2007 (uncertain $M_B$--$M_{1450}$ conversion)')
    plot_sample(14, color=cs[4], edgecolor=cs[4], label='COMBO17 Wolf et al. 2003 (why are we not using this?)')
    
    ax.set_xlabel(r'redshift')
    ax.set_ylabel(r'Number of quasars')

    plt.ylim(7e-1, 1.0e4)
    plt.xlim(0.7, 7.0)

    plt.legend(loc='upper right', fontsize=14, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.4, borderpad=0.2)

    plt.title('Quasars not included in analysis')
    
    plt.savefig('qsos_NotIncluded.pdf', bbox_inches='tight')

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


