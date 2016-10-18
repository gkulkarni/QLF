from composite import lf
import sys
import time
from summary import summary_plot as sp
from gammapi import plot_gamma as pg 
import numpy as np

# Defaults 
case = 3
method = 'Powell'

case = int(sys.argv[1])
method = sys.argv[2]

if case == 1:
    
    qlumfiles = ['Data/bossdr9color.dat']

    selnfiles = [('Data/ross13_selfunc2.dat',2236.0)]

elif case == 2:

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/dr7z3p7.dat']

    selnfiles = [('Data/ross13_selfunc2.dat',2236.0),
                 ('Data/sdss_selfunc2.dat',6248.0)]

elif case == 3: 

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/dr7z3p7.dat',
                 'Data/glikman11qso.dat']
    
    selnfiles = [('Data/ross13_selfunc2.dat',2236.0),
                 ('Data/sdss_selfunc2.dat',6248.0),
                 ('Data/glikman11_selfunc_dls.dat',1.71),
                 ('Data/glikman11_selfunc_ndwfs.dat',2.05)]

elif case == 4: 

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/dr7z3p7.dat',
                 'Data/glikman11qso.dat',
                 'Data/mcgreer13_s82sample.dat',
                 'Data/mcgreer13_dr7sample.dat',
                 'Data/mcgreer13_s82extend.dat']
    
    selnfiles = [('Data/ross13_selfunc2.dat',2236.0),
                 ('Data/sdss_selfunchigh.dat',6248.0),
                 ('Data/glikman11_selfunc_dls.dat',1.71),
                 ('Data/glikman11_selfunc_ndwfs.dat',2.05),
                 ('Data/mcgreer13_s82selfunc.dat',235.0),
                 ('Data/mcgreer13_dr7selfunc.dat',6248.0)]

elif case == 5: 

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/dr7z3p7.dat',
                 'Data/glikman11qso.dat',
                 'Data/mcgreer13_s82sample.dat',
                 'Data/mcgreer13_dr7sample.dat',
                 'Data/mcgreer13_s82extend.dat',
                 'Data/fan06_sample.dat',
                 'Data/jiang08_sample.dat',
                 'Data/jiang09_sample.dat',
                 'Data/willott10_cfhqsdeepsample.dat',
                 'Data/willott10_cfhqsvwsample.dat',
                 'Data/kashikawa15_sample.dat']
    
    selnfiles = [('Data/ross13_selfunc2.dat',2236.0),
                 ('Data/sdss_selfunchigh.dat',6248.0),
                 ('Data/glikman11_selfunc_dls.dat',1.71),
                 ('Data/glikman11_selfunc_ndwfs.dat',2.05),
                 ('Data/mcgreer13_s82selfunc.dat',235.0),
                 ('Data/mcgreer13_dr7selfunc.dat',6248.0),
                 ('Data/fan06_sel.dat',6600.0),
                 ('Data/jiang08_sel.dat',260.0),
                 ('Data/jiang09_sel.dat',195.0),
                 ('Data/willott10_cfhqsvwsel.dat',494.0),
                 ('Data/willott10_cfhqsdeepsel.dat',4.47),
                 ('Data/kashikawa15_sel.dat',6.50)]

elif case == 6: 

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/dr7z3p7.dat',
                 'Data/glikman11qso.dat',
                 'Data/mcgreer13_s82sample.dat',
                 'Data/mcgreer13_dr7sample.dat',
                 'Data/mcgreer13_s82extend.dat',
                 'Data/fan06_sample.dat',
                 'Data/jiang08_sample.dat',
                 'Data/jiang09_sample.dat',
                 'Data/willott10_cfhqsdeepsample.dat',
                 'Data/willott10_cfhqsvwsample.dat',
                 'Data/kashikawa15_sample.dat',
                 'Data/dr7z2p2.dat']
    
    selnfiles = [('Data/ross13_selfunc2.dat',2236.0),
                 ('Data/sdss_selfunchigh.dat',6248.0),
                 ('Data/glikman11_selfunc_dls.dat',1.71),
                 ('Data/glikman11_selfunc_ndwfs.dat',2.05),
                 ('Data/mcgreer13_s82selfunc.dat',235.0),
                 ('Data/mcgreer13_dr7selfunc.dat',6248.0),
                 ('Data/fan06_sel.dat',6600.0),
                 ('Data/jiang08_sel.dat',260.0),
                 ('Data/jiang09_sel.dat',195.0),
                 ('Data/willott10_cfhqsvwsel.dat',494.0),
                 ('Data/willott10_cfhqsdeepsel.dat',4.47),
                 ('Data/kashikawa15_sel.dat',6.50),
                 ('Data/sdss_selfunclow.dat',6248.0)]

elif case == 7:

    # All high-z (z >= 2.2) quasars; includes Giallongo 

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/dr7z3p7.dat',
                 'Data/glikman11qso.dat',
                 'Data/mcgreer13_s82sample.dat',
                 'Data/mcgreer13_dr7sample.dat',
                 'Data/mcgreer13_s82extend.dat',
                 'Data/fan06_sample.dat',
                 'Data/jiang08_sample.dat',
                 'Data/jiang09_sample.dat',
                 'Data/willott10_cfhqsdeepsample.dat',
                 'Data/willott10_cfhqsvwsample.dat',
                 'Data/kashikawa15_sample.dat',
                 'Data/giallongo15_sample.dat']
    
    selnfiles = [('Data/ross13_selfunc2.dat',2236.0),
                 ('Data/sdss_selfunchigh.dat',6248.0),
                 ('Data/glikman11_selfunc_dls.dat',1.71),
                 ('Data/glikman11_selfunc_ndwfs.dat',2.05),
                 ('Data/mcgreer13_s82selfunc.dat',235.0),
                 ('Data/mcgreer13_dr7selfunc.dat',6248.0),
                 ('Data/fan06_sel.dat',6600.0),
                 ('Data/jiang08_sel.dat',260.0),
                 ('Data/jiang09_sel.dat',195.0),
                 ('Data/willott10_cfhqsvwsel.dat',494.0),
                 ('Data/willott10_cfhqsdeepsel.dat',4.47),
                 ('Data/kashikawa15_sel.dat',6.50),
                 ('Data/giallongo15_sel.dat',0.047)]
    
    
lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,3,2,3])

# g = [-6.7, -1.8, 0.2, -26.0, -1.8, 0.2, -5.0, -1.8, -0.2, -1.6, -1.8, 0.6]
g = [-6.7, -1.8, 0.2, -26.0, -1.8, 0.2, -5.0, -1.8, -1.6, -1.8, 0.6]

b = lfg.bestfit(g, method=method)
print b

# Uncomment if you want automatic priors 
# lfg.create_param_range()

# lfg.prior_min_values = np.array([-8.0, -3.0, -2.0, -30.0, -3.0, -1.0, -5.0, -3.0, -1.0, -3.0, -1.0, -1.0])
# lfg.prior_max_values = np.array([-5.0, -0.1,  0.4, -20.0,  1.0,  2.0, -1.0,  3.0,  3.0, -0.1,  1.0,  1.5])

lfg.prior_min_values = np.array([-8.0, -3.0, -2.0, -30.0, -3.0, -1.0, -5.0, -3.0, -3.0, -1.0, -1.0])
lfg.prior_max_values = np.array([-5.0, -0.1,  0.4, -20.0,  1.0,  2.0, -1.0,  0.0, -0.1,  1.0,  1.5])

assert(np.all(lfg.prior_min_values < lfg.prior_max_values))

lfg.run_mcmc()

labels = [r'$a_0 [\phi_*]$', r'$a_1 [\phi_*]$', r'$a_2 [\phi_*]$', r'$a_0 [M_*]$', r'$a_1 [M_*]$', r'$a_2 [M_*]$', r'$a_0 [\alpha]$', r'$a_1 [\alpha]$', r'$a_2 [\alpha]$', r'$a_0 [\beta]$', r'$a_1 [\beta]$', r'$a_2 [\beta]$']

lfg.corner_plot(labels=labels)
lfg.chains(labels=labels)

sys.exit() 


# Uncomment if you want full individual calculation 
import bins
sp(lfg, individuals=bins.lfs, individuals_isfile=False)
pg(lfg, individuals=bins.lfs)

from waic_composite import waic
print 'WIAC=', waic(lfg)


