import sys
import numpy as np 
from composite import lf
# from summary_fromFile import summary_plot as sp
from summary_cfit_highz import summary_plot as sp
import drawlf

# qlumfiles = ['Data_new/dr7z2p2_sample.dat',
#              'Data_new/croom09sgp_sample.dat',
#              'Data_new/croom09ngp_sample.dat',
#              'Data_new/bossdr9color.dat',
#              'Data_new/dr7z3p7_sample.dat',
#              'Data_new/glikman11debug.dat',
#              'Data_new/yang16_sample.dat',
#              'Data_new/mcgreer13_dr7sample.dat',
#              'Data_new/mcgreer13_s82sample.dat',
#              'Data_new/mcgreer13_dr7extend.dat',
#              'Data_new/mcgreer13_s82extend.dat',
#              'Data_new/jiang16main_sample.dat',
#              'Data_new/jiang16overlap_sample.dat',
#              'Data_new/jiang16s82_sample.dat',
#              'Data_new/willott10_cfhqsdeepsample.dat',
#              'Data_new/willott10_cfhqsvwsample.dat',
#              'Data_new/kashikawa15_sample.dat',
#              'Data_new/giallongo15_sample.dat']

# selnfiles = [('Data_new/dr7z2p2_selfunc.dat', 0.1, 0.05, 6248.0, 13),
#              ('Data_new/croom09sgp_selfunc.dat', 0.3, 0.05, 64.2, 15),
#              ('Data_new/croom09ngp_selfunc.dat', 0.3, 0.05, 127.7, 15),
#              ('Data_new/ross13_selfunc2.dat', 0.1, 0.05, 2236.0, 1),
#              ('Data_new/dr7z3p7_selfunc.dat', 0.1, 0.05, 6248.0, 13),
#              ('Data_new/glikman11_selfunc_ndwfs.dat', 0.05, 0.02, 1.71, 6),
#              ('Data_new/glikman11_selfunc_dls.dat', 0.05, 0.02, 2.05, 6),
#              ('Data_new/yang16_sel.dat', 0.1, 0.05, 14555.0, 17),
#              ('Data_new/mcgreer13_dr7selfunc.dat', 0.1, 0.05, 6248.0, 8),
#              ('Data_new/mcgreer13_s82selfunc.dat', 0.1, 0.05, 235.0, 8),
#              ('Data_new/jiang16main_selfunc.dat', 0.1, 0.05, 11240.0, 18),
#              ('Data_new/jiang16overlap_selfunc.dat', 0.1, 0.05, 4223.0, 18),
#              ('Data_new/jiang16s82_selfunc.dat', 0.1, 0.05, 277.0, 18),
#              ('Data_new/willott10_cfhqsdeepsel.dat', 0.1, 0.025, 4.47, 10),
#              ('Data_new/willott10_cfhqsvwsel.dat', 0.1, 0.025, 494.0, 10),
#              ('Data_new/kashikawa15_sel.dat', 0.05, 0.05, 6.5, 11),
#              ('Data_new/giallongo15_sel.dat', 0.0, 0.0, 0.047, 7)]

qlumfiles = ['Data_new/dr7z3p7_sample.dat',
             'Data_new/glikman11debug.dat',
             'Data_new/yang16_sample.dat',
             'Data_new/mcgreer13_dr7sample.dat',
             'Data_new/mcgreer13_s82sample.dat',
             'Data_new/mcgreer13_dr7extend.dat',
             'Data_new/mcgreer13_s82extend.dat',
             'Data_new/jiang16main_sample.dat',
             'Data_new/jiang16overlap_sample.dat',
             'Data_new/jiang16s82_sample.dat',
             'Data_new/willott10_cfhqsdeepsample.dat',
             'Data_new/willott10_cfhqsvwsample.dat',
             'Data_new/kashikawa15_sample.dat',
             'Data_new/giallongo15_sample.dat']

selnfiles = [('Data_new/dr7z3p7_selfunc.dat', 0.1, 0.05, 6248.0, 13),
             ('Data_new/glikman11_selfunc_ndwfs.dat', 0.05, 0.02, 1.71, 6),
             ('Data_new/glikman11_selfunc_dls.dat', 0.05, 0.02, 2.05, 6),
             ('Data_new/yang16_sel.dat', 0.1, 0.05, 14555.0, 17),
             ('Data_new/mcgreer13_dr7selfunc.dat', 0.1, 0.05, 6248.0, 8),
             ('Data_new/mcgreer13_s82selfunc.dat', 0.1, 0.05, 235.0, 8),
             ('Data_new/jiang16main_selfunc.dat', 0.1, 0.05, 11240.0, 18),
             ('Data_new/jiang16overlap_selfunc.dat', 0.1, 0.05, 4223.0, 18),
             ('Data_new/jiang16s82_selfunc.dat', 0.1, 0.05, 277.0, 18),
             ('Data_new/willott10_cfhqsdeepsel.dat', 0.1, 0.025, 4.47, 10),
             ('Data_new/willott10_cfhqsvwsel.dat', 0.1, 0.025, 494.0, 10),
             ('Data_new/kashikawa15_sel.dat', 0.05, 0.05, 6.5, 11),
             ('Data_new/giallongo15_sel.dat', 0.0, 0.0, 0.047, 7)]

lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[2,2,2,2])

# g = np.array([-7.73388053, 1.06477161, -0.11304974,
#               -22.75923587, -0.96452704,
#               -3.22779072, -0.27456505,
#               -1.53566144, 0.02274886, -0.0125998]) 
#
# g = np.array([-1.99941234, -1.23726835, -23.38962662, -0.79762924,
#               -3.3512842, -0.26610338, -1.51236172, -0.13870576])

g = np.array([-8.618798, -2.0414926, -27.65694305, -1.31608818,
              -4.77493728, -0.43907063, -2.25443756, -0.2288645])

method = 'Nelder-Mead'
b = lfg.bestfit(g, method=method)
print b

# sys.exit()

lfg.prior_min_values = np.array([-20.0, -100.0, -30.0, -200.0, -7.0, -20.0, -5.0, -5.0])
lfg.prior_max_values = np.array([20.0, 5.0, 10.0, 5.0, 7.0, 20.0, 5.0, 5.0])

# lfg.prior_min_values = np.array([-11.0, -5.0, -5.0, -30.0, -10.0, -5.0, -5.0, -5.0, -2.0, -2.0])
# lfg.prior_max_values = np.array([-4.0, 5.0, 2.0, -10.0, 5.0, -1.0, 5.0, 5.0, 2.0, 2.0])
assert(np.all(lfg.prior_min_values < lfg.prior_max_values))

lfg.run_mcmc()

labels = [r'$a_0 [\phi_*]$', r'$a_1 [\phi_*]$', r'$a_2 [\phi_*]$',
          r'$a_0 [M_*]$', r'$a_1 [M_*]$',
          r'$a_0 [\alpha]$', r'$a_1 [\alpha]$',
          r'$a_0 [\beta]$', r'$a_1 [\beta]$', r'$a_2 [\beta]$']

labels = [r'$a_0 [\phi_*]$', r'$a_1 [\phi_*]$', r'$a_0 [M_*]$', r'$a_1 [M_*]$', r'$a_0 [\alpha]$', r'$a_1 [\alpha]$', r'$a_0 [\beta]$', r'$a_1 [\beta]$']

lfg.corner_plot(labels=labels)
lfg.chains(labels=labels)

import bins
import bins_mockData
lfs_mockData = bins_mockData.bins(lfg)

sp(composite=lfg, sample=True, lfs=bins.lfs, lfsMock=lfs_mockData)

for x in bins.lfs:
    drawlf.draw(x, composite=lfg)
