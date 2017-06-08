import sys
import numpy as np 
from composite import lf
from summary import summary_plot as sp

qlumfiles = ['Data_new/dr7z2p2_sample.dat',
             'Data_new/croom09sgp_sample.dat',
             'Data_new/croom09ngp_sample.dat',
             'Data_new/bossdr9color.dat',
             'Data_new/dr7z3p7_sample.dat',
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

selnfiles = [('Data_new/dr7z2p2_selfunc.dat', 0.1, 0.05, 6248.0, 13),
             ('Data_new/croom09sgp_selfunc.dat', 0.3, 0.05, 64.2, 15),
             ('Data_new/croom09ngp_selfunc.dat', 0.3, 0.05, 127.7, 15),
             ('Data_new/ross13_selfunc2.dat', 0.1, 0.05, 2236.0, 1),
             ('Data_new/dr7z3p7_selfunc.dat', 0.1, 0.05, 6248.0, 13),
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

lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,5])

g = np.array([-7.73388053, 1.06477161, -0.11304974,
              -1.96740317e+01,  -3.60163185e+00,   3.45950105e-01,  -1.36387048e-02,
              -3.22779072, -0.27456505,
              -2.52870151, 1.08701021, 3.48991378, 6.20831541, -0.21475542])

# g = np.array([-7.73388053, 1.06477161, -0.11304974,
#               -45.54987107,  66.51882126, -37.01499714, -18.47003438,
#               -3.22779072, -0.27456505,
#               -2.52870151, 1.08701021, 3.48991378, 6.20831541, -0.21475542])

method = 'Nelder-Mead'
b = lfg.bestfit(g, method=method)
print b

sys.exit()

lfg.prior_min_values = np.array([-15.0, 0.0, -5.0, -30.0, -10.0, 0.0, -2.0, -7.0, -5.0, -5.0, 0.0, 1.0, 1.0, -2.0])
lfg.prior_max_values = np.array([-5.0, 10.0, 5.0, -10.0, -1.0, 2.0, 2.0, -1.0, 5.0, 0.0, 5.0, 5.0, 10.0, 2.0])
assert(np.all(lfg.prior_min_values < lfg.prior_max_values))

lfg.run_mcmc()

# labels = [r'$a_0 [\phi_*]$', r'$a_1 [\phi_*]$', r'$a_2 [\phi_*]$',
#           r'$a_0 [M_*]$', r'$a_1 [M_*]$',
#           r'$a_0 [\alpha]$', r'$a_1 [\alpha]$',
#           r'$a_0 [\beta]$', r'$a_1 [\beta]$', r'$a_2 [\beta]$']

labels = 14*['a']

lfg.corner_plot(labels=labels)
lfg.chains(labels=labels)
sp(composite=lfg, sample=True)
