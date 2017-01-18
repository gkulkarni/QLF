import sys
import numpy as np 
from composite import lf 

qlumfiles = ['Data_new/dr7z2p2_sample.dat',
             'Data_new/croom09sgp_sample.dat',
             'Data_new/croom09ngp_sample.dat',
             'Data_new/bossdr9color.dat',
             'Data_new/dr7z3p7_sample.dat',
             'Data_new/glikman11qso.dat',
             'Data_new/yang16_sample.dat',
             'Data_new/mcgreer13_dr7sample.dat',
             'Data_new/mcgreer13_s82sample.dat',
             'Data_new/mcgreer13_dr7extend.dat',
             'Data_new/mcgreer13_s82extend.dat',
             'Data_new/jiang08_sample.dat',
             'Data_new/jiang09_sample.dat',
             'Data_new/jiang16main_sample.dat',
             'Data_new/fan06_sample.dat',
             'Data_new/jiang16overlap_sample.dat',
             'Data_new/jiang16s82_sample.dat',
             'Data_new/willott10_cfhqsdeepsample.dat',
             'Data_new/willott10_cfhqsvwsample.dat',
             'Data_new/kashikawa15_sample.dat']

selnfiles = [('Data_new/dr7z2p2_selfunc.dat', 0.1, 0.05, 6248.0),
             ('Data_new/croom09sgp_selfunc.dat', 0.3, 0.05, 64.2),
             ('Data_new/croom09ngp_selfunc.dat', 0.3, 0.05, 127.7),
             ('Data_new/ross13_selfunc2.dat', 0.1, 0.05, 2236.0),
             ('Data_new/dr7z3p7_selfunc.dat', 0.1, 0.05, 6248.0),
             ('Data_new/glikman11_selfunc_ndwfs_old.dat', 0.05, 0.02, 1.71),
             ('Data_new/glikman11_selfunc_dls_old.dat', 0.05, 0.02, 2.05),
             ('Data_new/yang16_sel.dat', 0.1, 0.05, 14555.0),
             ('Data_new/mcgreer13_dr7selfunc.dat', 0.1, 0.05, 6248.0),
             ('Data_new/mcgreer13_s82selfunc.dat', 0.1, 0.05, 235.0),
             ('Data_new/jiang08_sel.dat', 0.1, 0.025, 260.0),
             ('Data_new/jiang09_sel.dat', 0.1, 0.025, 195.0),
             ('Data_new/jiang16main_selfunc.dat', 0.1, 0.05, 11240.0),
             ('Data_new/fan06_sel.dat', 0.1, 0.025, 6600.0),
             ('Data_new/jiang16overlap_selfunc.dat', 0.1, 0.05, 4223.0),
             ('Data_new/jiang16s82_selfunc.dat', 0.1, 0.05, 277.0),
             ('Data_new/willott10_cfhqsdeepsel.dat', 0.1, 0.025, 4.47),
             ('Data_new/willott10_cfhqsvwsel.dat', 0.1, 0.025, 494.0),
             ('Data_new/kashikawa15_sel.dat', 0.05, 0.05, 6.5)]

lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,3,2,3])

g = [-6.7, -1.8, 0.2, -26.0, -1.8, 0.2, -5.0, -1.8, -1.6, -1.8, 0.6]

method = 'Powell'
b = lfg.bestfit(g, method=method)
print b

lfg.prior_min_values = np.array([-7.0, 0.0, 0.1, -25.0, 0.5, 2.0, -5.0, 0.0, 0.0, 1.0, 0.2])
lfg.prior_max_values = np.array([-4.0, 0.4, 0.5, -15.0, 2.5, 8.0, -1.0, 2.0, 1.0, 3.0, 4.2])

assert(np.all(lfg.prior_min_values < lfg.prior_max_values))

lfg.run_mcmc()

labels = [r'$a_0 [\phi_*]$', r'$a_1 [\phi_*]$', r'$a_2 [\phi_*]$', r'$a_0 [M_*]$',
          r'$a_1 [M_*]$', r'$a_2 [M_*]$', r'$a_0 [\alpha]$', r'$a_1 [\alpha]$',
          r'$a_2 [\alpha]$', r'$a_0 [\beta]$', r'$a_1 [\beta]$', r'$a_2 [\beta]$']

lfg.corner_plot(labels=labels)
lfg.chains(labels=labels)
