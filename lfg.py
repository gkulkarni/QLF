import sys
import numpy as np 
from composite import lf
from summary_fromFile import summary_plot as sp

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

lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,2,3,4])

# g = np.array([-7.73388053, 1.06477161, -0.11304974,
#               -22.75923587, -0.96452704,
#               -3.22779072, -0.27456505,
#               -1.53566144, 0.02274886, -0.0125998]) 

g = np.array([-8.4312489, 1.48352466, -0.14255848, -23.07419422,  -0.91502736, -3.79904767,  0.0485053,  -0.02002263, -2.85724627,  1.03598097, -0.13644424,  0.00469155])

method = 'Nelder-Mead'
b = lfg.bestfit(g, method=method)
print b

lfg.prior_min_values = np.array([-11.0, -5.0, -5.0,
                                 -30.0, -10.0,
                                 -5.0, -5.0, -5.0, 
                                 -5.0, -2.0, -2.0, -2.0])
lfg.prior_max_values = np.array([-4.0, 5.0, 2.0,
                                 -10.0, 5.0,
                                 -1.0, 5.0, 5.0, 
                                 5.0, 2.0, 2.0, 2.0])
assert(np.all(lfg.prior_min_values < lfg.prior_max_values))

# # Set bf.x. to different value to start MCMC from this value.  This is
# # the best fit from Chebyshev French curve fit with sigma.
# lfg.bf.x = np.array([-8.4312489, 1.48352466, -0.14255848,
#             -23.07419422, -0.91502736,
#             -3.43699074, -0.19373464,
#             -1.83439393,  0.19554403, -0.02286045])

lfg.run_mcmc()

labels = [r'$a_0 [\phi_*]$', r'$a_1 [\phi_*]$', r'$a_2 [\phi_*]$',
          r'$a_0 [M_*]$', r'$a_1 [M_*]$',
          r'$a_0 [\alpha]$', r'$a_1 [\alpha]$',
          r'$a_0 [\beta]$', r'$a_1 [\beta]$', r'$a_2 [\beta]$']

# Restore bf.x value for plotting 
# lfg.bf.x = b.x 

lfg.corner_plot(labels=labels)
lfg.chains(labels=labels)
sp(composite=lfg, sample=True)
