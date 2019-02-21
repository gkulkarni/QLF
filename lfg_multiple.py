import sys
import numpy as np 
from composite import lf
from composite import lf_polyb
from summary_fromFile import summary_plot as sp

# Model 1 

qlumfiles = ['Data_new/dr7z2p2_sample.dat',
             'Data_new/croom09sgp_sample.dat',
             'Data_new/croom09ngp_sample.dat',
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
             'Data_new/giallongo15_sample.dat',
             'Data_new/ukidss_sample.dat',
             'Data_new/banados_sample.dat']

selnfiles = [('Selmaps_with_tiles/dr7z2p2_selfunc.dat', 6248.0, 13),
             ('Selmaps_with_tiles/croom09sgp_selfunc.dat', 64.2, 15),
             ('Selmaps_with_tiles/croom09ngp_selfunc.dat', 127.7, 15),
             ('Selmaps_with_tiles/dr7z3p7_selfunc.dat', 6248.0, 13),
             ('Selmaps_with_tiles/glikman11_selfunc_ndwfs.dat', 1.71, 6),
             ('Selmaps_with_tiles/glikman11_selfunc_dls.dat', 2.05, 6),
             ('Selmaps_with_tiles/yang16_sel.dat', 14555.0, 17),
             ('Selmaps_with_tiles/mcgreer13_dr7selfunc.dat', 6248.0, 8),
             ('Selmaps_with_tiles/mcgreer13_s82selfunc.dat', 235.0, 8),
             ('Selmaps_with_tiles/jiang16main_selfunc.dat', 11240.0, 18),
             ('Selmaps_with_tiles/jiang16overlap_selfunc.dat', 4223.0, 18),
             ('Selmaps_with_tiles/jiang16s82_selfunc.dat', 277.0, 18),
             ('Selmaps_with_tiles/willott10_cfhqsdeepsel.dat', 4.47, 10),
             ('Selmaps_with_tiles/willott10_cfhqsvwsel.dat', 494.0, 10),
             ('Selmaps_with_tiles/kashikawa15_sel.dat', 6.5, 11),
             ('Selmaps_with_tiles/giallongo15_sel.dat', 0.047, 7),
             ('Selmaps_with_tiles/ukidss_sel_4.dat', 3370.0, 19),
             ('Selmaps_with_tiles/banados_sel_4.dat', 2500.0, 20)]

lfg1 = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,5])

g = np.array([-7.95061036, 1.15284665, -0.12037541,
              -18.64592897, -4.52638114, 0.47207865, -0.01890026,
              -3.35945526, -0.26211017,
              -2.47899576, 0.978408, 3.76233908, 10.96715636, -0.33557835])

method = 'Nelder-Mead'
b = lfg1.bestfit(g, method=method)

lfg1.prior_min_values = np.array([-15.0, 0.0, -5.0,
                                 -30.0, -10.0, 0.0, -2.0,
                                 -7.0, -5.0,
                                 -10.0, -10.0, 0.0, -10.0, -2.0])

lfg1.prior_max_values = np.array([-5.0, 10.0, 5.0,
                                 -10.0, -1.0, 2.0, 2.0,
                                 -1.0, 5.0,
                                 10.0, 10.0, 10.0, 200.0, 2.0])

assert(np.all(lfg1.prior_min_values < lfg1.prior_max_values))
assert(np.all(lfg1.bf.x < lfg1.prior_max_values))
assert(np.all(lfg1.prior_min_values < lfg1.bf.x))

lfg1.run_mcmc()

#------------------------------------------------------------

# Model 2 

qlumfiles = ['Data_new/dr7z2p2_sample.dat',
             'Data_new/croom09sgp_sample.dat',
             'Data_new/croom09ngp_sample.dat',
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
             'Data_new/kashikawa15_sample.dat']

selnfiles = [('Selmaps_with_tiles/dr7z2p2_selfunc.dat', 6248.0, 13),
             ('Selmaps_with_tiles/croom09sgp_selfunc.dat', 64.2, 15),
             ('Selmaps_with_tiles/croom09ngp_selfunc.dat', 127.7, 15),
             ('Selmaps_with_tiles/dr7z3p7_selfunc.dat', 6248.0, 13),
             ('Selmaps_with_tiles/glikman11_selfunc_ndwfs.dat', 1.71, 6),
             ('Selmaps_with_tiles/glikman11_selfunc_dls.dat', 2.05, 6),
             ('Selmaps_with_tiles/yang16_sel.dat', 14555.0, 17),
             ('Selmaps_with_tiles/mcgreer13_dr7selfunc.dat', 6248.0, 8),
             ('Selmaps_with_tiles/mcgreer13_s82selfunc.dat', 235.0, 8),
             ('Selmaps_with_tiles/jiang16main_selfunc.dat', 11240.0, 18),
             ('Selmaps_with_tiles/jiang16overlap_selfunc.dat', 4223.0, 18),
             ('Selmaps_with_tiles/jiang16s82_selfunc.dat', 277.0, 18),
             ('Selmaps_with_tiles/willott10_cfhqsdeepsel.dat', 4.47, 10),
             ('Selmaps_with_tiles/willott10_cfhqsvwsel.dat', 494.0, 10),
             ('Selmaps_with_tiles/kashikawa15_sel.dat', 6.5, 11)]

lfg2 = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,5])

g = np.array([-7.95061036, 1.15284665, -0.12037541,
              -18.64592897, -4.52638114, 0.47207865, -0.01890026,
              -3.35945526, -0.26211017,
              -2.47899576, 0.978408, 3.76233908, 10.96715636, -0.33557835])

method = 'Nelder-Mead'
b = lfg2.bestfit(g, method=method)

lfg2.prior_min_values = np.array([-15.0, 0.0, -5.0,
                                 -30.0, -10.0, 0.0, -2.0,
                                 -7.0, -5.0,
                                 -10.0, -10.0, 0.0, -10.0, -2.0])

lfg2.prior_max_values = np.array([-5.0, 10.0, 5.0,
                                 -10.0, -1.0, 2.0, 2.0,
                                 -1.0, 5.0,
                                 10.0, 10.0, 10.0, 200.0, 2.0])

assert(np.all(lfg2.prior_min_values < lfg2.prior_max_values))
assert(np.all(lfg2.bf.x < lfg2.prior_max_values))
assert(np.all(lfg2.prior_min_values < lfg2.bf.x))

lfg2.run_mcmc()

#------------------------------------------------------------

# Model 3

qlumfiles = ['Data_new/dr7z2p2_sample.dat',
             'Data_new/croom09sgp_sample.dat',
             'Data_new/croom09ngp_sample.dat',
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
             'Data_new/kashikawa15_sample.dat']

selnfiles = [('Selmaps_with_tiles/dr7z2p2_selfunc.dat', 6248.0, 13),
             ('Selmaps_with_tiles/croom09sgp_selfunc.dat', 64.2, 15),
             ('Selmaps_with_tiles/croom09ngp_selfunc.dat', 127.7, 15),
             ('Selmaps_with_tiles/dr7z3p7_selfunc.dat', 6248.0, 13),
             ('Selmaps_with_tiles/glikman11_selfunc_ndwfs.dat', 1.71, 6),
             ('Selmaps_with_tiles/glikman11_selfunc_dls.dat', 2.05, 6),
             ('Selmaps_with_tiles/yang16_sel.dat', 14555.0, 17),
             ('Selmaps_with_tiles/mcgreer13_dr7selfunc.dat', 6248.0, 8),
             ('Selmaps_with_tiles/mcgreer13_s82selfunc.dat', 235.0, 8),
             ('Selmaps_with_tiles/jiang16main_selfunc.dat', 11240.0, 18),
             ('Selmaps_with_tiles/jiang16overlap_selfunc.dat', 4223.0, 18),
             ('Selmaps_with_tiles/jiang16s82_selfunc.dat', 277.0, 18),
             ('Selmaps_with_tiles/willott10_cfhqsdeepsel.dat', 4.47, 10),
             ('Selmaps_with_tiles/willott10_cfhqsvwsel.dat', 494.0, 10),
             ('Selmaps_with_tiles/kashikawa15_sel.dat', 6.5, 11)]

lfg3 = lf_polyb(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,2])

g = np.array([-7.95061036, 1.15284665, -0.12037541,
              -18.64592897, -4.52638114, 0.47207865, -0.01890026,
              -3.35945526, -0.26211017,
              -1.30352181, -0.15925648])

method = 'Nelder-Mead'
b = lfg3.bestfit(g, method=method)

lfg3.prior_min_values = np.array([-15.0, 0.0, -5.0,
                                 -30.0, -10.0, 0.0, -2.0,
                                 -7.0, -5.0,
                                 -5.0, -5.0])

lfg3.prior_max_values = np.array([-5.0, 10.0, 5.0,
                                 -10.0, -1.0, 2.0, 2.0,
                                 -1.0, 5.0,
                                 0.0, 5.0])

assert(np.all(lfg3.prior_min_values < lfg3.prior_max_values))
assert(np.all(lfg3.bf.x < lfg3.prior_max_values))
assert(np.all(lfg3.prior_min_values < lfg3.bf.x))

lfg3.run_mcmc()

#------------------------------------------------------------

import bins 
