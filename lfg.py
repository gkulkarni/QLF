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

lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[4,4,4,4])

g = [-6.7, -1.8, 0.2, 0.0,
    -26.0, -1.8, 0.2, 0.0,
     -4.0, -0.5, 0.0, 0.0, 
     -1.3, -0.8, 0.0, 0.0]

g = [ -7.2251704 ,  -1.62984082,  -0.89243546,   0.39197742,
       -23.77361222,  -5.70066143,   1.84795056,  -0.72731527,
        -3.5079198 ,   0.80786039,   0.11189202,   1.075076  ,
        -0.9733357 ,  -3.14308065,   0.38280798,  -1.02040873]

g = [ -7.18595306,  -1.45689727,  -0.87085965,   0.50649104,
       -23.64216997,  -5.17943497,   2.05295088,  -0.3284502 ,
        -3.41406232,   0.5423091 ,   0.18818908,   0.97221802,
        -0.87767209,  -2.33164279,   0.69931523,  -0.34634807]

g = [ -7.83388608e+00, -2.17215699e+00, -1.40343551e+00, 6.53577205e-01,
      -2.44611061e+01, -6.11174021e+00, 1.39999445e+00, 5.46499981e-03,
      -1.58947053e+01, -2.31522736e+01, -1.15716990e+01, -4.22210346e+00,
      -6.41143451e-01, -2.67085666e+00, 1.01459750e+00, -2.06489421e-01]

method = 'Nelder-Mead'
b = lfg.bestfit(g, method=method)
print b

lfg.prior_min_values = np.array([-13.0, -10.0, -5.0, -5.0,
                                 -35.0, -20.0, -5.0, -7.0,
                                 -50.0, -100.0, -25.0, -10.0,
                                 -5.0, -10.0, -5.0, -7.0])

lfg.prior_max_values = np.array([-2.0, 10.0, 5.0, 5.0,
                                 -15.0, 10.0, 7.0, 5.0,
                                 50.0, 100.0, 25.0, 10.0,
                                 5.0, 10.0, 5.0, 5.0])

assert(np.all(lfg.prior_min_values < lfg.prior_max_values))

lfg.run_mcmc()

labels = [r'$a_0 [\phi_*]$', r'$a_1 [\phi_*]$', r'$a_2 [\phi_*]$',  r'$a_3 [\phi_*]$',
          r'$a_0 [M_*]$', r'$a_1 [M_*]$', r'$a_2 [M_*]$', r'$a_3 [M_*]$',
          r'$a_0 [\alpha]$', r'$a_1 [\alpha]$', r'$a_2 [\alpha]$', r'$a_3 [\alpha]$',
          r'$a_0 [\beta]$', r'$a_1 [\beta]$', r'$a_2 [\beta]$', r'$a_3 [\beta]$']

lfg.corner_plot(labels=labels)
lfg.chains(labels=labels)
sp(composite=lfg, sample=True)
