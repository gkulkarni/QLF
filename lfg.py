import sys
import numpy as np 
from composite import lf
from summary_fromFile import summary_plot as sp

qlumfiles = ['Data_new/dr7z2p2_sample.dat',
             'Data_new/croom09sgp_sample.dat',
             'Data_new/croom09ngp_sample.dat',
             #'Data_new/bossdr9color.dat',
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
             #'Data_new/giallongo15_sample.dat',
             'Data_new/ukidss_sample.dat',
             'Data_new/banados_sample.dat']

selnfiles = [('Data_new/dr7z2p2_selfunc.dat', 0.1, 0.05, 6248.0, 13),
             ('Data_new/croom09sgp_selfunc.dat', 0.3, 0.05, 64.2, 15),
             ('Data_new/croom09ngp_selfunc.dat', 0.3, 0.05, 127.7, 15),
             #('Data_new/ross13_selfunc2.dat', 0.1, 0.05, 2236.0, 1),
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
             #('Data_new/giallongo15_sel.dat', 0.0, 0.0, 0.047, 7)
             ('Data_new/ukidss_sel.dat', 0.8, 0.45, 3370.0, 19),
             ('Data_new/banados_sel.dat', 1.0, 1.6, 2500.0, 20)]

case = 0

if case == 0:

    # Currently favoured model
    
    lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,5])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -18.64592897, -4.52638114, 0.47207865, -0.01890026,
                  -3.35945526, -0.26211017,
                  -2.47899576, 0.978408, 3.76233908, 10.96715636, -0.33557835])

    lfg.prior_min_values = np.array([-15.0, 0.0, -5.0, -30.0, -10.0,
                                     0.0, -2.0, -7.0, -5.0, -5.0, 0.0,
                                     1.0, 1.0, -2.0])
    lfg.prior_max_values = np.array([-5.0, 10.0, 5.0, -10.0, -1.0,
                                     2.0, 2.0, -1.0, 5.0, 0.0, 5.0,
                                     5.0, 200.0, 2.0])

    assert(np.all(lfg.prior_min_values < lfg.prior_max_values))

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b

    import bins

    # lfg.run_mcmc()

    # labels = 14*['a']

    # lfg.corner_plot(labels=labels)
    # lfg.chains(labels=labels)

    # import bins

    # sp(composite=lfg, individuals=bins.lfs, sample=True)
    
elif case == 1:

    lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,2])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -18.64592897, -4.52638114, 0.47207865, -0.01890026,
                  -3.35945526, -0.26211017,
                  -1.30352181, -0.15925648])

    lfg.prior_min_values = np.array([-15.0, 0.0, -5.0, -30.0, -10.0,
                                     0.0, -2.0, -7.0, -5.0, -5.0, -5.0])
    lfg.prior_max_values = np.array([-5.0, 10.0, 5.0, -10.0, -1.0,
                                     2.0, 2.0, -1.0, 5.0, 0.0, 5.0])

    assert(np.all(lfg.prior_min_values < lfg.prior_max_values))

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b

    lfg.run_mcmc()

    labels = 14*['a']

    lfg.corner_plot(labels=labels)
    lfg.chains(labels=labels)
    sp(composite=lfg, sample=True)
    
elif case == 2:

    lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,5])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -18.64592897, -4.52638114, 0.47207865, -0.01890026,
                  -3.35945526, -0.26211017,
                  -2.53705144, 0.65781084, 4.41364161, 12.5716938, 0.26329899])

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b
    
    lfg.prior_min_values = np.array([-15.0, 0.0, -5.0, -30.0, -10.0,
                                     0.0, -2.0, -7.0, -5.0, -5.0, 0.0,
                                     1.0, 1.0, -2.0])
    lfg.prior_max_values = np.array([-5.0, 10.0, 5.0, -10.0, -1.0,
                                     2.0, 2.0, -1.0, 5.0, 0.0, 5.0,
                                     5.0, 50.0, 2.0])

    assert(np.all(lfg.prior_min_values < lfg.prior_max_values))
    lfg.run_mcmc()
    labels = 14*['a']

    lfg.corner_plot(labels=labels)
    lfg.chains(labels=labels)
    sp(composite=lfg, sample=True)

elif case == 3:

    lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,4])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -18.64592897, -4.52638114, 0.47207865, -0.01890026,
                  -3.35945526, -0.26211017,
                  -2.19480099,  0.46906026, -0.07710908,  0.00297377])

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b

    lfg.prior_min_values = np.array([-15.0, 0.0, -5.0, -30.0, -10.0,
                                     0.0, -2.0, -7.0, -5.0, -5.0, 0.0,
                                     1.0, 1.0, -2.0])
    lfg.prior_max_values = np.array([-5.0, 10.0, 5.0, -10.0, -1.0,
                                     2.0, 2.0, -1.0, 5.0, 0.0, 5.0,
                                     5.0, 15.0, 2.0])

    assert(np.all(lfg.prior_min_values < lfg.prior_max_values))

    # lfg.run_mcmc()

    # labels = 14*['a']

    # lfg.corner_plot(labels=labels)
    # lfg.chains(labels=labels)

    # import bins

    # sp(composite=lfg, individuals=bins.lfs, sample=True)
    
elif case == 4:

    lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,5])

    g = np.array([-14.91608053,   9.56315388,  -6.79813429,
                  -8.84766506, -29.42149341,  11.77064009,  -3.14985466,
                  -3.32340877, -1.86158462,
                  -2.47899576, 0.978408, 3.76233908, 10.96715636, -0.33557835])

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b
    
