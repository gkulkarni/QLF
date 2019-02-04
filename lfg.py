import sys
import numpy as np 
from composite import lf
from composite import lf_polyb
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
             'Data_new/kashikawa15_sample.dat']
             # 'Data_new/giallongo15_sample.dat',
             # 'Data_new/ukidss_sample.dat',
             # 'Data_new/banados_sample.dat']

selnfiles = [('Selmaps_with_tiles/dr7z2p2_selfunc.dat', 6248.0, 13),
             ('Selmaps_with_tiles/croom09sgp_selfunc.dat', 64.2, 15),
             ('Selmaps_with_tiles/croom09ngp_selfunc.dat', 127.7, 15),
             #('Selmaps_with_tiles/ross13_selfunc2.dat', 2236.0, 1),
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
             # ('Selmaps_with_tiles/giallongo15_sel.dat', 0.047, 7),
             # ('Selmaps_with_tiles/ukidss_sel_4.dat', 3370.0, 19),
             # ('Selmaps_with_tiles/banados_sel_4.dat', 2500.0, 20)]

case = 12

if case == 0:

    # Currently favoured model
    
    lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,5])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -18.64592897, -4.52638114, 0.47207865, -0.01890026,
                  -3.35945526, -0.26211017,
                  -2.47899576, 0.978408, 3.76233908, 10.96715636, -0.33557835])

    lfg.prior_min_values = np.array([-15.0, 0.0, -5.0, -30.0, -10.0, 0.0, -2.0, -7.0, -5.0, -10.0, -10.0, 0.0, -10.0, -2.0])
    lfg.prior_max_values = np.array([-5.0, 10.0, 5.0, -10.0, -1.0, 2.0, 2.0, -1.0, 5.0, 10.0, 10.0, 10.0, 200.0, 2.0])

    assert(np.all(lfg.prior_min_values < lfg.prior_max_values))

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b

    #import bins

    lfg.run_mcmc()

    # labels = 14*['a']

    # lfg.corner_plot(labels=labels)
    # lfg.chains(labels=labels)

    # import bins

    # sp(composite=lfg, individuals=bins.lfs, sample=True)
    
elif case == 1:

    lfg = lf_polyb(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,3,3])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -18.64592897, -4.52638114, 0.47207865, -0.01890026,
                  -3.24008528, -0.36906544,  0.01059615,
                  #-3.35945526, -0.26211017,
                  -1.75002369, -0.02546334, -0.00525714])

    # lfg.prior_min_values = np.array([-15.0, 0.0, -5.0, -30.0, -10.0,
    #                                  0.0, -2.0, -7.0, -5.0, -5.0, -5.0, -5.0])
    # lfg.prior_max_values = np.array([-5.0, 10.0, 5.0, -10.0, -1.0,
    #                                  2.0, 2.0, -1.0, 5.0, 0.0, 5.0, 1.0])

    lfg.prior_min_values = np.array([-15.0, 0.0, -5.0, -30.0, -10.0, 0.0, -2.0, -7.0, -5.0, -5.0, -5.0, -5.0, -5.0])
    lfg.prior_max_values = np.array([-5.0, 10.0, 5.0, -10.0, -1.0, 2.0, 2.0, -1.0, 5.0, 5.0, 0.0, 5.0, 1.0])
    
    assert(np.all(lfg.prior_min_values < lfg.prior_max_values))

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b

    lfg.run_mcmc()

    # labels = 14*['a']

    # lfg.corner_plot(labels=labels)
    # lfg.chains(labels=labels)
    # sp(composite=lfg, sample=True)
    
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

    lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,3])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -18.64592897, -4.52638114, 0.47207865, -0.01890026,
                  -3.35945526, -0.26211017,
                  -1.60670033, -0.02759287, -0.00685381])

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b

    lfg.prior_min_values = np.array([-10, 0, -5, -20, -10, 0, -2, -7, -5, -10, -10, -10])
    lfg.prior_max_values = np.array([-2, 10, 5, -10, -1, 2, 2, -1, 5, 10, 10, 10])
    assert(np.all(lfg.prior_min_values < lfg.prior_max_values))
    lfg.run_mcmc()

elif case == 5:

    lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,2])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -18.64592897, -4.52638114, 0.47207865, -0.01890026,
                  -3.35945526, -0.26211017,
                  -1.41863171, -0.13546455])

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b

elif case == 6:

    lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,3,2,5])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -8.03341756,  1.780554,   -0.18695025, 
                  -3.35945526, -0.26211017,
                  -2.47899576, 0.978408, 3.76233908, 10.96715636, -0.33557835])
    
    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b

elif case == 7:

    lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,3,3])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -18.64592897, -4.52638114, 0.47207865, -0.01890026,
                  -4.28592068, 1.13320416, -0.14003202, 
                  -1.60670033, -0.02759287, -0.00685381])

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b

elif case == 8:

    lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,3,3,3])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -22.58743676,  -1.20805348,   0.02333263,
                  -4.28592068, 1.13320416, -0.14003202, 
                  -1.60670033, -0.02759287, -0.00685381])

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b

elif case == 8:

    lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,2,3,3])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -23.26763262,  -0.81679979,
                  -4.28592068, 1.13320416, -0.14003202, 
                  -1.60670033, -0.02759287, -0.00685381])

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b
    
elif case == 9:

    lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,2,4])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -18.64592897, -4.52638114, 0.47207865, -0.01890026,
                  -3.35945526, -0.26211017,
                  -2.19480099,  0.46906026, -0.07710908,  0.00297377])

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)
    print b

    lfg.prior_min_values = np.array([-10, 0, -5, -20, -10, 0, -2, -7, -5, -10, -10, -10, -10])
    lfg.prior_max_values = np.array([-2, 10, 5, -10, -1, 2, 2, -1, 5, 10, 10, 10, 10])
    assert(np.all(lfg.prior_min_values < lfg.prior_max_values))
    lfg.run_mcmc()

elif case == 10:

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
    
elif case == 11:

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

    
elif case == 12:

    lfg = lf_polyb(quasar_files=qlumfiles, selection_maps=selnfiles, pnum=[3,4,4,3])

    g = np.array([-7.95061036, 1.15284665, -0.12037541,
                  -18.64592897, -4.52638114, 0.47207865, -0.01890026,
                  -0.55605685, -2.45096893,  0.26563797, -0.00963905,
                  -1.75002369, -0.02546334, -0.00525714])

    method = 'Nelder-Mead'
    b = lfg.bestfit(g, method=method)

    lfg.prior_min_values = np.array([-15.0, 0.0, -5.0,
                                     -30.0, -10.0, 0.0, -2.0,
                                      -10.0, -20.0, -10.0, -5.0,
                                      -5.0, -5.0, -5.0])

    lfg.prior_max_values = np.array([-5.0, 10.0, 5.0,
                                     -10.0, -1.0, 2.0, 2.0,
                                     10.0, 20.0, 10.0, 5.0, 
                                      5.0, 5.0, 5.0])

    assert(np.all(lfg.prior_min_values < lfg.prior_max_values))
    assert(np.all(lfg.bf.x < lfg.prior_max_values))
    assert(np.all(lfg.prior_min_values < lfg.bf.x))

    lfg.run_mcmc()

    

