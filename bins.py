import sys 
import numpy as np
from individual import lf
import mosaic

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
             'Data_new/kashikawa15_sample.dat']
             #'Data_new/giallongo15_sample.dat']

selnfiles = [('Data_new/dr7z2p2_selfunc.dat', 0.1, 0.05, 6248.0, 13, r'SDSS DR7 Richards et al.\ 2006'),
             ('Data_new/croom09sgp_selfunc.dat', 0.3, 0.05, 64.2, 15, r'2SLAQ Croom et al.\ 2009'),
             ('Data_new/croom09ngp_selfunc.dat', 0.3, 0.05, 127.7, 15, r'2SLAQ Croom et al.\ 2009'),
             ('Data_new/ross13_selfunc2.dat', 0.1, 0.05, 2236.0, 1, r'BOSS DR9 Ross et al.\ 2013'),
             ('Data_new/dr7z3p7_selfunc.dat', 0.1, 0.05, 6248.0, 13, r'SDSS DR7 Richards et al.\ 2006'),
             ('Data_new/glikman11_selfunc_ndwfs.dat', 0.05, 0.02, 1.71, 6, r'NDWFS+DLS Glikman et al.\ 2011'),
             ('Data_new/glikman11_selfunc_dls.dat', 0.05, 0.02, 2.05, 6, r'NDWFS+DLS Glikman et al.\ 2011'),
             ('Data_new/yang16_sel.dat', 0.1, 0.05, 14555.0, 17, r'SDSS+WISE Yang et al.\ 2016'),
             ('Data_new/mcgreer13_dr7selfunc.dat', 0.1, 0.05, 6248.0, 8, r'SDSS+S82 McGreer et al.\ 2013'),
             ('Data_new/mcgreer13_s82selfunc.dat', 0.1, 0.05, 235.0, 8, r'SDSS+S82 McGreer et al.\ 2013'),
             ('Data_new/jiang16main_selfunc.dat', 0.1, 0.05, 11240.0, 18, r'SDSS Jiang et al.\ 2016'),
             ('Data_new/jiang16overlap_selfunc.dat', 0.1, 0.05, 4223.0, 18, r'SDSS Jiang et al.\ 2016'),
             ('Data_new/jiang16s82_selfunc.dat', 0.1, 0.05, 277.0, 18, r'SDSS Jiang et al.\ 2016'),
             ('Data_new/willott10_cfhqsdeepsel.dat', 0.1, 0.025, 4.47, 10, r'CFHQS Willott et al.\ 2010'),
             ('Data_new/willott10_cfhqsvwsel.dat', 0.1, 0.025, 494.0, 10, r'CFHQS Willott et al.\ 2010'),
             ('Data_new/kashikawa15_sel.dat', 0.05, 0.05, 6.5, 11, r'Subaru Kashikawa et al.\ 2015')]
             #('Data_new/giallongo15_sel.dat', 0.0, 0.0, 0.047, 7, 'Giallongo et al.\ 2015')]

method = 'Nelder-Mead'

zls = [(0.1, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0),
       (1.0, 1.2), (1.2, 1.4), (1.4, 1.6), (1.6, 1.8), (2.2, 2.3),
       (2.3, 2.4), (2.4, 2.5), (2.5, 2.6), (2.6, 2.7),
       (2.7, 2.8), (2.8, 2.9), (2.9, 3.0), (3.0, 3.1), (3.1, 3.2), (3.2, 3.3), (3.3, 3.4), (3.4, 3.5),
       (3.7, 4.1), (4.1, 4.7), (4.7, 5.5), (5.5, 6.5)]

lfs = [] 

for i, zl in enumerate(zls):

    print 'z =', zl

    lfi = lf(quasar_files=qlumfiles, selection_maps=selnfiles, zlims=zl)

    print '{:d} quasars in this bin.'.format(lfi.z.size)

    print 'sids: '+'  '.join(['{:2d}'.format(x.sid) for x in lfi.maps])
    print 'size: '+'  '.join(['{:5}'.format(x.z.size) for x in lfi.maps])
    print 'pmax: '+'  '.join(['{:.6f}'.format(x.p.max()) for x in lfi.maps])
    print 'pmin: '+'  '.join(['{:.6f}'.format(x.p.min()) for x in lfi.maps])
    print ' '
    
    g = (np.log10(1.e-6), -25.0, -3.0, -1.5)
    b = lfi.bestfit(g, method=method)
    print b 

    lfi.prior_min_values = np.array([-14.0, -32.0, -7.0, -4.0])
    zmin, zmax = zl 
    if zmin > 5.4:
        # Special priors for z = 6 data.
        lfi.prior_max_values = np.array([-4.0, -20.0, -4.0, 0.0])
        # Change result of optimize.minimize so that emcee works.
        lfi.bf.x[2] = -5.0
    else:
        lfi.prior_max_values = np.array([-4.0, -20.0, 0.0, 0.0])
    assert(np.all(lfi.prior_min_values < lfi.prior_max_values))

    lfi.run_mcmc()
    lfi.get_percentiles()

    write=False
    if write: 
        with open('phi_star_fineBins.dat', 'a') as f:
            f.write(('{:.3f} '*6).format(lfi.z.mean(), zl[0], zl[1],
                                         lfi.phi_star[0], lfi.phi_star[1], lfi.phi_star[2]))
            f.write('\n')

        with open('M_star_fineBins.dat', 'a') as f:
            f.write(('{:.3f} '*6).format(lfi.z.mean(), zl[0], zl[1],
                                         lfi.M_star[0], lfi.M_star[1], lfi.M_star[2]))
            f.write('\n')

        with open('alpha_fineBins.dat', 'a') as f:
            f.write(('{:.3f} '*6).format(lfi.z.mean(), zl[0], zl[1],
                                         lfi.alpha[0], lfi.alpha[1], lfi.alpha[2]))
            f.write('\n')

        with open('beta_fineBins.dat', 'a') as f:
            f.write(('{:.3f} '*6).format(lfi.z.mean(), zl[0], zl[1],
                                         lfi.beta[0], lfi.beta[1], lfi.beta[2]))
            f.write('\n')
            
    lfs.append(lfi)

    # mosaic.draw(lfs)
