import sys
import numpy as np 
from individual import lf 

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

selnfiles = [('Data_new/dr7z2p2_selfunc.dat', 0.1, 0.05, 6248.0, 13, r'SDSS DR7 Richards et al.\ 2006'),
             ('Data_new/croom09sgp_selfunc.dat', 0.3, 0.05, 64.2, 15, r'2SLAQ SGP Croom et al.\ 2009'),
             ('Data_new/croom09ngp_selfunc.dat', 0.3, 0.05, 127.7, 2, r'2SLAQ NGP Croom et al.\ 2009'),
             ('Data_new/ross13_selfunc2.dat', 0.1, 0.05, 2236.0, 1, r'BOSS DR9 Ross et al.\ 2013'),
             ('Data_new/dr7z3p7_selfunc.dat', 0.1, 0.05, 6248.0, 13, r'SDSS DR7 Richards et al.\ 2006'),
             ('Data_new/glikman11_selfunc_ndwfs_old.dat', 0.05, 0.02, 1.71, 3, r'NDWFS Glikman et al.\ 2011'),
             ('Data_new/glikman11_selfunc_dls_old.dat', 0.05, 0.02, 2.05, 6, r'DLS Glikman et al.\ 2011'),
             ('Data_new/yang16_sel.dat', 0.1, 0.05, 14555.0, 17, r'SDSS+WISE Yang et al.\ 2016'),
             ('Data_new/mcgreer13_dr7selfunc.dat', 0.1, 0.05, 6248.0, 8, r'SDSS DR7 McGreer et al.\ 2013'),
             ('Data_new/mcgreer13_s82selfunc.dat', 0.1, 0.05, 235.0, 4, r'SDSS S82 McGreer et al.\ 2013'),
             ('Data_new/jiang08_sel.dat', 0.1, 0.025, 260.0, 9, r'SDSS Deep Jiang et al.\ 2008'),
             ('Data_new/jiang09_sel.dat', 0.1, 0.025, 195.0, 5, r'SDSS Deep Jiang et al.\ 2009'),
             ('Data_new/jiang16main_selfunc.dat', 0.1, 0.05, 11240.0, 18, r'SDSS Main Jiang et al.\ 2016'),
             ('Data_new/fan06_sel.dat', 0.1, 0.025, 6600.0, 16, r'SDSS Main Fan et al.\ 2006'),
             ('Data_new/jiang16overlap_selfunc.dat', 0.1, 0.05, 4223.0, 19, r'SDSS Overlap Jiang et al.\ 2016'),
             ('Data_new/jiang16s82_selfunc.dat', 0.1, 0.05, 277.0, 20, r'SDSS S82 Jiang et al.\ 2016'),
             ('Data_new/willott10_cfhqsdeepsel.dat', 0.1, 0.025, 4.47, 10, r'CFHQS Deep Willott et al.\ 2010'),
             ('Data_new/willott10_cfhqsvwsel.dat', 0.1, 0.025, 494.0, 21, r'CFHQS Very Wide Willott et al.\ 2010'),
             ('Data_new/kashikawa15_sel.dat', 0.05, 0.05, 6.5, 11, r'Subaru Kashikawa et al.\ 2015')]

method = 'Nelder-Mead'

zmin = float(sys.argv[1])
zmax = float(sys.argv[2]) 

zl = (zmin, zmax) 
# zl = (2.6, 2.7)
lfi = lf(quasar_files=qlumfiles, selection_maps=selnfiles, zlims=zl)
print '{:d} quasars in this bin.'.format(lfi.z.size)

g = (np.log10(1.e-6), -25.0, -3.0, -1.5)
b = lfi.bestfit(g, method=method)
print b

lfi.prior_min_values = np.array([-10.0, -29.0, -7.0, -4.0])
lfi.prior_max_values = np.array([-6.0, -23.0, -2.0, -1.0])
assert(np.all(lfi.prior_min_values < lfi.prior_max_values))

lfi.run_mcmc()

lfi.get_percentiles()

with open('phi_star.dat', 'a') as f:
    f.write(('{:.3f}  '*6).format(lfi.z.mean(), zl[0], zl[1], lfi.phi_star[0], lfi.phi_star[1], lfi.phi_star[2]))
    f.write('\n')

with open('M_star.dat', 'a') as f:
    f.write(('{:.3f}  '*6).format(lfi.z.mean(), zl[0], zl[1], lfi.M_star[0], lfi.M_star[1], lfi.M_star[2]))
    f.write('\n')

with open('alpha.dat', 'a') as f:
    f.write(('{:.3f}  '*6).format(lfi.z.mean(), zl[0], zl[1], lfi.alpha[0], lfi.alpha[1], lfi.alpha[2]))
    f.write('\n')

with open('beta.dat', 'a') as f:
    f.write(('{:.3f}  '*6).format(lfi.z.mean(), zl[0], zl[1], lfi.beta[0], lfi.beta[1], lfi.beta[2]))
    f.write('\n')
    
lfi.corner_plot()
lfi.chains()

lfi.draw(lfi.z.mean())
