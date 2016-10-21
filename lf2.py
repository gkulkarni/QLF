import sys
import numpy as np 
from individual import lf 
import waic 

# Defaults 
case = 8
method = 'Nelder-Mead'

if case == 1:
    
    qlumfiles = ['Data/bossdr9color.dat']

    selnfiles = [('Data/ross13_selfunc2.dat',2236.0,1)]

elif case == 2:

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/dr7z3p7.dat']

    selnfiles = [('Data/ross13_selfunc2.dat',2236.0,1),
                 ('Data/sdss_selfunc2.dat',6248.0,13)]

elif case == 3: 

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/dr7z3p7.dat',
                 'Data/glikman11qso.dat']
    
    selnfiles = [('Data/ross13_selfunc2.dat',2236.0,1),
                 ('Data/sdss_selfunc2.dat',6248.0,13),
                 ('Data/glikman11_selfunc_dls.dat',1.71,15),
                 ('Data/glikman11_selfunc_ndwfs.dat',2.05,6)]

elif case == 6: 

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/dr7z3p7.dat',
                 'Data/glikman11qso.dat',
                 'Data/mcgreer13_s82sample.dat',
                 'Data/mcgreer13_dr7sample.dat',
                 'Data/mcgreer13_s82extend.dat',
                 'Data/fan06_sample.dat',
                 'Data/jiang08_sample.dat',
                 'Data/jiang09_sample.dat',
                 'Data/willott10_cfhqsdeepsample.dat',
                 'Data/willott10_cfhqsvwsample.dat',
                 'Data/kashikawa15_sample.dat',
                 'Data/dr7z2p2.dat']

    selnfiles = [('Data/ross13_selfunc2.dat',2236.0,1),
                 ('Data/sdss_selfunchigh.dat',6248.0,13),
                 ('Data/glikman11_selfunc_ndwfs.dat',1.71,15),
                 ('Data/glikman11_selfunc_dls.dat',2.05,6),
                 ('Data/mcgreer13_s82selfunc.dat',235.0,8),
                 ('Data/mcgreer13_dr7selfunc.dat',6248.0,16),
                 ('Data/fan06_sel.dat',6600.0,17),
                 ('Data/jiang08_sel.dat',260.0,9),
                 ('Data/jiang09_sel.dat',195.0,18),
                 ('Data/willott10_cfhqsvwsel.dat',494.0,19),
                 ('Data/willott10_cfhqsdeepsel.dat',4.47,10),
                 ('Data/kashikawa15_sel.dat',6.50,11),
                 ('Data/sdss_selfunclow.dat',6248.0,14)]
    
elif case == 7: 

    qlumfiles = ['Data/dr7z3p7.dat',
                 'Data/glikman11qso.dat']

    selnfiles = [('Data/sdss_selfunchigh.dat',6248.0,13),
                 ('Data/glikman11_selfunc_ndwfs.dat',1.71,15),
                 ('Data/glikman11_selfunc_dls.dat',2.05,6)]

elif case == 8:

    # Only high-z (z >= 2.2) quasars; includes Giallongo 

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/dr7z3p7.dat',
                 'Data/glikman11qso.dat',
                 'Data/mcgreer13_s82sample.dat',
                 'Data/mcgreer13_dr7sample.dat',
                 'Data/mcgreer13_s82extend.dat',
                 'Data/fan06_sample.dat',
                 'Data/jiang08_sample.dat',
                 'Data/jiang09_sample.dat',
                 'Data/willott10_cfhqsdeepsample.dat',
                 'Data/willott10_cfhqsvwsample.dat',
                 'Data/kashikawa15_sample.dat',
                 'Data/giallongo15_sample.dat']

    selnfiles = [('Data/ross13_selfunc2.dat',2236.0,1),
                 ('Data/sdss_selfunchigh.dat',6248.0,13),
                 ('Data/glikman11_selfunc_ndwfs.dat',1.71,15),
                 ('Data/glikman11_selfunc_dls.dat',2.05,6),
                 ('Data/mcgreer13_s82selfunc.dat',235.0,8),
                 ('Data/mcgreer13_dr7selfunc.dat',6248.0,16),
                 ('Data/fan06_sel.dat',6600.0,17),
                 ('Data/jiang08_sel.dat',260.0,9),
                 ('Data/jiang09_sel.dat',195.0,18),
                 ('Data/willott10_cfhqsvwsel.dat',494.0,19),
                 ('Data/willott10_cfhqsdeepsel.dat',4.47,10),
                 ('Data/kashikawa15_sel.dat',6.50,11),
                 ('Data/sdss_selfunclow.dat',6248.0,14),
                 ('Data/giallongo15_sel.dat',0.047,7)]
    
    
zl = (4.7,5.1) 

lfi = lf(quasar_files=qlumfiles, selection_maps=selnfiles, zlims=zl)

g = (np.log10(1.e-6), -25.0, -3.0, -1.5)

b = lfi.bestfit(g, method=method)
print b

lfi.create_param_range()

# lfi.prior_min_values = np.array([-9.0, -28.0, -7.0, -4.0])
# lfi.prior_max_values = np.array([-5.0, -22.0, -2.0, -1.0])
# assert(np.all(lfi.prior_min_values < lfi.prior_max_values))

lfi.run_mcmc()
lfi.get_percentiles()

lfi.corner_plot()
lfi.chains()

lfi.draw(lfi.z.mean(),plotlit=False)

# print 'WIAC=', waic.waic(lfi) 

