import numpy as np
from individual import lf
from summary import get_percentiles

# Defaults 
case = 7
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

    # Note: Glikman entries wrong; see case = 5 instead.
    selnfiles = [('Data/ross13_selfunc2.dat',2236.0,1),
                 ('Data/sdss_selfunc2.dat',6248.0,13),
                 ('Data/glikman11_selfunc_dls.dat',1.71,15),
                 ('Data/glikman11_selfunc_ndwfs.dat',2.05,6)]

elif case == 4: 

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/dr7z3p7.dat',
                 'Data/glikman11qso.dat',
                 'Data/mcgreer13_s82sample.dat',
                 'Data/mcgreer13_dr7sample.dat',
                 'Data/mcgreer13_s82extend.dat']

    # Note: Glikman entries wrong; see case = 5 instead.    
    selnfiles = [('Data/ross13_selfunc2.dat',2236.0,1),
                 ('Data/sdss_selfunchigh.dat',6248.0,13),
                 ('Data/glikman11_selfunc_dls.dat',1.71,15),
                 ('Data/glikman11_selfunc_ndwfs.dat',2.05,6),
                 ('Data/mcgreer13_s82selfunc.dat',235.0,8),
                 ('Data/mcgreer13_dr7selfunc.dat',6248.0,16)]

elif case == 5: 

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
                 'Data/kashikawa15_sample.dat']
    
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
                 ('Data/kashikawa15_sel.dat',6.50,11)]

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

    # All high-z (z >= 2.2) quasars; includes Giallongo 

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
    
    
z = np.array([0.8, 0.9, 1.0, 1.1, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 
              2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.5, 4.0, 4.5, 5.5, 7.0])

lfs = [] 

for i, rs in enumerate(z[:-1]):

    print rs 

    zl = (z[i], z[i+1])
    lfi = lf(quasar_files=qlumfiles, selection_maps=selnfiles, zlims=zl)

    g = (np.log10(1.e-6), -25.0, -3.0, -1.5)
    
    b = lfi.bestfit(g, method=method)
    print b

    # lfi.prior_min_values = np.array([-9.0, -28.0, -7.0, -4.0])
    # lfi.prior_max_values = np.array([-5.0, -22.0, -2.0, -1.0])
    # assert(np.all(lfi.prior_min_values < lfi.prior_max_values))

    lfi.create_param_range()
    
    lfi.run_mcmc()
    lfi.get_percentiles()
    lfi.draw(lfi.z.mean(), dirname='set9/', plotlit=False)
    # lfi.get_gammapi_percentiles(lfi.z.mean(),rt=False) 
    
    lfs.append(lfi)

write_percentiles = True
if write_percentiles:
    zs = np.array([m.z.mean() for m in lfs])
    phi_star = get_percentiles(lfs, param=1, individuals_isfile=False)
    m_star   = get_percentiles(lfs, param=2, individuals_isfile=False)
    alpha    = get_percentiles(lfs, param=3, individuals_isfile=False)
    beta     = get_percentiles(lfs, param=4, individuals_isfile=False)
    np.savez('percs_full_lowz.npz', zs=zs, phi_star=phi_star, m_star=m_star,
             alpha=alpha, beta=beta)
    
