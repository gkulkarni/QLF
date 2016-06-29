import numpy as np
import individual
reload(individual) 
from individual import lf
from summary import get_percentiles

# Defaults 
case = 4
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

elif case == 4: 

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/new_data/dr7z3p7.dat',
                 'Data/glikman11qso.dat',
                 'Data/new_data/mcgreer13_s82sample.dat',
                 'Data/new_data/mcgreer13_dr7sample.dat',
                 'Data/new_data/mcgreer13_s82extend.dat']
    
    selnfiles = [('Data/ross13_selfunc2.dat',2236.0,1),
                 ('Data/new_data/sdss_selfunchigh.dat',6248.0,13),
                 ('Data/glikman11_selfunc_dls.dat',1.71,15),
                 ('Data/glikman11_selfunc_ndwfs.dat',2.05,6),
                 ('Data/new_data/mcgreer13_s82selfunc.dat',235.0,8),
                 ('Data/new_data/mcgreer13_dr7selfunc.dat',6248.0,16)]
    
z = np.array([2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.5, 4.0, 4.5, 5.0])

lfs = [] 

for i, rs in enumerate(z[:-1]):

    print rs 

    zl = (z[i], z[i+1])
    lfi = lf(quasar_files=qlumfiles, selection_maps=selnfiles, zlims=zl)

    g = (np.log10(1.e-6), -25.0, -3.0, -1.5)
    
    b = lfi.bestfit(g, method=method)
    print b

    lfi.prior_min_values = np.array([-9.0, -28.0, -7.0, -4.0])
    lfi.prior_max_values = np.array([-5.0, -22.0, -2.0, -1.0])
    assert(np.all(lfi.prior_min_values < lfi.prior_max_values))

    lfi.run_mcmc()
    lfi.get_percentiles()
    lfi.draw(lfi.z.mean(), dirname='set5/', plotlit=True)
    lfi.get_gammapi_percentiles(lfi.z.mean()) 
    
    lfs.append(lfi)

write_percentiles = True
if write_percentiles:
    zs = np.array([m.z.mean() for m in lfs])
    phi_star = get_percentiles(lfs, param=1, individuals_isfile=False)
    m_star   = get_percentiles(lfs, param=2, individuals_isfile=False)
    alpha    = get_percentiles(lfs, param=3, individuals_isfile=False)
    beta     = get_percentiles(lfs, param=4, individuals_isfile=False)
    np.savez('percs_newdata.npz', zs=zs, phi_star=phi_star, m_star=m_star,
             alpha=alpha, beta=beta)
    
