import sys
import numpy as np 
from individual import lf 

# Defaults 
case = 3
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

# zl = (2.2, 2.25)
# zl = (3.1, 3.15)

# zl = (3.775, 3.825)
zl = (3.725, 3.875)

zl = (3.6, 4.0)
lfi = lf(quasar_files=qlumfiles, selection_maps=selnfiles, zlims=zl)

g = (np.log10(1.e-6), -25.0, -3.0, -1.5)

b = lfi.bestfit(g, method=method)
print b

lfi.prior_min_values = np.array([-9.0, -28.0, -7.0, -4.0])
lfi.prior_max_values = np.array([-5.0, -22.0, -2.0, -1.0])
assert(np.all(lfi.prior_min_values < lfi.prior_max_values))

lfi.run_mcmc()
lfi.get_percentiles()

lfi.corner_plot()
lfi.chains()

lfi.draw(lfi.z.mean(),plotlit=True)

