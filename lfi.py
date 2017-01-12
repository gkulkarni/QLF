import sys
import numpy as np 
from individual import lf 

qlumfiles = ['Data_new/dr7z2p2_sample.dat',
             'Data_new/croom09sgp_sample.dat',
             'Data_new/croom09ngp_sample.dat']

selnfiles = [('Data_new/dr7z2p2_selfunc.dat', 0.1, 0.05, 6248.0, 13),
             ('Data_new/croom09sgp_selfunc.dat', 0.3, 0.05, 64.2, 15),
             ('Data_new/croom09ngp_selfunc.dat', 0.3, 0.05, 127.7, 2)]

method = 'Nelder-Mead'
zl = (1.0, 1.5)
lfi = lf(quasar_files=qlumfiles, selection_maps=selnfiles, zlims=zl)

g = (np.log10(1.e-6), -25.0, -3.0, -1.5)
b = lfi.bestfit(g, method=method)
print b

lfi.prior_min_values = np.array([-8.0, -28.0, -7.0, -3.0])
lfi.prior_max_values = np.array([-5.0, -22.0, -2.0, -1.0])
assert(np.all(lfi.prior_min_values < lfi.prior_max_values))

lfi.run_mcmc()

lfi.get_percentiles()
lfi.corner_plot()
lfi.chains()

lfi.draw(lfi.z.mean())
