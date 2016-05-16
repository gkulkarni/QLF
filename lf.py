from composite import lf
import sys
import time
from summary import summary_plot as sp
from gammapi import plot_gamma as pg 

# Defaults 
case = 3
method = 'Powell'

case = int(sys.argv[1])
method = sys.argv[2]

if case == 1:
    
    qlumfiles = ['Data/bossdr9color.dat']

    selnfiles = [('Data/ross13_selfunc2.dat',2236.0)]

elif case == 2:

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/dr7z3p7.dat']

    selnfiles = [('Data/ross13_selfunc2.dat',2236.0),
                 ('Data/sdss_selfunc2.dat',6248.0)]

elif case == 3: 

    qlumfiles = ['Data/bossdr9color.dat',
                 'Data/dr7z3p7.dat',
                 'Data/glikman11qso.dat']
    
    selnfiles = [('Data/ross13_selfunc2.dat',2236.0),
                 ('Data/sdss_selfunc2.dat',6248.0),
                 ('Data/glikman11_selfunc_dls.dat',1.71),
                 ('Data/glikman11_selfunc_ndwfs.dat',2.05)]

lfg = lf(quasar_files=qlumfiles, selection_maps=selnfiles)

# g = (0.2, -6.7, 1.1, -29, 0.1, -2.4, -0.6, -3.5)
# g = (-0.5, -3.5, -0.5, -23.5, -0.0, -3.0, -0.0, -2.0)

g = (0.2, -6.7, 1.1, -29, -2.5, -2.0)
# g = (-0.5, -3.5, -0.5, -23.5, -1.0,  -1.0) 

b = lfg.bestfit(g, method=method)
print b

# lfg.create_param_range()
import numpy as np 
# lfg.prior_min_values = (-3.0, -10.0, -3.0, -40.0, -1.0, -5.0, -1.0, -5.0)
# lfg.prior_max_values = ( 3.0,  -2.0,  3.0, -20.0,  1.0, 2.0,  1.0, 2.0)

lfg.prior_min_values = (-3.0, -10.0, -3.0, -40.0, -6.0, -3.0)
lfg.prior_max_values = ( 3.0,  -2.0,  3.0, -20.0,  1.0,  2.0)

assert(np.all(lfg.prior_min_values < lfg.prior_max_values))

lfg.run_mcmc()

labels = [r'$a_0 [\phi_*]$', r'$a_1 [\phi_*]$',
          r'$a_0 [M_*]$', r'$a_1 [M_*]$',
          r'$a_0 [\alpha]$', r'$a_1 [\alpha]$',
          r'$a_0 [\beta]$', r'$a_1 [\beta]$']

lfg.corner_plot(labels=labels)
lfg.chains(labels=labels)

sp(lfg)
pg(lfg) 

