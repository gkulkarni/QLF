"""Assign sizes to completeness "tiles".

You only need to set data, z_tolerance, and mag_tolerance.

"""

import numpy as np
import sys

# filename = sys.argv[1]
#data = np.loadtxt('Data_new/croom09sgp_selfunc.dat')


in_file = 'smooth_maps/' + sys.argv[1]
out_file = 'smooth_maps/' + sys.argv[1][:-4] + '_with_tiles.dat'
print 'Reading ' + in_file
print 'Writing to ' + out_file

data = np.loadtxt(in_file)
fl = open(out_file, 'w')

z_tolerance = 0.0001
mag_tolerance = 0.00001 

fl.write('# Using z_tolerance = {:.4f} and mag_tolerance = {:.4f}\n'.format(z_tolerance, mag_tolerance))

def is_same_z(z1, z2):

    if np.abs(z1-z2) < z_tolerance:
        return True

    return False 

def is_same_mag(m1, m2):

    if np.abs(m1-m2) < mag_tolerance:
        return True

    return False 

def next_z(i, data):

    current_z = data[i, 1]

    try:
        next_z = data[i+1, 1]
        while(is_same_z(current_z, next_z)):
            i = i+1
            next_z = data[i+1, 1]
    except(IndexError):
        next_z = current_z 

    return next_z


def next_mag(i, data):

    current_mag = data[i, 2]

    try:
        dz = np.abs(data[i, 1] - data[i+1, 1])
        if is_same_z(dz, 0.0):
            next_mag = data[i+1, 2]
        else:
            next_mag = current_mag 
    except(IndexError):
        next_mag = current_mag

    return next_mag


def prev_mag(i, data):

    current_mag = data[i, 2]

    if i > 0: 
        dz = np.abs(data[i, 1] - data[i-1, 1])
        if is_same_z(dz, 0.0):
            prev_mag = data[i-1, 2]
        else:
            prev_mag = current_mag
    else:
        prev_mag = current_mag

    return prev_mag


def prev_z(i, data):

    current_z = data[i, 1]

    try:
        if i == 0:
            raise(IndexError)
        prev_z = data[i-1, 1]
        while(is_same_z(current_z, prev_z)):
            i = i-1
            if i == 0:
                raise(IndexError)
            prev_z = data[i-1, 1]
    except(IndexError):
        prev_z = current_z

    return prev_z

dzs = []
dmags = [] 

for i, tile in enumerate(data):

    z = tile[1]
    mag = tile[2]
    p = tile[3]
    
    dz_next = np.abs(next_z(i, data) - z)
    dz_prev = np.abs(prev_z(i, data) - z)

    if is_same_z(dz_next, 0.0):
        dz_next = dz_prev

    if is_same_z(dz_prev, 0.0):
        dz_prev = dz_next    

    dmag_next = np.abs(next_mag(i, data) - mag)
    dmag_prev = np.abs(prev_mag(i, data) - mag)

    if is_same_mag(dmag_next, 0.0):
        dmag_next = dmag_prev

    if is_same_mag(dmag_prev, 0.0):
        dmag_prev = dmag_next

    dz = (dz_prev + dz_next)/2.0
    dmag = (dmag_prev + dmag_next)/2.0

    dzs.append(dz)
    dmags.append(dmag)
    
    fl.write('{:>5d}  {:.3f}  {:.2f}  {:.5f}  {:.4f}  {:.3f}\n'.format(i+1, z, mag, p, dz, dmag))

dzs = np.array(dzs)
dmags = np.array(dmags)
    
fl.write('# dzs = ' + str(np.unique(dzs.round(decimals=4))) + '\n')
fl.write('# dmags = ' + str(np.unique(dmags.round(decimals=4))) + '\n') 

fl.close()

print '# dzs = ' + str(np.unique(dzs.round(decimals=4)))
print '# dmags = ' + str(np.unique(dmags.round(decimals=4)))





