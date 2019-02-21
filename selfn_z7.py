import numpy as np

def ukidss_sel():
    
    M_min = -29.3
    M_max = -26.2

    z_min = 6.5
    z_max = 7.4

    dM = 0.1
    dz = 0.1

    n_M = (M_max - M_min)/dM + 1
    n_z = (z_max - z_min)/dz + 1

    M = np.arange(M_min, M_max, dM)
    z = np.arange(z_min, z_max, dz)

    i = 0 
    for red in z:
        for lum in M:
            i = i + 1  
            print '{:d}  {:.2f}  {:.2f}  1.0  {:.2f}  {:.2f}'.format(i, red, lum, dz, dM)

def banados_sel():
    
    M_min = -29.3
    M_max = -26.2

    z_min = 7.4
    z_max = 9.0

    dM = 0.1
    dz = 0.1

    n_M = (M_max - M_min)/dM + 1
    n_z = (z_max - z_min)/dz + 1

    M = np.arange(M_min, M_max, dM)
    z = np.arange(z_min, z_max, dz)

    i = 0 
    for red in z:
        for lum in M:
            i = i + 1  
            print '{:d}  {:.2f}  {:.2f}  1.0  {:.2f}  {:.2f}'.format(i, red, lum, dz, dM)
            

banados_sel()
#ukidss_sel()
