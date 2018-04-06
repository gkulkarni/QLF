import numpy as np

data = np.loadtxt('emissivity.txt')

zmean = data[:,0]
zmin = data[:,2]
zmax = data[:,1]

zbin = (zmin+zmax)/2.0

e912_18 = data[:,3]
de912_18_plus = data[:,4]
de912_18_minus = data[:,5]

e1450_18 = data[:,6]
de1450_18_plus = data[:,7]
de1450_18_minus = data[:,8]

e912_21 = data[:,9]
de912_21_plus = data[:,10]
de912_21_minus = data[:,11]

e1450_21 = data[:,12]
de1450_21_plus = data[:,13]
de1450_21_minus = data[:,14]

fs = (r'${:.2f}$ & ${:.2f}$ & ${:.2f}$ & ${:.2f}$ '
      r'& ${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ '
      r'& ${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ '
      r'& ${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ '
      r'& ${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ \\')

fs2 = (r'{:.2f} & {:.2f} & {:.2f} & {:.2f} '
      r'& {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} '
      r'& {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} '
      r'& {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} '
      r'& {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} \\')


for i in range(len(zmean)):

    s = fs2.format(zmean[i], zbin[i], zmin[i], zmax[i],
                  e912_18[i], de912_18_plus[i], de912_18_minus[i],
                  e1450_18[i], de1450_18_plus[i], de1450_18_minus[i],
                  e912_21[i], de912_21_plus[i], de912_21_minus[i],
                  e1450_21[i], de1450_21_plus[i], de1450_21_minus[i])
    print s 
