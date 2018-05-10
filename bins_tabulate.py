import numpy as np

zmean, zmin, zmax, phil, phiu, phic, ml, mu, mc, al, au, ac, bl, bu, bc = np.loadtxt('bins.dat', unpack=True)
zbin = (zmin+zmax)/2.0

phi_uperr = phiu-phic
phi_downerr = phic-phil

m_uperr = mu-mc
m_downerr = mc-ml

a_uperr = au-ac
a_downerr = ac-al

b_uperr = bu-bc
b_downerr = bc-bl

for i in range(len(zmean)):
    # s = r'${:.2f}$ & ${:.2f}$ & ${:.2f}$ & ${:.2f}$ & ${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ & ${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ & ${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ & ${:.2f}^{{+{:.2f}}}_{{-{:.2f}}}$ \\'.format(zmean[i], zbin[i], zmin[i], zmax[i], phic[i], phi_uperr[i], phi_downerr[i], mc[i], m_uperr[i], m_downerr[i], ac[i], a_uperr[i], a_downerr[i], bc[i], b_uperr[i], b_downerr[i])


    s = r'{:.2f} & {:.2f} & {:.2f} & {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} & {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} & {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} & {:.2f}^{{+{:.2f}}}_{{-{:.2f}}} \\'.format(zmean[i], zmin[i], zmax[i], phic[i], phi_uperr[i], phi_downerr[i], mc[i], m_uperr[i], m_downerr[i], ac[i], a_uperr[i], a_downerr[i], bc[i], b_uperr[i], b_downerr[i])

    print s 

