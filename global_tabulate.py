import numpy as np 

def write(lfg1, lfg2, lfg3):

    c1 = np.median(lfg1.samples, axis=0)
    l1 = np.percentile(lfg1.samples, 15.87, axis=0) 
    u1 = np.percentile(lfg1.samples, 84.13, axis=0)
    uperr1 = u1-c1
    downerr1 = c1-l1

    c2 = np.median(lfg2.samples, axis=0)
    l2 = np.percentile(lfg2.samples, 15.87, axis=0) 
    u2 = np.percentile(lfg2.samples, 84.13, axis=0)
    uperr2 = u2-c2
    downerr2 = c2-l2

    c3 = np.median(lfg3.samples, axis=0)
    l3 = np.percentile(lfg3.samples, 15.87, axis=0) 
    u3 = np.percentile(lfg3.samples, 84.13, axis=0)
    uperr3 = u3-c3
    downerr3 = c3-l3

    c3 = np.concatenate((c3, np.array([0.0,0.0,0.0])))
    uperr3 = np.concatenate((uperr3, np.array([0.0,0.0,0.0])))
    downerr3 = np.concatenate((downerr3, np.array([0.0,0.0,0.0])))   
    
    d = np.stack((c1, uperr1, downerr1, c2, uperr2, downerr2, c3, uperr3, downerr3), axis=1)
    
    for i in range(14):
        fs = r'${:.3f}^{{+{:.3f}}}_{{-{:.3f}}}$ & ${:.3f}^{{+{:.3f}}}_{{-{:.3f}}}$ & ${:.3f}^{{+{:.3f}}}_{{-{:.3f}}}$ \\'
        print fs.format(*d[i])

    
