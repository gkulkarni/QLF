import numpy as np
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
from drawlf import get_lf 

def chisq(lf, composite=None):

    chiSqrds = []; chiSqrdsGlobal = []
    sids = np.unique(lf.sid)
    for i in sids:
        z = lf.z.mean() 
        mags, left, right, logphi, uperr, downerr = get_lf(lf, i, z)

        # Remove -inf points 
        mags = mags[logphi>-100]
        uperr = uperr[logphi>-100]
        downerr = downerr[logphi>-100]
        logphi = logphi[logphi>-100]

        yObs = logphi
        sigmaObs = uperr + downerr

        theta = np.median(lf.samples, axis=0)
        yEst = lf.log10phi(theta, mags)

        chiSqrd = np.sum((yObs-yEst)**2 / sigmaObs**2)
        chiSqrds.append(chiSqrd)

        if composite is not None:
            theta = np.median(composite.samples, axis=0)
            yEst = composite.log10phi(theta, mags, z)

            chiSqrd = np.sum((yObs-yEst)**2 / sigmaObs**2)
            chiSqrdsGlobal.append(chiSqrd)

            return np.sum(np.array(chiSqrds)), np.sum(np.array(chiSqrdsGlobal))

    return np.sum(np.array(chiSqrds))


