import numpy as np
import emcee
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
import cosmolopy.distance as cd
cosmo = {'omega_M_0':0.3,
         'omega_lambda_0':0.7,
         'omega_k_0':0.0,
         'h':0.70}

def lfsampleComp(theta, composite, n, mlims, zlims):

    """
    Return n qso magnitudes between mlims[0] and mlims[1] when the LF
    is described by parameters theta.

    """

    zmin = zlims[0]
    zmax = zlims[1]

    mmin = mlims[0]
    mmax = mlims[1]

    def lnprob(p, theta):

        m, z = p

        if m < mmax and m > mmin and z < zmax and z > zmin:
            return composite.log10phi(theta, m, z)/np.log10(np.e)
        else:
            return -np.inf

    ndim = 2
    nwalkers = 250
    mstart = np.mean(mlims)
    zstart = np.mean(zlims)
    dm = np.diff(mlims)
    dz = np.diff(zlims)
    p = np.array([mstart, zstart])
    p0 = np.random.rand(ndim * nwalkers).reshape((nwalkers, ndim))
    p0[:,0] = p0[:,0]*dm + mlims[0]
    p0[:,1] = p0[:,1]*dz + zlims[0]
    
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[theta])
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 1000)

    idx = np.random.randint(nwalkers*1000, size=n)
    
    ms = sampler.flatchain[idx, 0]
    zs = sampler.flatchain[idx, 1]

    return ms, zs
    
def lfsampleIdvl(theta, individual, n, mlims):

    """
    Return n qso magnitudes between mlims[0] and mlims[1] when the LF
    is described by parameters theta.

    """

    mmin = mlims[0]
    mmax = mlims[1]

    def lnprob(x, theta):

        if x < mmax and x > mmin: 
            return individual.log10phi(theta, x)/np.log10(np.e)
        else:
            return -np.inf 
    
    ndim = 1 
    nwalkers = 250
    dm = np.abs(mmin-mmax)
    p0 = (np.random.rand(ndim*nwalkers)*dm + mmin).reshape((nwalkers, ndim))

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[theta])
    pos, prob, state = sampler.run_mcmc(p0, 100)
    sampler.reset()
    sampler.run_mcmc(pos, 1000)

    sample = sampler.flatchain[:,0]

    return np.random.choice(sample, n)
