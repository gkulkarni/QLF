import numpy as np
import scipy.optimize as op
import emcee
import corner

def fit(x, y, sigma):

    def func(z, a, b, c, d, e):
        e = 10.0**a * (1.0+z)**b * np.exp(-c*z) / (np.exp(d*z)+e)
        return e # erg s^-1 Hz^-1 Mpc^-3

    def neglnlike(theta, x, y, yerr):
        a, b, c, d, e = theta
        model = func(x, a, b, c, d, e)
        inv_sigma2 = 1.0/yerr**2
        return 0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

    guess = np.array([24.6, 4.68, 0.28, 1.77, 26.3])
    method = 'Nelder-Mead'
    
    result = op.minimize(neglnlike,
                         guess,
                         method=method,
                         options={'maxfev': 8000,
                                  'maxiter': 8000,
                                  'disp': False},
                         args=(x, y, sigma))

    def lnprior(theta):
        a, b, c, d, e = theta
        if 10.0 < a < 40.0 and -100.0 < b < 100.0 and -100.0 < c < 100.0 and -500.0 < d < 500.0 and -1000.0 < e < 1000.0:
            return 0.0
        return -np.inf

    def lnprob(theta, x, y, yerr):
        lp = lnprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp - neglnlike(theta, x, y, yerr)

    ndim, nwalkers = 5, 100
    mcmc_start = result.x
    #np.random.seed(5)
    pos = [mcmc_start + 1e-4*np.random.randn(ndim) for i
           in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim,
                                         lnprob, args=(x, y, sigma))

    sampler.run_mcmc(pos, 1000)
    samples = sampler.chain[:, 500:, :].reshape((-1, ndim))
    
    bf = np.median(samples, axis=0)
    print 'bf=', bf
    down = np.percentile(samples, 15.87, axis=0)
    print 'down=', down
    up = np.percentile(samples, 84.13, axis=0)
    print 'up=', up 

    # plt.figure()
    # f = corner.corner(samples, labels=['a', 'b', 'c', 'd', 'e'], truths=bf)
    # f.savefig('corner.pdf')

    return samples, bf#, up, down

   

