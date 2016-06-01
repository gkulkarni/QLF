import numpy as np 

def ppost(lf, mag, z, thetas, norms):
    """
    Creates an array of likelihoods of a given data point (mag, z) for
    model lf with a set of parameters thetas and corresponding LF
    normalisations norms.

    This likelihood is similar to, e.g., Equation (21) of Fan et
    al. 2001 (ApJ 121 54), but is different from the likelihood used
    in individual.py.  That likelihood is Equation (20) of Fan et
    al. 2001.  The two likelihoods are related but the following form
    is more useful for this application.

    """

    return [10.0**lf.log10phi(t, mag, z)/n for t, n in zip(thetas, norms)]

def lnepost(lf, mag, z, thetas, norms): 
    """
    Given a data point mag, return two things:

    (1) The logarithm of the expectation value of the model likelihood
    of mag over the posterior distribution of theta, and 

    (2) The variance of the logarithm of the model likelihood of mag
    over the posterior distribution of theta. 

    For (1), see Equation (7.4) of BDA3.  For (2), see discussion
    above Equation 7.12 of BDA3.

    """
    
    p = ppost(lf, mag, z, thetas, norms) 
    lnp = np.log(np.mean(p))
    var = np.var(np.log(p), ddof=1) 

    return np.array([lnp, var])

def waic(lf):
    """
    Calculate the Watanabe-Akaike Information Criterion (WAIC) for a
    quasar luminosity model lf (defined in composite.py).

    See page 173 of BDA3. 

    """

    # Sample the posterior and calculate corresponding LF norms.  Note
    # that this is done differently from waic.py to save time.  But
    # the two codes are equivalent.
    S = 100 
    n = np.random.randint(len(lf.samples), size=S)
    thetas = lf.samples[n]
    norms = np.array([lf.lfnorm(theta) for theta in thetas])

    p = np.array([lnepost(lf, m, z, thetas, norms) for m, z in zip(lf.M1450, lf.z)])

    lppd = np.sum(p[:,0]) # Equation (7.5) of BDA3 
    p_waic_2 = np.sum(p[:,1]) # Equation (7.12) of BDA3 

    return -2.0 * (lppd - p_waic_2)
    
