import numpy as np

def ppost(lf, mag, theta):
    """
    Likelihood of data point mag for model lf with parameters theta.

    This likelihood is similar to, e.g., Equation (21) of Fan et
    al. 2001 (ApJ 121 54), but is different from the likelihood used
    in individual.py.  That likelihood is Equation (20) of Fan et
    al. 2001.  The two likelihoods are related but the following form
    is more useful for this application.

    """

    return 10.0**lf.log10phi(theta, mag)/lf.lfnorm(theta)

def ppostdist(lf, mag, S=100): 
    """
    Given a data point mag, return an array of probabilities under
    model parameters sampled from the posterior.  See, e.g., Equation
    (7.5) of BDA3.

    S values are sampled from the posterior.

    """
    
    n = np.random.randint(len(lf.samples), size=S)
    return(np.array([ppost(lf, mag, t) for t in lf.samples[n]]))

def lnepost(lf, mag): 
    """
    Given a data point mag, return two things:

    (1) The logarithm of the expectation value of the model likelihood
    of mag over the posterior distribution of theta, and 

    (2) The variance of the logarithm of the model likelihood of mag
    over the posterior distribution of theta. 

    For (1), see Equation (7.4) of BDA3.  For (2), see discussion
    above Equation 7.12 of BDA3.

    """
    
    p = ppostdist(lf, mag) 
    lnp = np.log(np.mean(p))
    var = np.var(np.log(p), ddof=1) 

    return np.array([lnp, var])

def waic(lf):
    """
    Calculate the Watanabe-Akaike Information Criterion (WAIC) for a
    quasar luminosity model lf (defined in individual.py).

    See page 173 of BDA3. 

    """

    p = np.array([lnepost(lf, m) for m in lf.M1450])

    lppd = np.sum(p[:,0]) # Equation (7.5) of BDA3 
    p_waic_2 = np.sum(p[:,1]) # Equation (7.12) of BDA3 

    return -2.0 * (lppd - p_waic_2)
    

