import numpy as np 

def parameter_atz(lf, z, param_number):
    """Return value of one of phi*, M*, alpha, and beta at redshift z for
    luminosity function lf.

    param_number sould be 0, 1, 2, 3 for phi*, M*, alpha, beta.

    """
    
    np.random.seed()
    nsample = 1000
    rsample = lf.samples[np.random.randint(len(lf.samples), size=nsample)]
    
    prm = np.zeros(nsample)
    for i, theta in enumerate(rsample):
        params = lf.getparams(theta)
        if param_number == 3 and len(theta) > 11:
            prm[i] = lf.atz_beta(z, params[param_number])
        else:
            prm[i] = lf.atz(z, params[param_number])

    best_fit = np.median(prm)
    
    return best_fit 
