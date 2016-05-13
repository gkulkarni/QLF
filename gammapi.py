import numpy as np 

def luminosity(M):
    return 10.0**((51.60-M)/2.5) # ergs s^-1 Hz^-1 

def f(loglf, theta, M, z, fit='composite'):
    # SED power law index is from Beta's paper.
    L = luminosity(M)
    if fit=='individual':
        # If this gammapi calculation is for an individual fit (in,
        # say, fitlf.py) then loglf might not have any z argument.
        return (10.0**loglf(theta, M))*L*((912.0/1450.0)**0.61)
    return (10.0**loglf(theta, M, z))*L*((912.0/1450.0)**0.61)
        

def emissivity(loglf, theta, z, mlims, fit='composite'):
    # mlims = (lowest magnitude, brightest magnitude)
    #       = (brightest magnitude, faintest magnitude)
    m = np.linspace(mlims[0], mlims[1], num=1000)
    if fit=='individual':
        farr = f(loglf, theta, m, z, fit='individual')
    else:
        farr = f(loglf, theta, m, z)
    return np.trapz(farr, m) # erg s^-1 Hz^-1 Mpc^-3 

def Gamma_HI(loglf, theta, z, fit='composite'):

    if fit=='composite':
        fit_type = 'composite'
    else:
        fit_type = 'individual' 

    # Taken from Equation 11 of Lusso et al. 2015.
    em = emissivity(loglf, theta, z, (-30.0, -23.0), fit=fit_type)
    alpha_EUV = -1.7
    part1 = 4.6e-13 * (em/1.0e24) * ((1.0+z)/5.0)**(-2.4) / (1.5-alpha_EUV) # s^-1 

    em = emissivity(loglf, theta, z, (-23.0, -20.0), fit=fit_type)
    alpha_EUV = -0.56
    part2 = 4.6e-13 * (em/1.0e24) * ((1.0+z)/5.0)**(-2.4) / (1.5-alpha_EUV) # s^-1

    return part1+part2 

def Gamma_HI_singleslope(loglf, theta, z, fit='composite'):

    if fit=='composite':
        fit_type = 'composite'
    else:
        fit_type = 'individual' 

    # Taken from Equation 11 of Lusso et al. 2015.
    em = emissivity(loglf, theta, z, (-30.0, -20.0), fit=fit_type)
    alpha_EUV = -1.7
    return 4.6e-13 * (em/1.0e24) * ((1.0+z)/5.0)**(-2.4) / (1.5-alpha_EUV) # s^-1 

