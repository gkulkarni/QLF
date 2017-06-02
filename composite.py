import numpy as np
import scipy.optimize as op
import emcee
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
import corner
import cosmolopy.distance as cd
cosmo = {'omega_M_0':0.3,
         'omega_lambda_0':0.7,
         'omega_k_0':0.0,
         'h':0.70}
from numpy.polynomial import Chebyshev as T
from numpy.polynomial.polynomial import polyval

def getselfn(selfile):

    """Read selection map."""

    with open(selfile,'r') as f: 
        z, mag, p = np.loadtxt(selfile, usecols=(1,2,3), unpack=True)
    return z, mag, p 

def getqlums(lumfile):

    """Read quasar luminosities."""

    z, mag, p, area, sample_id = np.loadtxt(lumfile, usecols=(1,2,3,4,5),
                                            unpack=True)

    sample_id = np.atleast_1d(sample_id)

    select = None 

    if sample_id[0] == 13:
        # Restrict Richards (SDSS) sample.
        select = (((z < 2.2) & (z >= 0.68) & (mag < -23.0)) |
                  ((z >= 3.5) & (p > 0.9) & (z < 4.7)))

    if sample_id[0] == 15:
        # Restrict Croom (2SLAQ) sample.
        # select = ((z < 2.2) & (z >= 0.6) & (mag < -23.0))
        select = ((z < 2.2) & (z >= 0.68) & (p > 0.5))

    if sample_id[0] == 1:
        # Restrict BOSS sample.
        select = ((z < 2.2) | (z >= 2.8))
    
    if sample_id[0] == 8:
        # Restrict McGreer's samples to faint quasars to avoid
        # overlap with Yang.
        select = (mag>-26.73)

    z = z[select]
    mag = mag[select]
    p = p[select]

    return z, mag, p 

def volume(z, area, cosmo=cosmo):

    omega = (area/41253.0)*4.0*np.pi # str
    volperstr = cd.diff_comoving_volume(z,**cosmo) # cMpc^3 str^-1 dz^-1

    return omega*volperstr # cMpc^3 dz^-1 

class selmap:

    def __init__(self, selection_map_file, dm, dz, area, sample_id):

        self.z, self.m, self.p = getselfn(selection_map_file)

        self.dz = dz
        self.dm = dm 
        print 'dz={:.3f}, dm={:.3f}, sample_id={:d}'.format(dz, dm, sample_id)

        self.sid = sample_id 

        if sample_id == 7:
            # Giallongo's sample needs special treatment due to
            # non-uniform selection map grid.  
            self.dz = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1.5, 1.5, 1.5])
            self.dm = np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

            # Correct Giallongo's p values to match published LF.  See
            # comments in giallongo15_sel_correction.dat.
            with open('Data_new/giallongo15_sel_correction.dat', 'r') as f: 
                corr = np.loadtxt(f, usecols=(4,), unpack=True)
            self.p = self.p/corr

        if sample_id == 1:
            # Restrict BOSS sample 
            select = ((self.z<2.2) | (self.z>=2.8))
            self.z = self.z[select]
            self.m = self.m[select]
            self.p = self.p[select]
            
        if sample_id == 13:
            # Restrict Richards sample
            select = (((self.z < 2.2) & (self.z >= 0.68) & (self.m < -23.0)) |
                      ((self.z >= 3.5) & (self.p > 0.9) & (self.z < 4.7)))
            self.z = self.z[select]
            self.m = self.m[select]
            self.p = self.p[select]

        if sample_id == 15:
            # Restrict Croom sample
            # select = ((self.z < 2.2) & (self.z >= 0.6) & (self.m < -23.0))
            select = ((self.z < 2.2) & (self.z >= 0.68) & (self.p > 0.5))
            self.z = self.z[select]
            self.m = self.m[select]
            self.p = self.p[select]
            
        if sample_id == 8:
            # Restrict McGreer's samples to faint quasars to avoid
            # overlap with Yang.
            select = (self.m>-26.73)
            self.z = self.z[select]
            self.m = self.m[select]
            self.p = self.p[select]

        if self.z.size == 0:
            return # This selmap has no points in zlims

        self.area = area
        self.volume = volume(self.z, self.area) # cMpc^3 dz^-1 

        return

    def nqso(self, lumfn, theta):

        psi = 10.0**lumfn.log10phi(theta, self.m, self.z)
        tot = psi*self.p*self.volume*self.dz*self.dm
        
        return np.sum(tot) 
            
class lf:

    def __init__(self, quasar_files=None, selection_maps=None, pnum=np.array([2,2,1,1])):

        self.pnum = pnum 
        
        for datafile in quasar_files:
            z, m, p = getqlums(datafile)
            try:
                self.z=np.append(self.z,z)
                self.M1450=np.append(self.M1450,m)
                self.p=np.append(self.p,p)
            except(AttributeError):
                self.z=z
                self.M1450=m
                self.p=p

        self.maps = [selmap(*x) for x in selection_maps]

        return

    def atz(self, z, p):

        """Redshift evolution of QLF parameters.

        See dlfParamsdz() below before changing.
        """
        
        return T(p)(1+z)

    def atz_beta(self, z, p):

        """Redshift evolution of QLF parameters.

        See dlfParamsdz() below before changing.
        """

        h, f0, z0, a, b = p 
        zeta = np.log10((1.0+z)/(1.0+z0))
        return h + f0/(10.0**(a*zeta) + 10.0**(b*zeta))
    
    def getparams(self, theta):

        if isinstance(self.pnum, int):
            # Evolution of each LF parameter described by 'atz' using same
            # number 'self.pnum' of parameters.
            splitlocs = self.pnum*np.array([1,2,3])
        else:
            # Evolution of each LF parameter described by 'atz' using
            # different number 'self.pnum[i]' of parameters.
            splitlocs = np.cumsum(self.pnum)

        return np.split(theta,splitlocs)

    def log10phi(self, theta, mag, z):

        params = self.getparams(theta)

        log10phi_star = self.atz(z, params[0])
        M_star = self.atz(z, params[1])
        alpha = self.atz(z, params[2])
        beta = self.atz_beta(z, params[3])
        
        phi = 10.0**log10phi_star / (10.0**(0.4*(alpha+1)*(mag-M_star)) +
                                     10.0**(0.4*(beta+1)*(mag-M_star)))
        return np.log10(phi)

    def lfnorm(self, theta):

        ns = np.array([x.nqso(self, theta) for x in self.maps])
        return np.sum(ns) 
        
    def neglnlike(self, theta):

        logphi = self.log10phi(theta, self.M1450, self.z) # Mpc^-3 mag^-1
        logphi /= np.log10(np.e) # Convert to base e 

        return -2.0*logphi.sum() + 2.0*self.lfnorm(theta)

    def bestfit(self, guess, method='Nelder-Mead'):
        result = op.minimize(self.neglnlike,
                             guess,
                             method=method, options={'maxfev': 8000, 'maxiter': 8000, 'disp': True})

        if not result.success:
            print 'Likelihood optimisation did not converge.'

        self.bf = result 
        return result

    def create_param_range(self):

        half = self.bf.x/2.0
        double = 2.0*self.bf.x
        self.prior_min_values = np.where(half < double, half, double) 
        self.prior_max_values = np.where(half > double, half, double)
        assert(np.all(self.prior_min_values < self.prior_max_values))

        return

    def dlfParamsdz(self, z, theta):

        params = self.getparams(theta)
        pPhiStar = params[0]
        pMStar = params[1]
        pAlpha = params[2]
        pBeta = params[3]
        
        dlog10phiStardz = T.deriv(T(pPhiStar))(1+z)
        dmStardz = T.deriv(T(pMStar))(1+z)
        dalphadz = T.deriv(T(pAlpha))(1+z)

        h, f0, z0, a, b = pBeta
        zeta = np.log10((1.0+z)/(1.0+z0))
        dbetadz = (-f0*(a*10.0**((a-1)*zeta)/(1.0+z0)
                       + b*10.0**((b-1)*zeta)/(1.0+z0))/
                   (10.0**(a*zeta) + 10.0**(b*zeta))**2)

        return dlog10phiStardz, dmStardz, dalphadz, dbetadz

    def dphidphiStar(self, z, m, theta):

        return 10.0**self.log10phi(theta, m, z) * np.log(10.0)

    def dphidalpha(self, z, m, theta):

        params = self.getparams(theta)
        log10phiStar = self.atz(z, params[0])
        mStar = self.atz(z, params[1])
        alpha = self.atz(z, params[2])
        beta = self.atz_beta(z, params[3])

        phi = 10.0**log10phiStar / (10.0**(0.4*(alpha+1)*(m-mStar)) +
                                     10.0**(0.4*(beta+1)*(m-mStar)))
        d = (- phi * np.log(10.0) * 0.4 * (m-mStar) *  10.0**(0.4*(alpha+1)*(m-mStar)) /
             (10.0**(0.4*(alpha+1)*(m-mStar)) + 10.0**(0.4*(beta+1)*(m-mStar))))
        return d

    def dphidbeta(self, z, m, theta):

        params = self.getparams(theta)
        log10phiStar = self.atz(z, params[0])
        mStar = self.atz(z, params[1])
        alpha = self.atz(z, params[2])
        beta = self.atz_beta(z, params[3])
        
        phi = 10.0**log10phiStar / (10.0**(0.4*(alpha+1)*(m-mStar)) +
                                     10.0**(0.4*(beta+1)*(m-mStar)))
        d = (- phi * np.log(10.0) * 0.4 * (m-mStar) *  10.0**(0.4*(beta+1)*(m-mStar)) /
             (10.0**(0.4*(alpha+1)*(m-mStar)) + 10.0**(0.4*(beta+1)*(m-mStar))))
        return d

    def dphidMStar(self, z, m, theta):

        params = self.getparams(theta)
        log10phiStar = self.atz(z, params[0])
        mStar = self.atz(z, params[1])
        alpha = self.atz(z, params[2])
        beta = self.atz_beta(z, params[3])
        
        phi = 10.0**log10phiStar / (10.0**(0.4*(alpha+1)*(m-mStar)) +
                                     10.0**(0.4*(beta+1)*(m-mStar)))
        d1 = 10.0**(0.4*(alpha+1)*(m-mStar)) * np.log(10.0) * (-0.4*(alpha+1))
        d2 = 10.0**(0.4*(beta+1)*(m-mStar)) * np.log(10.0) * (-0.4*(beta+1))
        d = (- phi * (d1+d2) /
             (10.0**(0.4*(alpha+1)*(m-mStar)) + 10.0**(0.4*(beta+1)*(m-mStar))))
        return d 
    
    def dphidz(self, z, m, theta):

        dphiStardz, dmStardz, dalphadz, dbetadz = self.dlfParamsdz(z, theta)

        return (self.dphidphiStar(z, m, theta)*dphiStardz + 
                self.dphidMStar(z, m, theta)*dmStardz + 
                self.dphidalpha(z, m, theta)*dalphadz + 
                self.dphidbeta(z, m, theta)*dbetadz)
    
    def drhoqsodz(self, z, theta):

        mlim = -18.0 # choice should not matter 
        m = np.linspace(-35.0, mlim, num=1000)
        farr = self.dphidz(z, m, theta) 
        return np.trapz(farr, m)
    
    # def lnprior(self, theta):
    #     """
    #     Set up uniform priors.

    #     """

    #     params = self.getparams(theta)
    #     alpha = params[2]
    #     alpha_atz6 = self.atz(6.0, alpha) 
        
    #     if (np.all(theta < self.prior_max_values) and
    #         np.all(theta > self.prior_min_values) and
    #         alpha_atz6 < -4.0):
    #         return 0.0 

    #     return -np.inf

    def lnprior(self, theta):
        """
        Set up uniform priors.

        """

        # if (np.all(theta < self.prior_max_values) and
        #     np.all(theta > self.prior_min_values) and
        #     self.drhoqsodz(10.0, theta) < 0.0):
        #     return 0.0

        z = np.linspace(5,20,num=10)
        d = np.array([self.drhoqsodz(x, theta) for x in z])
        if np.all(d<0.0):
            return 0.0 
        
        # if self.drhoqsodz(10.0, theta) < 0.0:
        #     return 0.0 

        return -np.inf
    
    def lnprob(self, theta):

        lp = self.lnprior(theta)
        
        if not np.isfinite(lp):
            return -np.inf

        return lp - self.neglnlike(theta)

    def run_mcmc(self):
        """
        Run emcee.

        """
        self.ndim, self.nwalkers = self.bf.x.size, 100
        self.mcmc_start = self.bf.x 
        pos = [self.mcmc_start + 1e-4*np.random.randn(self.ndim) for i
               in range(self.nwalkers)]
        
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim,
                                             self.lnprob)

        self.sampler.run_mcmc(pos, 10000)
        self.samples = self.sampler.chain[:, 500:, :].reshape((-1, self.ndim))

        return

    def corner_plot(self, labels=None, dirname=''):

        mpl.rcParams['font.size'] = '14'
        f = corner.corner(self.samples, labels=labels, truths=self.bf.x)
        plotfile = dirname+'triangle.png'
        f.savefig(plotfile)
        mpl.rcParams['font.size'] = '22'

        return

    def plot_chains(self, fig, param, ylabel):
        ax = fig.add_subplot(self.bf.x.size, 1, param+1)
        for i in range(self.nwalkers): 
            ax.plot(self.sampler.chain[i,:,param], c='k', alpha=0.1)
        ax.axhline(self.bf.x[param], c='#CC9966', dashes=[7,2], lw=2) 
        ax.set_ylabel(ylabel)
        if param+1 != self.bf.x.size:
            ax.set_xticklabels('')
        else:
            ax.set_xlabel('step')
            
        return 

    def chains(self, labels=None, dirname=''):

        mpl.rcParams['font.size'] = '10'
        nparams = self.bf.x.size
        plot_number = 0 

        fig = plt.figure(figsize=(12, 2*nparams), dpi=100)
        for i in range(nparams): 
            self.plot_chains(fig, i, ylabel=labels[i])
            
        plotfile = dirname+'chains.pdf' 
        plt.savefig(plotfile,bbox_inches='tight')
        plt.close('all')
        mpl.rcParams['font.size'] = '22'
        
        return
    
    

    
