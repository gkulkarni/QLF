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
import triangle 
import cosmolopy.distance as cd
cosmo = {'omega_M_0':0.3,
         'omega_lambda_0':0.7,
         'omega_k_0':0.0,
         'h':0.70}

def getselfn(selfile):

    """Read selection map."""

    with open(selfile,'r') as f: 
        z, mag, p = np.loadtxt(selfile, usecols=(1,2,3), unpack=True)
    return z, mag, p 

def getqlums(lumfile):

    """Read quasar luminosities."""

    with open(lumfile,'r') as f: 
        z, mag, p = np.loadtxt(lumfile, usecols=(1,2,3), unpack=True)
    return z, mag, p

def volume(z, area, cosmo=cosmo):

    omega = (area/41253.0)*4.0*np.pi # str
    volperstr = cd.diff_comoving_volume(z,**cosmo) # cMpc^3 str^-1 dz^-1

    return omega*volperstr # cMpc^3 dz^-1 

class selmap:

    def __init__(self, selection_map_file, area):

        self.z, self.m, self.p = getselfn(selection_map_file)

        self.dz = np.unique(np.diff(self.z))[-1]
        self.dm = np.unique(np.diff(self.m))[-1]
        # print 'dz={0:.3f}, dm={1:.3f}'.format(self.dz,self.dm)

        self.area = area
        self.volume = volume(self.z, self.area) 

        return

    def nqso(self, lumfn, theta):

        psi = 10.0**lumfn.log10phi(theta, self.m, self.z)
        tot = psi*self.p*self.volume*self.dz*self.dm
        
        return np.sum(tot) 
            
class lf:

    def __init__(self, quasar_files=None, selection_maps=None):

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

        self.maps = [selmap(x[0], x[1]) for x in selection_maps]

        return

    def atz(self, z, p):

        """Redshift evolution of QLF parameters."""

        if len(p) == 2: 
            a0, a1 = p
        elif len(p) == 1:
            a0 = 0
            a1 = p 

        return a0*(1.0+z) + a1

    def getparams(self, theta, pnum=np.array([2,2,1,1])):

        if isinstance(pnum,int):
            # Evolution of each LF parameter described by 'atz' using same
            # number 'pnum' of parameters.
            splitlocs = pnum*np.array([1,2,3])
        else:
            # Evolution of each LF parameter described by 'atz' using
            # different number 'pnum[i]' of parameters.
            splitlocs = np.cumsum(pnum)

        return np.split(theta,splitlocs)

    def log10phi(self, theta, mag, z):

        params = self.getparams(theta)

        log10phi_star = self.atz(z, params[0])
        M_star = self.atz(z, params[1])
        alpha = self.atz(z, params[2])
        beta = self.atz(z, params[3])

        phi = 10.0**log10phi_star / (10.0**(0.4*(alpha+1)*(mag-M_star)) +
                                     10.0**(0.4*(beta+1)*(mag-M_star)))
        return np.log10(phi)

    def lfnorm(self, theta):

        ns = [x.nqso(self, theta) for x in self.maps]
        return sum(ns) 
        
    def neglnlike(self, theta):

        logphi = self.log10phi(theta, self.M1450, self.z) # Mpc^-3 mag^-1
        logphi /= np.log10(np.e) # Convert to base e 

        return -2.0*logphi.sum() + 2.0*self.lfnorm(theta)

    def bestfit(self, guess, method='Nelder-Mead'):
        result = op.minimize(self.neglnlike,
                             guess,
                             method=method, options={'ftol': 1.0e-10})

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

    def lnprior(self, theta):
        """
        Set up uniform priors.

        """
        if (np.all(theta < self.prior_max_values) and
            np.all(theta > self.prior_min_values)):
            return 0.0 

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

        self.sampler.run_mcmc(pos, 1000)
        self.samples = self.sampler.chain[:, 500:, :].reshape((-1, self.ndim))

        return

    def corner_plot(self, labels=None, dirname=''):

        mpl.rcParams['font.size'] = '14'
        f = triangle.corner(self.samples, labels=labels, truths=self.bf.x)
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
    
    

    
