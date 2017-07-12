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
from astropy.stats import poisson_conf_interval as pci
from astropy.stats import knuth_bin_width  as kbw
from scipy.integrate import quad 
import cosmolopy.distance as cd
cosmo = {'omega_M_0':0.3,
         'omega_lambda_0':0.7,
         'omega_k_0':0.0,
         'h':0.70}
import gammapi
import rtg
import corner

m_cutoff_13 = -22.1
m_cutoff_15 = 0.0

def getqlums(lumfile, zlims=None):

    """Read quasar luminosities."""

    with open(lumfile,'r') as f: 
        z, mag, p, area, sample_id = np.loadtxt(lumfile,
                                                usecols=(1,2,3,4,5),
                                                unpack=True)
    if zlims is None:
        select = None
    else:
        z_min, z_max = zlims 
        select = ((z>=z_min) & (z<z_max))

    z_all = z[select]
    mag_all = mag[select]
    p_all = p[select]
    area_all = area[select]
    sample_id_all = sample_id[select]

    try:
        sid = sample_id[0]
    except(IndexError):
        sid = sample_id

    select = None 
        
    if sid == 13:
        # Restrict Richards sample (1) to z < 2.2 as there are
        # only three qsos with z = 2.2; (2) to z >= 0.6 to avoid
        # host galaxy contamination; (3) to m > -23 at low z to
        # avoid incompleteness; and (4) to m < -26 at high z to
        # avoid incompleteness.  Also see selmap below.
        select = (((z_all<2.2) & (mag_all<m_cutoff_13) )|
                  ((z_all>=3.5) & (z_all<4.7) & (p_all>0.94)))

    if sid == 15:
        select = ((z_all < 2.2) & (mag_all < m_cutoff_15))

    if sid == 8:
        select = (mag_all > -26.73)

    z = z_all[select]
    mag = mag_all[select]
    p = p_all[select]
    area = area_all[select]
    sample_id = sample_id_all[select]

    return (z, mag, p, area, sample_id, z_all, mag_all, p_all,
            area_all, sample_id_all)
        
def getselfn(selfile, zlims=None):

    """Read selection map."""

    with open(selfile,'r') as f: 
        z, mag, p = np.loadtxt(f, usecols=(1,2,3), unpack=True)

    if zlims is None:
        select = None
    else:
        z_min, z_max = zlims 
        select = ((z>=z_min) & (z<z_max))

    return z[select], mag[select], p[select]

def volume(z, area, cosmo=cosmo):

    omega = (area/41253.0)*4.0*np.pi # str
    volperstr = cd.diff_comoving_volume(z,**cosmo) # cMpc^3 str^-1 dz^-1

    return omega*volperstr # cMpc^3 dz^-1

def percentiles(x):
    
    u = np.percentile(x, 15.87) 
    l = np.percentile(x, 84.13)
    c = np.median(x) 

    return [u, l, c] 

class selmap:

    def __init__(self, x, zlims=None):

        selection_map_file, dm, dz, area, sample_id, label = x

        self.label = label
        self.sid = sample_id
        self.dz = dz
        self.dm = dm
        
        if sample_id == 7:
            # Set dz and dm for Giallongo's sample.  This sample needs
            # special treatment due to non-uniform selection map grid.
            # Here, we are assuming that np.diff(zlims) is smaller
            # than the delta-z values in Giallongo's selection maps.
            self.dz = np.diff(zlims)
            self.dm = np.array([1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
            with open(selection_map_file, 'r') as f: 
                z, mag, p = np.loadtxt(f, usecols=(1,2,3), unpack=True)
            z_min, z_max = zlims 
            select = ((z>=z_min) & (z<z_max))
            self.dm = self.dm[select]
            
        self.z_all, self.m_all, self.p_all = getselfn(selection_map_file, zlims=zlims)

        self.z = self.z_all
        self.m = self.m_all
        self.p = self.p_all 
    
        select = None 
        
        if sample_id == 7:
            # Correct Giallongo's p values to match published LF.  See
            # comments in giallongo15_sel_correction.dat.
            with open(selection_map_file, 'r') as f: 
                z, mag, p = np.loadtxt(f, usecols=(1,2,3), unpack=True)
            z_min, z_max = zlims 
            select = ((z>=z_min) & (z<z_max))
            with open('Data_new/giallongo15_sel_correction.dat', 'r') as f: 
                corr = np.loadtxt(f, usecols=(4,), unpack=True)
            corr = corr[select]
            self.p_all = self.p_all/corr

        if sample_id == 13:
            # Restrict Richards sample (1) to z < 2.2 as there are
            # only three qsos with z = 2.2; (2) to z >= 0.6 to avoid
            # host galaxy contamination; (3) to m > -23 at low z to
            # avoid incompleteness; and (4) to m < -26 at high z to
            # avoid incompleteness.  Also see getqlums above.
            select = (((self.z_all<2.2) & (self.m_all<m_cutoff_13)) |
                      ((self.z_all>=3.5) & (self.z_all<4.7) & (self.p_all>0.94)))

        if sample_id == 15:
            z_min, z_max = zlims 
            select = ((self.z_all<2.2) & (self.m_all<m_cutoff_15))
        
        if sample_id == 8:
            # Restrict McGreer's samples to faint quasars to avoid
            # overlap with Yang.
            select = (self.m_all>-26.73)

        if self.z_all.size == 0:
            return # This selmap has no points in zlims
            
        self.z = self.z_all[select]
        self.m = self.m_all[select]
        self.p = self.p_all[select]

        if self.z.size == 0:
            return # This selmap has no points in zlims

        self.area = area
        self.volarr = volume(self.z, self.area)*self.dz
        self.volarr_all = volume(self.z_all, self.area)*self.dz

        return

    def nqso(self, lumfn, theta):

        try: 
            psi = 10.0**lumfn.log10phi(theta, self.m)
            # Except for Giallongo's sample, self.dm is assumed to be
            # constant here; may not be true.
            tot = psi*self.p*self.volarr*self.dm
            return np.sum(tot)
        except(AttributeError):
            return 0 
            
class lf:

    def __init__(self, quasar_files=None, selection_maps=None, zlims=None):

        self.zlims = zlims

        for datafile in quasar_files:
            (z, m, p, area, sid,
             z_all, m_all, p_all,
             area_all, sid_all) = getqlums(datafile, zlims=zlims)

            try:
                self.z=np.append(self.z,z)
                self.M1450=np.append(self.M1450,m)
                self.p=np.append(self.p,p)
                self.area=np.append(self.area,area)
                self.sid=np.append(self.sid,sid)
            except(AttributeError):
                self.z=z
                self.M1450=m
                self.p=p
                self.area=area
                self.sid=sid

            try:
                self.z_all=np.append(self.z_all,z_all)
                self.M1450_all=np.append(self.M1450_all,m_all)
                self.p_all=np.append(self.p_all,p_all)
                self.area_all=np.append(self.area_all,area_all)
                self.sid_all=np.append(self.sid_all,sid_all)
            except(AttributeError):
                self.z_all=z_all
                self.M1450_all=m_all
                self.p_all=p_all
                self.area_all=area_all
                self.sid_all=sid_all
                
        if zlims is not None:
            self.dz = zlims[1]-zlims[0]

        self.maps = [selmap(x, zlims) for x in selection_maps]

        # Remove selection maps that lie outside our redshift range.
        self.maps = [x for x in self.maps if x.z.size > 0]
        
        return

    def log10phi(self, theta, mag):

        log10phi_star, M_star, alpha, beta = theta 

        phi = 10.0**log10phi_star / (10.0**(0.4*(alpha+1)*(mag-M_star)) +
                                     10.0**(0.4*(beta+1)*(mag-M_star)))
        return np.log10(phi)

    def lfnorm(self, theta):

        ns = np.array([x.nqso(self, theta) for x in self.maps])
        return sum(ns) 

    def neglnlike(self, theta):

        logphi = self.log10phi(theta, self.M1450) # Mpc^-3 mag^-1
        logphi /= np.log10(np.e) # Convert to base e 

        return -2.0*logphi.sum() + 2.0*self.lfnorm(theta)

    def bestfit(self, guess, method='Nelder-Mead'):
        result = op.minimize(self.neglnlike,
                             guess,
                             method=method,
                             options={'maxfev': 8000,
                                      'maxiter': 8000,
                                      'disp': True})

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

    def get_percentiles(self):
        """
        Get 1-sigma errors on the LF parameters.

        """
        self.phi_star = percentiles(self.samples[:,0])
        self.M_star = percentiles(self.samples[:,1])
        self.alpha = percentiles(self.samples[:,2])
        self.beta = percentiles(self.samples[:,3])

        return 

    def corner_plot(self, labels=[r'$\phi_*$', r'$M_*$', r'$\alpha$', r'$\beta$'], dirname=''):

        mpl.rcParams['font.size'] = '14'
        self.medians = np.median(self.samples, axis=0)
        f = corner.corner(self.samples, labels=labels, truths=self.medians)
        plotfile = dirname+'triangle.png'
        f.savefig(plotfile)
        mpl.rcParams['font.size'] = '22'

        return
    
    def plot_chains(self, fig, param, ylabel):
        ax = fig.add_subplot(self.bf.x.size, 1, param+1)
        for i in range(self.nwalkers): 
            ax.plot(self.sampler.chain[i,:,param], c='k', alpha=0.1)
        self.medians = np.median(self.samples, axis=0)
        ax.axhline(self.medians[param], c='#CC9966', dashes=[7,2], lw=2) 
        ax.set_ylabel(ylabel)
        if param+1 != self.bf.x.size:
            ax.set_xticklabels('')
        else:
            ax.set_xlabel('step')
            
        return 

    def chains(self, labels=[r'$\phi_*$', r'$M_*$', r'$\alpha$', r'$\beta$'], dirname=''):

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
    
    def plot_posterior_sample_lfs(self, ax, mags, **kwargs):

        random_thetas = self.samples[np.random.randint(len(self.samples), size=300)]
        for theta in random_thetas:
            phi_fit = self.log10phi(theta, mags)
            ax.plot(mags, phi_fit, **kwargs)

        return

    def plot_bestfit_lf(self, ax, mags, **kwargs):

        phi_fit = self.log10phi(self.bf.x, mags)
        ax.plot(mags, phi_fit, **kwargs)
        ax.plot(mags, phi_fit, lw=1, c='k', zorder=kwargs['zorder'])

        return

    def quasar_volume(self, sample_id):

        smap = [x for x in self.maps if x.sid == sample_id]

        return smap[0].volume # cMpc^3

    def binVol(self, selmap, mrange, zrange):

        """

        Calculate volume in an M-z bin for *one* selmap.

        """

        v = 0.0
        for i in xrange(selmap.m.size):
            if (selmap.m[i] >= mrange[0]) and (selmap.m[i] < mrange[1]):
                if (selmap.z[i] >= zrange[0]) and (selmap.z[i] < zrange[1]):
                    if selmap.sid == 7:
                        v += selmap.volarr[i]*selmap.p[i]*selmap.dm[i]
                    else:
                        v += selmap.volarr[i]*selmap.p[i]*selmap.dm

        return v

    def totBinVol(self, m, mbins, selmaps):

        """
        
        Given magnitude bins mbins and a list of selection maps
        selmaps, compute the volume for an object with magnitude m.

        """

        idx = np.searchsorted(mbins, m)
        mlow = mbins[idx-1]
        mhigh = mbins[idx]
        mrange = (mlow, mhigh)

        v = np.array([self.binVol(x, mrange, self.zlims) for x in selmaps])
        total_vol = v.sum() 

        return total_vol
    
    def get_lf(self, sid, z_plot):

        # Bin data.  This is only for visualisation and to compare
        # with reported binned values.  

        z = self.z[self.sid==sid]
        m = self.M1450[self.sid==sid]
        p = self.p[self.sid==sid]

        selmaps = [x for x in self.maps if x.sid == sid]

        if sid==6:
            # Glikman's sample needs wider bins.
            bins = np.array([-26.0, -25.0, -24.0, -23.0, -22.0, -21])
        else:
            bins = np.arange(-30.9, -17.3, 0.6)
        
        v1 = np.array([self.totBinVol(x, bins, selmaps) for x in m])

        v1_nonzero = v1[np.where(v1>0.0)]
        m = m[np.where(v1>0.0)]

        h = np.histogram(m, bins=bins, weights=1.0/(v1_nonzero))

        nums = h[0]
        mags = (h[1][:-1] + h[1][1:])*0.5
        dmags = np.diff(h[1])*0.5

        left = mags - h[1][:-1]
        right = h[1][1:] - mags
        
        phi = nums
        logphi = np.log10(phi) # cMpc^-3 mag^-1

        # Calculate errorbars on our binned LF.  These have been estimated
        # using Equations 1 and 2 of Gehrels 1986 (ApJ 303 336), as
        # implemented in astropy.stats.poisson_conf_interval.  The
        # interval='frequentist-confidence' option to that astropy function is
        # exactly equal to the Gehrels formulas, although the documentation
        # does not say so.
        n = np.histogram(m, bins=bins)[0]
        nlims = pci(n,interval='frequentist-confidence')
        nlims *= phi/n 
        uperr = np.log10(nlims[1]) - logphi 
        downerr = logphi - np.log10(nlims[0])
        
        return mags, left, right, logphi, uperr, downerr

    def plot_literature(self, ax, z_plot):

        """
        Magic number warning: the selection function below is set by hand! 

        """
        
        qlf_file = 'Data/allqlfs.dat'
        (counter, sample, z_bin, z_min, z_max, z_mean, M1450, left, right,
         logphi, uperr, downerr, nqso, Veff, P) = np.loadtxt(qlf_file, unpack=True)
        
        selection = ((z_min<=z_plot) & (z_max>z_plot)) 
        
        def select(a):
            return a[selection]

        z_bin   = select(z_bin)
        z_min   = select(z_min)
        z_max   = select(z_max)
        z_mean  = select(z_mean)
        M1450   = select(M1450)
        left    = select(left)
        right   = select(right) 
        logphi  = select(logphi)
        uperr   = select(uperr)
        downerr = select(downerr)
        nqso    = select(nqso)
        Veff    = select(Veff)
        P       = select(P)
        sample  = select(sample)

        ax.scatter(M1450, logphi, c='#d7191c',
                   edgecolor='None', zorder=301,
                   label=r'reported values')
        
        ax.errorbar(M1450, logphi, ecolor='#d7191c', capsize=0,
                    xerr=np.vstack((left, right)),
                    yerr=np.vstack((uperr, downerr)),
                    fmt='None', zorder=302)

        return 
        
    def plot_hopkins(self, ax, filename):

        with open(filename, 'r') as f:
            M1450, phi = np.loadtxt(f, usecols=(1,4), unpack=True)

        # 0.4 changes from per units log_10(L) to per unit M
        phi = np.log10(phi) - np.log10(0.4)
        ax.plot(M1450, phi, lw=2, c='k', label='Hopkins')

        return

    def draw(self, z_plot, composite=None, dirname='', plotlit=False):
        """
        Plot data, best fit LF, and posterior LFs.

        """
        mpl.rcParams['font.size'] = '22'

        fig = plt.figure(figsize=(7, 7), dpi=100)
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params('both', which='major', length=7, width=1)
        ax.tick_params('both', which='minor', length=3, width=1)

        mag_plot = np.linspace(-30.0,-20.0,num=100) 
        self.plot_posterior_sample_lfs(ax, mag_plot, lw=1,
                                       c='#ffbf00', alpha=0.1, zorder=2) 
        self.plot_bestfit_lf(ax, mag_plot, lw=2,
                             c='#ffbf00', label='fit', zorder=3)

        cs = {13: 'r', 15:'g', 1:'b', 17:'m', 8:'c', 6:'#ff7f0e',
              7:'#8c564b', 18:'#7f7f7f', 10:'#17becf', 11:'r'}

        def dsl(i):
            for x in self.maps:
                if x.sid == i:
                    return x.label
            return

        sids = np.unique(self.sid)
        for i in sids:
            mags, left, right, logphi, uperr, downerr = self.get_lf(i, z_plot)
            ax.scatter(mags, logphi, c=cs[i], edgecolor='None', zorder=4, s=35, label=dsl(i))
            ax.errorbar(mags, logphi, ecolor=cs[i], capsize=0,
                        xerr=np.vstack((left, right)), 
                        yerr=np.vstack((uperr, downerr)),
                        fmt='None', zorder=4)

        if plotlit: 
            self.plot_literature(ax, z_plot) 
            # self.plot_hopkins(ax, 'hopkins_bol_z3.8.dat')
            # self.plot_hopkins(ax, 'hopkins2.dat')
            
        ax.set_xlim(-17.0, -31.0)
        ax.set_ylim(-12.0, -5.0)
        ax.set_xticks(np.arange(-31,-16, 2))

        ax.set_xlabel(r'$M_{1450}$')
        ax.set_ylabel(r'$\log_{10}\left(\phi/\mathrm{cMpc}^{-3}\,\mathrm{mag}^{-1}\right)$')

        legend_title = r'$\langle z\rangle={0:.3f}$'.format(z_plot)
        plt.legend(loc='lower left', fontsize=12, handlelength=3,
                   frameon=False, framealpha=0.0, labelspacing=.1,
                   handletextpad=0.4, borderpad=0.2, scatterpoints=1, title=legend_title)

        plottitle = r'${:g}\leq z<{:g}$'.format(self.zlims[0], self.zlims[1]) 
        plt.title(plottitle, size='medium', y=1.01)

        plotfile = dirname+'lf_z{0:.3f}.pdf'.format(z_plot)
        plt.savefig(plotfile, bbox_inches='tight')

        plt.close('all') 

        return 

    def get_gammapi_percentiles(self, z, rt=True):
        """
        Calculate photoionization rate posterior mean value and 1-sigma
        percentile.

        """
        if rt:
            rindices = np.random.randint(len(self.samples), size=100)
            g = np.array([np.log10(rtg.gamma_HI(z, self.log10phi, theta,
                                                individual=True))
                          for theta
                          in self.samples[rindices]])
            u = np.percentile(g, 15.87) 
            l = np.percentile(g, 84.13)
            c = np.mean(g)
            self.gammapi = [u, l, c]

            gammafile = 'gammahi_z{0:.3f}'.format(z)
            np.savez(gammafile, z=z, g=g, ulc=self.gammapi) 
            
        else:
            rindices = np.random.randint(len(self.samples), size=300)
            g = np.array([np.log10(gammapi.Gamma_HI(self.log10phi, theta, z,
                                                    fit='individual'))
                          for theta
                          in self.samples[rindices]])
            u = np.percentile(g, 15.87) 
            l = np.percentile(g, 84.13)
            c = np.mean(g)
            self.gammapi = [u, l, c]
            
        return 
    
