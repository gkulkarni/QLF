import numpy as np
import emcee
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '10'
import matplotlib.pyplot as plt

theta = np.array([-9.03733821, -27.89606624, -4.57536883, -2.31774305])

def lnprob(x, theta):

    if x < -17.0 and x > -31.0: 
        mag = x 
        log10phi_star, M_star, alpha, beta = theta 
        phi = 10.0**log10phi_star / (10.0**(0.4*(alpha+1)*(mag-M_star)) +
                                     10.0**(0.4*(beta+1)*(mag-M_star)))
        return np.log(phi)
    else:
        return -np.inf 

ndim = 1 
nwalkers = 250
p0 = (np.random.rand(ndim*nwalkers)*10.0 - 31.0).reshape((nwalkers, ndim))

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=[theta])
pos, prob, state = sampler.run_mcmc(p0, 100)
sampler.reset()
sampler.run_mcmc(pos, 1000)

fig = plt.figure(figsize=(12, 2), dpi=100)
ax = fig.add_subplot(1,1,1)
for i in range(nwalkers):
    ax.plot(sampler.chain[i,:,0], c='k', alpha=0.1)
ax.set_xlabel('step')
ax.set_ylabel('$M_{1450}$')
plt.savefig('chain.pdf', bbox_inches='tight')
plt.close('all')

