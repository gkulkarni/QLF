import numpy as np 
import matplotlib as mpl
mpl.use('Agg') 
mpl.rcParams['text.usetex'] = True 
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'cm'
mpl.rcParams['font.size'] = '22'
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import chebfit
from numpy.polynomial import Chebyshev as T

import sys 
case = int(sys.argv[1])

if case == 0:
    
    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\alpha$')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7)

    ax.set_ylim(-6, -1)

    zmean, zl, zu, u, l, c = np.loadtxt('alpha.dat', unpack=True)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color='k', edgecolor='None', zorder=2)
    ax.errorbar(zmean, c, ecolor='k', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=2)

    z = np.linspace(0, 7, 500)

    coeffs = chebfit(zmean+1.0, c, 1)
    print coeffs
    plt.plot(z, T(coeffs)(z+1), lw=2, c='forestgreen') 

    plt.savefig('cfit_alpha.pdf',bbox_inches='tight')
    plt.close('all')

elif case == 1: 

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\beta$')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7)

    ax.set_ylim(-3, 0)

    zmean, zl, zu, u, l, c = np.loadtxt('beta.dat', unpack=True)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color='k', edgecolor='None', zorder=2)
    ax.errorbar(zmean, c, ecolor='k', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=2)

    z = np.linspace(0, 7, 500)

    coeffs = chebfit(zmean+1, c, 2)
    print coeffs
    plt.plot(z, T(coeffs)(z+1), lw=2, c='forestgreen') 

    # coeffs = chebfit(zmean, c, 3)
    # print coeffs
    # plt.plot(z, T(coeffs)(z), lw=2, c='tomato') 

    plt.savefig('cfit_beta.pdf',bbox_inches='tight')
    plt.close('all')

elif case == 2: 

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$M_*$')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7)

    ax.set_ylim(-31, -21)

    zmean, zl, zu, u, l, c = np.loadtxt('M_star.dat', unpack=True)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color='k', edgecolor='None', zorder=2)
    ax.errorbar(zmean, c, ecolor='k', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=2)

    z = np.linspace(0, 7, 500)

    coeffs = chebfit(zmean+1, c, 1)
    print coeffs
    plt.plot(z, T(coeffs)(z+1), lw=2, c='forestgreen') 

    plt.savefig('cfit_mstar.pdf',bbox_inches='tight')
    plt.close('all')

elif case == 3:

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)

    ax.set_ylabel(r'$\phi_*$')
    ax.set_xlabel('$z$')
    ax.set_xlim(0.,7)

    ax.set_ylim(-11, -4)

    zmean, zl, zu, u, l, c = np.loadtxt('phi_star.dat', unpack=True)
    left = zmean-zl
    right = zu-zmean
    uperr = u-c
    downerr = c-l
    ax.scatter(zmean, c, color='k', edgecolor='None', zorder=2)
    ax.errorbar(zmean, c, ecolor='k', capsize=0,
                xerr=np.vstack((left, right)), 
                yerr=np.vstack((uperr, downerr)),
                fmt='None', zorder=2)

    z = np.linspace(0, 7, 500)

    coeffs = chebfit(zmean+1, c, 2)
    print coeffs
    plt.plot(z, T(coeffs)(z+1), lw=2, c='forestgreen') 

    plt.savefig('cfit_phistar.pdf',bbox_inches='tight')
    plt.close('all')
    
