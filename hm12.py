
def plot_f():

    """Plot HM12 HI column density distribution.

    Compare result to left panel of Figure 1 of HM12. 

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'$\log_{10} f(N_\mathrm{HI},z)$')
    ax.set_xlabel(r'$\log_{10}(N_\mathrm{HI}/\mathrm{cm}^{-2})$') 

    ax.set_ylim(-25.0, -7.0)
    ax.set_xlim(11.0, 21.5)

    locs = range(-24, -6, 2)
    labels = ['$'+str(x)+'$' for x in locs]
    plt.yticks(locs, labels)

    n = np.logspace(11.0,23.0,num=1000)
    z = 3.5
    f = vf_HM12(n, z)
    ax.plot(np.log10(n), np.log10(f), lw=2, c='k', label='$z=3.5$') 
    
    z = 2.0
    f = vf_HM12(n, z)
    ax.plot(np.log10(n), np.log10(f/50), lw=2, c='r', label='$z=2.0$') 

    z = 5.0
    f = vf_HM12(n, z)
    ax.plot(np.log10(n), np.log10(f*50), lw=2, c='b', label='$z=5.0$')

    plt.legend(loc='lower left', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.1, scatterpoints=1)

    
    plt.savefig('f.pdf'.format(z),bbox_inches='tight')
    plt.close('all')


def plot_f_vs_z():

    """Plot HM12 HI column density distribution evolution. 

    Just to confirm continuity.

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'$\log_{10} f(N_\mathrm{HI},z)$')
    ax.set_xlabel(r'$z$') 

    n = 1.0e12
    z = np.linspace(0,7,num=100)
    f = vf_HM12(n, z)
    ax.plot(z, np.log10(f), lw=2, c='g',
            label='$N_\mathrm{HI}=10^{12} \mathrm{cm}^{-2}$') 
    
    n = 1.0e16
    z = np.linspace(0,7,num=100)
    f = vf_HM12(n, z)
    ax.plot(z, np.log10(f), lw=2, c='k',
            label='$N_\mathrm{HI}=10^{16} \mathrm{cm}^{-2}$') 

    n = 1.0e18
    z = np.linspace(0,7,num=100)
    f = vf_HM12(n, z)
    ax.plot(z, np.log10(f), lw=2, c='b',
            label='$N_\mathrm{HI}=10^{18} \mathrm{cm}^{-2}$') 
    
    n = 1.0e20
    z = np.linspace(0,7,num=100)
    f = vf_HM12(n, z)
    ax.plot(z, np.log10(f), lw=2, c='r',
            label='$N_\mathrm{HI}=10^{20} \mathrm{cm}^{-2}$') 

    n = 1.0e22
    z = np.linspace(0,7,num=100)
    f = vf_HM12(n, z)
    ax.plot(z, np.log10(f), lw=2, c='brown',
            label='$N_\mathrm{HI}=10^{22} \mathrm{cm}^{-2}$') 
    
    plt.legend(loc='lower left', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.1, scatterpoints=1)

    plt.savefig('fz.pdf'.format(z),bbox_inches='tight')
    plt.close('all')

def plot_dtaudn():

    """Plot dtau_eff/dN_HI. 

    Compare result to right panel of Figure 1 of HM12. 

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'$\log_{10} N_\mathrm{HI} f(N_\mathrm{HI},z)'+
                  '[1-\exp{(-N_\mathrm{HI}\sigma_{912})}]$')
    ax.set_xlabel(r'$\log_{10}(N_\mathrm{HI}/\mathrm{cm}^{-2})$') 

    ax.set_ylim(-3, 0)
    ax.set_xlim(11.0, 21.5)

    locs = range(-3, 1) 
    labels = ['$'+str(x)+'$' for x in locs]
    plt.yticks(locs, labels)

    n = np.logspace(11.0, 23.0, num=1000)
    
    z = 3.5
    fn = n * vf_HM12(n, z) * (1.0-np.exp(-n*sigma_HI(nu0)))
    ax.plot(np.log10(n), np.log10(fn), lw=2, c='k', label='$z=3.5$') 

    print 'z=', z
    t1 = np.trapz(fn, x=np.log(n))
    print 't1=', t1 
    
    fn = n * f(n, z) * (1.0-np.exp(-n*sigma_HI(nu0)))
    ax.plot(np.log10(n), np.log10(fn), lw=2, c='k', dashes=[7,2])

    t2 = np.trapz(fn, x=np.log(n))
    print 't2=', t2
    
    z = 2.0
    fn = n * vf_HM12(n, z) * (1.0-np.exp(-n*sigma_HI(nu0)))
    ax.plot(np.log10(n), np.log10(fn), lw=2, c='r', label='$z=2.0$') 

    print 'z=', z
    t1 = np.trapz(fn, x=np.log(n))
    print 't1=', t1 
    
    fn = n * f(n, z) * (1.0-np.exp(-n*sigma_HI(nu0)))
    ax.plot(np.log10(n), np.log10(fn), lw=2, c='r', dashes=[7,2])

    t2 = np.trapz(fn, x=np.log(n))
    print 't2=', t2
    
    z = 5.0
    fn = n * vf_HM12(n, z) * (1.0-np.exp(-n*sigma_HI(nu0)))
    ax.plot(np.log10(n), np.log10(fn), lw=2, c='b', label='$z=5.0$')

    print 'z=', z
    t1 = np.trapz(fn, x=np.log(n))
    print 't1=', t1 
    
    fn = n * f(n, z) * (1.0-np.exp(-n*sigma_HI(nu0)))
    ax.plot(np.log10(n), np.log10(fn), lw=2, c='b', dashes=[7,2])

    t2 = np.trapz(fn, x=np.log(n))
    print 't2=', t2
    
    plt.legend(loc='upper left', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.1, scatterpoints=1)

    plt.savefig('dtaudn.pdf'.format(z), bbox_inches='tight')
    plt.close('all')

    return

def check_z_refinement_emissivity():

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_yscale('log')

    ws = 1231.55060329
    
    zmax = 7.0
    zmin = 5.0
    dz = 0.1
    n = (zmax-zmin)/dz+1
    zs = np.linspace(zmax, zmin, num=n)

    e = emissivity_HM12(ws/(1.0+zs), zs, grid=False)
    ax.plot(zs, e, lw=2, c='k')
    plt.title('{:g}'.format(dz))

    print np.sum(np.abs(dtdz(zs))*e*dz*c_mpcPerYr*(1.0+zs)**3*cmbympc**2/(4.0*np.pi))

    j = 0.0
    for z in zs:
        e2 = emissivity_HM12(ws/(1.0+z), z, grid=False)
        j = j + (e2*c_mpcPerYr*np.abs(dtdz(z))*dz*(1.0+z)**3)/(4.0*np.pi)
        j = j*cmbympc**2 
    print j

    plt.savefig('czre.pdf', bbox_inches='tight')
    plt.close('all')

    return

def draw_j(j, w, z):

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'$j_\nu$ [$10^{-22}$ erg s$^{-1}$ Hz$^{-1}$ sr$^{-1}$ cm$^{-2}$]')
    ax.set_xlabel(r'$\lambda_\mathrm{rest}$ [\AA]')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylim(1.0e-5, 1.0e5)
    ax.set_xlim(5.0, 4.0e3)
    
    ax.plot(w, j/1.0e-22, lw=2, c='k')
    j_hm12 = bkgintens_HM12(w, z*np.ones_like(w), grid=False)
    ax.plot(w, j_hm12/1.0e-22, lw=2, c='tomato')

    ax.axvline(1216.0, lw=1, c='k', dashes=[7,2])
    ax.axvline(912.0, lw=1, c='k', dashes=[7,2])
    ax.axvline(304.0, lw=1, c='k', dashes=[7,2])
    ax.axvline(228.0, lw=1, c='k', dashes=[7,2])

    plt.title('$z={:g}$'.format(z))
    plt.savefig('j_z{:g}.pdf'.format(z),bbox_inches='tight')
    plt.close('all')

    return


def plot_evol(z, q):

    """Generic function to calculate evolution. 

    Set q to some quantity in j( ) above and then plot it using this
    function.

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_yscale('log')

    ax.plot(z, q, lw=2, c='k')

    plt.savefig('evol.pdf'.format(z),bbox_inches='tight')
    plt.close('all')

    return

def plot_qso_emissivity():

    """Plot HM12 qso emissivity. 
    
    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'emissivity [$10^{39}$~erg s$^{-1}$ Hz$^{-1}$ cMpc$^{-3}$]')
    ax.set_xlabel(r'$\nu$~[Hz]')

    ax.set_yscale('log')
    ax.set_xscale('log')
    
    nu = np.logspace(13.0,17.0,num=10000)

    z = 1.1
    e = np.array([qso_emissivity_hm12(x, z) for x in nu])
    ax.plot(nu, e, lw=2, c='k', label='$z=1.1$') 
    
    z = 3.0
    e = np.array([qso_emissivity_hm12(x, z) for x in nu])
    ax.plot(nu, e, lw=2, c='r', label='$z=3.0$') 

    z = 4.9
    e = np.array([qso_emissivity_hm12(x, z) for x in nu])
    ax.plot(nu, e, lw=2, c='g', label='$z=4.9$') 

    z = 8.1
    e = np.array([qso_emissivity_hm12(x, z) for x in nu])
    ax.plot(nu, e, lw=2, c='b', label='$z=8.1$')

    ax.axvline(c_angPerSec/912.0, lw=1, c='k', dashes=[7,2])
    
    plt.legend(loc='lower left', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.1, scatterpoints=1)
    
    plt.savefig('e_qso.pdf'.format(z),bbox_inches='tight')
    plt.close('all')

    return
    
def check_emissivity():

    """Plot HM12 galaxy emissivity.

    This for comparison with HM12 figure 15 (right panel).  Galaxy
    emissivity is obtained by subtracting the qso emissivity from the
    published total emissivity.

    If show_qso_spectrum is set, also plots qsi emissivity.

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'comoving emissivity per log bandwidth '+
                  '[$10^{39}$~erg s$^{-1}$ cMpc$^{-3}$]', fontsize=14)
    ax.set_xlabel(r'$E$~[eV]', fontsize=14)

    ax.set_yscale('log')
    ax.set_xscale('log')
    
    ax.set_ylim(0.3, 3000.0)
    ax.set_xlim(1.0, 60.0)

    locs = (1.0, 5.0, 10.0, 50.0)
    labels = ('1', '5', '10', '50')
    plt.xticks(locs, labels)

    locs = (1.0, 10.0, 100.0, 1000.0)
    labels = ('1', '10', '100', '1000')
    plt.yticks(locs, labels)
    
    nu = np.logspace(13.0,17.0,num=10000)

    dnu = np.diff(nu)
    num = (nu[1:]+nu[:-1])/2.0
    erg_to_eV = 6.2415091e11

    dnu2 = np.diff(np.log(nu))
    print np.unique(dnu2)

    show_qso_spectrum = False
    
    z = 1.1
    e_qso = np.array([qso_emissivity_hm12(x, z) for x in num])
    e = emissivity_HM12(c_angPerSec/num, z*np.ones_like(num), grid=False)
    if show_qso_spectrum: 
        ax.plot(hplanck*num*erg_to_eV, 1.0e-39*e_qso*dnu/dnu2,
                lw=2, c='k', label='$z=1.1$', dashes=[7,2]) 
    ax.plot(hplanck*num*erg_to_eV, 1.0e-39*(e-e_qso)*dnu/dnu2,
            lw=2, c='k', label='$z=1.1$') 
    
    z = 3.0
    e_qso = np.array([qso_emissivity_hm12(x, z) for x in num])
    e = emissivity_HM12(c_angPerSec/num, z*np.ones_like(num), grid=False)
    if show_qso_spectrum: 
        ax.plot(hplanck*num*erg_to_eV, 1.0e-39*e_qso*dnu/dnu2,
                lw=2, c='r', label='$z=3.0$', dashes=[7,2]) 
    ax.plot(hplanck*num*erg_to_eV, 1.0e-39*(e-e_qso)*dnu/dnu2,
            lw=2, c='r', label='$z=3.0$') 

    z = 4.9
    e_qso = np.array([qso_emissivity_hm12(x, z) for x in num])
    e = emissivity_HM12(c_angPerSec/num, z*np.ones_like(num), grid=False)
    if show_qso_spectrum: 
        ax.plot(hplanck*num*erg_to_eV, 1.0e-39*e_qso*dnu/dnu2,
                lw=2, c='g', label='$z=4.9$', dashes=[7,2]) 
    ax.plot(hplanck*num*erg_to_eV, 1.0e-39*(e-e_qso)*dnu/dnu2,
            lw=2, c='g', label='$z=4.9$') 

    z = 8.1
    e_qso = np.array([qso_emissivity_hm12(x, z) for x in num])
    e = emissivity_HM12(c_angPerSec/num, z*np.ones_like(num), grid=False)
    if show_qso_spectrum: 
        ax.plot(hplanck*num*erg_to_eV, 1.0e-39*e_qso*dnu/dnu2,
                lw=2, c='b', label='$z=8.1$', dashes=[7,2]) 
    ax.plot(hplanck*num*erg_to_eV, 1.0e-39*(e-e_qso)*dnu/dnu2,
            lw=2, c='b', label='$z=8.1$') 
    
    ax.axvline(13.6, lw=1, c='k', dashes=[7,2])

    plt.legend(loc='lower left', fontsize=12, handlelength=3,
               frameon=False, framealpha=0.0, labelspacing=.1,
               handletextpad=0.1, borderpad=0.1, scatterpoints=1)
    
    plt.savefig('e.pdf'.format(z),bbox_inches='tight')
    plt.close('all')

    return

def check_emissivity_evolution():

    """Plot emissivity contribution from higher redshifts.

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'emissivity [erg s$^{-1}$ Hz$^{-1}$ cMpc$^{-3}$]')
    ax.set_xlabel(r'$\lambda_\mathrm{rest}$ [\AA]')

    ax.set_yscale('log')
    ax.set_xscale('log')
    
    ax.set_xlim(1.0, 1.0e4)

    ws = np.logspace(0.0, 5.0, num=200)
    zref = 1.0
    e = emissivity_HM12(ws/(1.0+zref), zref*np.ones_like(ws), grid=False)
    ax.plot(ws/(1.0+zref), e, lw=2, c='k')

    zs = np.arange(1.5, 10.)
    for z in zs:
        e = emissivity_HM12(ws/(1.0+z), z*np.ones_like(ws), grid=False)
        ax.plot(ws/(1.0+zref), e, lw=1, c='tomato')
        
    ax.axvline(1216.0, lw=1, c='k', dashes=[7,2])
    ax.axvline(912.0, lw=1, c='k', dashes=[7,2])
    ax.axvline(304.0, lw=1, c='k', dashes=[7,2])
    ax.axvline(228.0, lw=1, c='k', dashes=[7,2])
    
    plt.savefig('e_evol.pdf',bbox_inches='tight')
    plt.close('all')

    return

def check_tau_evolution():

    """Plot opacity contribution from higher redshifts.

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'$\bar\tau$')
    ax.set_xlabel(r'$\lambda_\mathrm{rest}$ [\AA]')

    ax.set_yscale('log')
    ax.set_xscale('log')
    
    ax.set_xlim(1.0, 1.0e4)

    ws = np.logspace(0.0, 5.0, num=200)
    zref = 1.0
    nu_rest = c_angPerSec*(1.0+zref)/ws 
    t = np.array([tau_eff(x, zref) for x in nu_rest])
    ax.plot(ws/(1.0+zref), t, lw=2, c='k')

    zs = np.arange(1.5, 10.)
    for z in zs:
        nu_rest = c_angPerSec*(1.0+z)/ws 
        t = np.array([tau_eff(x, z) for x in nu_rest])
        ax.plot(ws/(1.0+zref), t, lw=1, c='tomato')
        
    ax.axvline(1216.0, lw=1, c='k', dashes=[7,2])
    ax.axvline(912.0, lw=1, c='k', dashes=[7,2])
    ax.axvline(304.0, lw=1, c='k', dashes=[7,2])
    ax.axvline(228.0, lw=1, c='k', dashes=[7,2])
    
    plt.savefig('tau_evol.pdf',bbox_inches='tight')
    plt.close('all')

    return

def check_1Ry_emissivity_evolution():

    """Plot evolution of 1Ry emissivity in HM12.

    Compare with their Figure 7 (right panel). 

    """

    fig = plt.figure(figsize=(7, 7), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    ax.tick_params('both', which='major', length=7, width=1)
    ax.tick_params('both', which='minor', length=5, width=1)
    ax.tick_params('x', which='major', pad=6)

    ax.set_ylabel(r'emissivity [erg s$^{-1}$ Hz$^{-1}$ cMpc$^{-3}$]')
    ax.set_xlabel(r'$z$')

    ax.set_yscale('log')
    
    zs = np.arange(0.0, 10., 0.1)
    e = emissivity_HM12(912.0*np.ones_like(zs), zs, grid=False)
    ax.plot(zs, e, lw=2, c='k')

    
    plt.savefig('e_1ry_evol.pdf',bbox_inches='tight')
    plt.close('all')

    return


