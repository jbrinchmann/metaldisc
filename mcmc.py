import json
import numpy as np
import pymultinest
import matplotlib.pyplot as plt
from matplotlib import gridspec


class MCMC(object):

    def __init__(self, galaxy, obsSim, data):
        
        self.galaxy = galaxy
        self.obsSim = obsSim
        self.data = data
        
    def prior(self, cube, ndim, nparams):
        cube[0] = 10.**(4. * cube[0] - 2.) #SFR log 0.01 - 100
        cube[1] = 4. * cube[1] + 0. #r_d lin 0 - 4
        cube[2] = 2. * cube[2] + 7.5 #Z_in lin 7.9 - 9.3
        cube[3] = 2. * cube[3] + 7.5 #Z_out lin 7.9 - 9.3
        cube[5] = 10.**(2.602059991 * cube[4] + -2.) #tauV_in log 0.01-4
        cube[6] = 10.**(2.602059991 * cube[5] + -2.) #tauV_out log 0.01-4

    def loglike(self, cube, ndim, nparams):

        self.galaxy.sf_density.total_SFR = cube[0]
        self.galaxy.sf_density.r_d = cube[1]
        self.galaxy.metallicity.r_d = cube[1]
        self.galaxy.extinction.r_d = cube[1]
        self.galaxy.metallicity.Z_in = cube[2]
        self.galaxy.metallicity.Z_out = cube[3]
        self.galaxy.extinction.tauV_in = cube[4]
        self.galaxy.extinction.tauV_out = cube[5]


        model = self.obsSim.model()
        model['flux'] *= 1e20
        model['var'] *= 1e40


        res = self.data['flux'] - model['flux']
        var = self.data['var'] + model['var']
    
        norm = 1. / (2.*np.pi*var) ** 0.5
        likelihood = norm * np.exp(-0.5 * res**2. / var)

        return np.log(likelihood).sum()

    def plot(self, cube):
        self.galaxy.sf_density.total_SFR = cube[0]
        self.galaxy.sf_density.r_d = cube[1]
        self.galaxy.metallicity.r_d = cube[1]
        self.galaxy.extinction.r_d = cube[1]
        self.galaxy.metallicity.Z_in = cube[2]
        self.galaxy.metallicity.Z_out = cube[3]
        self.galaxy.extinction.tauV_in = cube[4]
        self.galaxy.extinction.tauV_out = cube[5]

        model = self.obsSim.model()
        model['flux'] *= 1e20
        model['var'] *= 1e40


        n_x = int(np.ceil(len(self.data)/2.))

        fig = plt.figure(figsize=(2.5*n_x,4.), tight_layout=True)
        gs = gridspec.GridSpec(n_x,2)

        for i in xrange(len(self.data)):
            ax = fig.add_subplot(gs[i/2,i%2])
            line_obs = self.data[i]
            line_model = model[i]
            x = np.arange(len(line_obs['flux'])) + 1
            ax.errorbar(x, line_obs['flux'], yerr=np.sqrt(line_obs['var']),
                        color='k', capsize=0, ls='', marker='o', label='Data')
            ax.errorbar(x, line_model['flux'], yerr=np.sqrt(line_model['var']),
                        color='r', capsize=0, ls='', marker='o', label='Model')
            plot_name = {'OII':r'$\left[\textrm{O}\textsc{ii}\right]$',
                         'OIII_5007':r'$\left[\textrm{O}\textsc{iii}\right]$',
                         'H_BETA':r'$\textrm{H}_\beta$',
                         'H_GAMMA':r'$\textrm{H}_\gamma$'}
            plot_name = plot_name[line_obs['name']]
            ax.text(0.15, 0.1, plot_name, va='bottom', ha='left',
                    transform=ax.transAxes, fontsize=18)
#            ax.set_title(line_obs['name'])
            if i in [2,3]:
                ax.set_xlabel('Annular bin')
            if i in [0,2]:
                ax.set_ylabel(r'$\textrm{Flux} \left[10^{-20} \textrm{erg}/\textrm{s}/\textrm{cm}^2\right]')
            
            if i == 0:
                ax.legend()
        
            ax.set_xlim([0, None])
        return fig,  model



    def run(self, outputfiles_basename):
        parameters = ["totalSFR", "r_d", "Z_in", 'Z_out', 'tauV_in', 'tauV_out']
        n_params = len(parameters)

        pymultinest.run(self.loglike, self.prior, n_params,
                        outputfiles_basename=outputfiles_basename,
                        resume=False, verbose=True, n_live_points=600,
                        sampling_efficiency='parameter')
        json.dump(parameters, open(outputfiles_basename+'params.json', 'w')) # save parameter names



if __name__ == '__main__':
    
    from galaxy import Galaxy
    from obssim import ObsSim
    from SFdensity import ExpDisc
    from metallicity import LinearMetallicity
    from extinction import LinearExtinction
    from lineflux import EmpiricalLineFlux
    from seeing import MoffatSeeing

    from astropy.cosmology import FlatLambdaCDM

    sf_density = ExpDisc(1., 1.)
    metallicity = LinearMetallicity(1., 8.5, 8.5)
    extinction = LinearExtinction(1., 8.5, 8.5)
    filename = '/data2/MUSE/metallicity_calibration/flux_cal_singlevar.h5'
    lineflux = EmpiricalLineFlux(filename,
            ['OIII_5007', 'OIII_4959', 'H_BETA', 'H_GAMMA', 'H_DELTA', 'OII'],
            [5007., 4959., 4861., 4340., 4102., 3727.])

    cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

    galaxy = Galaxy(0.564497, cosmo, 4., 76.6-90., 18.94, 45,
                    sf_density, metallicity, lineflux,  extinction)

    wave = np.array([4750., 7000., 9300.])
    fwhm = np.array([0.76, 0.66, 0.61])
    beta = np.array([2.6, 2.6, 2.6])
    moffat = MoffatSeeing(wave, fwhm, beta)

    obsSim = ObsSim(galaxy, 338.2239, -60.560436, '/data2/MUSE/metallicity_analysis/spectra_extraction/0003/binmap-wcs.fits', moffat)

    import pyfits 
    from astro.sdss_utils import scale_fiber_errors

    fibers = pyfits.getdata('/data2/MUSE/metallicity_analysis/spectra_extraction/0003/binmap_platefit.fit')
    fibers = np.array(fibers)
    fibers = scale_fiber_errors(fibers, method='normal')

    dt = np.dtype([('flux', float, len(fibers)),
                   ('var', float, len(fibers))])
    ['OIII_5007', 'OIII_4959', 'H_BETA', 'H_GAMMA', 'H_DELTA', 'OII']
    data = np.zeros(6, dtype=dt)
    data['flux'][0] = fibers['OIII_5007_FLUX']
    data['var'][0] = fibers['OIII_5007_FLUX_ERR'] ** 2.

    data['flux'][1] = fibers['OIII_4959_FLUX']
    data['var'][1] = fibers['OIII_4959_FLUX_ERR'] ** 2.

    data['flux'][2] = fibers['H_BETA_FLUX']
    data['var'][2] = fibers['H_BETA_FLUX_ERR'] ** 2.

    data['flux'][3] = fibers['H_GAMMA_FLUX']
    data['var'][3] = fibers['H_GAMMA_FLUX_ERR'] ** 2.

    data['flux'][4] = fibers['H_DELTA_FLUX']
    data['var'][4] = fibers['H_DELTA_FLUX_ERR'] ** 2.

    OII = fibers['OII_3726_FLUX'] + fibers['OII_3729_FLUX'] 
    OII_var = fibers['OII_3726_FLUX_ERR']**2. + fibers['OII_3729_FLUX_ERR']**2.
    data['flux'][5] = OII
    data['var'][5] = OII_var

    print data['flux'] / np.sqrt(data['var'])


    mcmc = MCMC(galaxy, obsSim, data)

    
    cube = [0.488769752326064211E+01,
            0.328284375884051072e+00,
            0.912771062978907644E+01,
            0.899735891743846139E+01,
            0.156811471521022350E+01,
            0.120431183675847819E+01]

    fig = mcmc.plot(cube)
    plt.show()
    fig.close()
    
    mcmc.run(outputfiles_basename='/data1/chains/1-')
