from __future__ import absolute_import
import numpy as np

from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM

from metaldisc.fluxgrid import FluxGrid
from metaldisc.galaxy import GalaxyMap
from metaldisc.seeing import MoffatPSF
from metaldisc.obssim import BinmapObsSim, ImageObsSim
from metaldisc.fitting import MultinestFitting
import os

def init_fluxgrid():
    """Initialize emission-line model instance"""
    #select model grid to use
    grid = '../grids/grid_Dopita13_kappa=inf.h5'

    #list of line names that we will be using
    lines = ['O2-3727', 'O2-3729', 'Hg', 'Hb', 'O3-5007']
    #these names must match those defined in the model grid file
    
    #preload model flux grid
    fluxgrid = FluxGrid(grid, lines)

    return fluxgrid


def init_seeing(file_):
    """Initialize PSF model instance"""

    with fits.open(file_, mode='readonly') as fh:
        #get psf info
        psf_data = np.array(fh['psf'].data)

    seeing = MoffatPSF(psf_data['wave'], psf_data['fwhm'], psf_data['beta'])

    return seeing


def init_galaxy(file_, fluxgrid):
    """Initialize galaxy model instance"""

    with fits.open(file_, mode='readonly') as fh:
        #get basic info from primary header
        ra = fh[0].header['RA'] #ra of galaxy centre
        dec = fh[0].header['Dec'] #dec of galaxy centre
        z = fh[0].header['z'] #redshift

        inc = fh[0].header['inc'] #disc inclination
        pa = fh[0].header['pa'] #position angle (North=0, East=90)

        #get approximate SFR map to be used as a fixed model input
        sfrmap = fh['sfrmap'].copy()
        #sfrmap must be a fits ImageHDU with the correct WCS information

    #choose a cosmology used to calcuate luminosity distance
    cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

    galaxy = GalaxyMap(sfrmap, ra, dec, z, pa, inc, cosmo, fluxgrid)

    return galaxy


def init_obssim(file_, galaxy, seeing):
    """Initialize observation model instance"""

    with fits.open(file_, mode='readonly') as fh:
        #get segmentation map indicating spatial positions of the above emission
        #line fluxes 

        binmap = fh['binmap'].copy()
        #binmap must be a fits ImageHDU with the correct WCS information

    obssim = BinmapObsSim(binmap, galaxy, seeing)

    return obssim


def init_fitter(file_, obssim):


    with fits.open(file_, mode='readonly') as fh:
        #get observed fluxes and associated errors 
        flux = np.array(fh['flux'].data)
        flux_err = np.array(fh['flux_err'].data)

        #flux and flux_err are structured arrays
        #use magic to convert them to regular arrays
        flux = flux.view((flux.dtype[0], len(flux.dtype.names)))
        flux_err = flux_err.view((flux_err.dtype[0], len(flux_err.dtype.names)))


    #convert flux errors to variances
    flux_var = flux_err ** 2.

    #list of line names that we will be using
    lines = [['O2-3727', 'O2-3729'], ['Hg'], ['Hb'], ['O3-5007']]
    #len(lines) must equal the number of columns in the observed flux array

    #note here the nested list ['O2-3727', 'O2-3729'] indicates that these lines
    #are to be coadded (because this is how the observed fluxes are provided)
    
    #additional 4% error to add to model line fluxes
    model_err = np.array([0.04, 0.04, 0.04, 0.04, 0.04])


    #initialize fitter with default priors
    fitter = MultinestFitting(lines, flux, flux_var, obssim, model_err)

    return fitter



if __name__ == '__main__':
    #data file
    file_ = 'example_HDFS_0003.fit'

    #build up model from components
    fluxgrid = init_fluxgrid()
    seeing = init_seeing(file_)
    galaxy = init_galaxy(file_, fluxgrid)
    obssim = init_obssim(file_, galaxy, seeing)
    fitter = init_fitter(file_, obssim)


    #now run the fitter and store the results in the "out" directory

    #set multinest parameters
    #N.B. These are definitely NOT sensible defaults! (they are simply choosen
    #to make this example run faster)
    sampling_efficiency = 1.0 #0.3 -- 0.8 would be more sensible choices
    n_live_points = 100       #>500 would be more a sensible choice

    #create output directory if needed
    try:
        os.mkdir('out')
    except OSError:
        pass
    fitter.multinest_run('out/0-', sampling_efficiency=sampling_efficiency,
                         n_live_points=n_live_points)

    #now type
    #multinest_marginals.py out/0-
    #this produces a post-processed summary of the output: 0-stats.json
    #and tringle plots 0-marg.pdf/0-marg.png
    #
    #this also will print an output similar to 
    #parameters:
    #SFRtotal       20.6 +- 2.0
    #logZ_0         0.186 +- 0.017
    #dlogZ          -0.0274 +- 0.0025
    #logU_sol       -3.604 +- 0.029
    #tauV_0         1.806 +- 0.075

    #SFRtotal is the total star formation rate of the model in [M_sun / yr]

    #logZ_0 is the central metallicity relative to solar in [dex], where solar
    #metallicity is 12 + log(O/H) = 8.69 (for the Dopita+13 models)

    #dlogZ is the metallicity gradient in [dex / kpc]

    #logU_sol is the dimensionless ionization parameter at solar metallicity

    #tauV_0 is the dust attenuation
    

