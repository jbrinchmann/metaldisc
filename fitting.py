from collections import OrderedDict
import json

import numpy as np
from scipy.special import gammaln

import matplotlib.pyplot as plt
from matplotlib import gridspec

import pymultinest

import galaxy


def linear_prior(low, upp):
    """Linear prior

    Maps x [0-1] uniformly between low and upp

    Parameters
    ----------
    low : float
        lower limit of range
    upp : float
        upper limit of range

    Returns
    -------
    y : function that takes one argument
        x : float [between 0 and 1]
            percentile

    """

    y = lambda x: (upp-low) * x + low
    return y


def logarithmic_prior(low, upp):
    """Logarithmic prior

    Maps x [0-1] logarithmically between low and upp

    Parameters
    ----------
    low : float
        lower limit of range
    upp : float
        upper limit of range

    Returns
    -------
    y : function that takes one argument
        x : float [between 0 and 1]
            percentile

    """

    l_low = np.log10(low)
    l_upp = np.log10(upp)
    y = lambda x: 10. ** ((l_upp-l_low) * x + l_low)
    return y


class MultinestFitting(object):

    def __init__(self, lines, flux, var, obssim, model_err,
            likelihood='student', dlogZ_prior_range=(-0.5, 0.5), **kwargs):
        """Fitting object for ObsSim and data

        Parameters
        ----------
        lines : list of strings
            list of strings identifing emission lines
            lines to be coadded should be nested in a 2nd list
            e.g. lines = [['O2-3727', 'O2-3729'], 'Hg', 'Hb', 'O3-5007']
        flux : (Nbins*Nlines) array of floats
            line fluxes in [10^-20 erg/s/cm^2], Nlines = len(lines)
        var : (Nbins*Nlines) array of floats
            associated variances of flux
        obssim : instance of BaseObsSim or subclass
            model to be fitted
        model_err : array of floats
            systematic error to add to model fluxes
            array must match flattened line list
        likelihood : string [default='student']
            type of likelihood function to use
            (options: 'student', 'gaussian')

        Keywords
        --------
        dof : int [default=3]
            if using likelihood='student', set the Degrees of Freedom to use

        """
        
        if flux.shape != var.shape:
            raise Exception("flux and var arrays must be the same shape")

        if flux.shape[1] != len(lines):
            raise Exception("2nd dimension of flux must match len(lines)")

        if likelihood == 'student':
            self.loglikelihood = self._student_loglikelihood
            try:
                self.dof = float(kwargs['dof'])
            except KeyError:
                self.dof = 3.
        elif likelihood == 'gaussian':
            self.loglikelihood = self._gaussian_loglikelihood
        else:
            raise Exception("likelihood function '%s' unknown" % likelihood)
        
        self.dlogZ_prior_range = np.array(dlogZ_prior_range)


        self.obs_flux = flux
        self.obs_var = var

        self.obssim = obssim
        
        self.params = OrderedDict()
        if type(obssim.galaxy) == galaxy.GalaxyDisc:
            self.init_galdisc_params()
        elif type(obssim.galaxy) == galaxy.GalaxyDiscFixedrd:
            self.init_galdisc_fixedrd_params()
        elif type(obssim.galaxy) == galaxy.GalaxyMap:
            self.init_galmap_params()
        else:
            raise Exception('Galaxy model %s unknown' % str(obssim.galaxy))
        self.init_basegal_params()
            

        lines, line_mapping = self.parse_lines(lines)
        if len(model_err) != line_mapping.shape[1]:
            raise Exception("len(model_err) must equal number of lines")

        self.lines = lines

        self.line_mapping_flux = line_mapping
        self.line_mapping_var = (line_mapping * model_err) ** 2.


    @staticmethod
    def parse_lines(lines):
        """Given a nested line list create a flat list of lines and associated mapping matrix
        
        """


        #flatten linelist and MAKE IT UNIQUE
        lines_flat = OrderedDict()
        for line_components in lines:
            for l in line_components:
                lines_flat[l] = None
        lines_flat = tuple(lines_flat.keys())

        mapping = np.zeros([len(lines),len(lines_flat)], dtype=float)
        for i_out, line_components in enumerate(lines):
            for l in line_components:
                j_in = lines_flat.index(l)
                mapping[i_out,j_in] = 1.


        return lines_flat, mapping


    def _student_loglikelihood(self, obs_flux, model_flux, obs_var, model_var):
        """Student log-likelihood function
        
        Parameters
        ----------
        obs_flux : array of floats
            observed flux
        model_flux : array of floats
            model flux
        obs_var : array of floats
            observed variance
        model_var : array of floats
            model variance

        Returns
        -------
        out : float
            log-likelihood

        """

        res = obs_flux - model_flux
        var = obs_var + model_var

        dof = self.dof

        scale_sqd = (dof-2.) * var / dof
        out = ((gammaln((dof+1.)/2.) - gammaln(dof/2.)) *
               np.ones_like(var, dtype=float))
        out -= 0.5 * np.log(np.pi * dof * scale_sqd)
        out -= (dof+1.)/2. * np.log(1. + 1./dof * res**2. / scale_sqd)
        out = out.sum()

        return out


    def _gaussian_loglikelihood(self, obs_flux, model_flux, obs_var, model_var):
        """Gaussian log-likelihood function
        
        Parameters
        ----------
        obs_flux : array of floats
            observed flux
        model_flux : array of floats
            model flux
        obs_var : array of floats
            observed variance
        model_var : array of floats
            model variance

        Returns
        -------
        out : float
            log-likelihood

        """
        res = obs_flux - model_flux
        var = obs_var + model_var

        n = res.size
        out = -n/2. * np.log(2.*np.pi)
        out -= np.log(var).sum() / 2.
        out -= 1./2. * ((res**2.) / var).sum()
        
        return out


    def init_galdisc_params(self):

        self.params['SFRtotal'] = logarithmic_prior(0.01, 100)
        self.params['r_d'] = linear_prior(0., 2.)

    def init_galdisc_fixedrd_params(self):

        self.params['SFRtotal'] = logarithmic_prior(0.01, 100)

    def init_galmap_params(self):

        self.params['SFRtotal'] = logarithmic_prior(0.01, 100)
    
    def init_basegal_params(self):

        #get fluxgrid range in logZ and logU
        fluxgrid = self.obssim.galaxy.fluxgrid

        logZ_min = fluxgrid.logZ_min
        logZ_max = fluxgrid.logZ_max

        logU_min = fluxgrid.logU_min
        logU_max = fluxgrid.logU_max

        logU_0_max = logU_max + 0.8 * logZ_max
        logU_0_min = logU_min + 0.8 * logZ_min

        self.params['logZ_0'] = linear_prior(logZ_min, logZ_max)
        dlogZ_min, dlogZ_max = self.dlogZ_prior_range
        self.params['dlogZ'] = linear_prior(dlogZ_min, dlogZ_max)
        self.params['logU_sol'] = linear_prior(logU_0_min, logU_0_max)
        self.params['tauV_0'] = linear_prior(0., 4.)
        
    
    def multinest_prior(self, cube, ndim, nparams):

        for i_param, prior_func in enumerate(self.params.itervalues()):
            cube[i_param] = prior_func(cube[i_param])


    def cube_to_params(self, cube):
        
        params = OrderedDict(zip(self.params, cube))

        return params


    def multinest_loglike(self, cube, ndim, nparams):
        model_flux, model_var = self.model(cube)

        loglike = self.loglikelihood(self.obs_flux, model_flux,
                                     self.obs_var, model_var)
       
        return loglike


    def model(self, cube):
        params = self.cube_to_params(cube)

        flux = self.obssim(self.lines, params)
        flux /= 1e-20

        model_flux = np.dot(self.line_mapping_flux, flux.T).T
        model_var = np.dot(self.line_mapping_var, (flux**2.).T).T

        return model_flux, model_var


    def multinest_run(self, basename):
        parameters = [i for i in self.params.iterkeys()]
        n_params = len(parameters)

        pymultinest.run(self.multinest_loglike, self.multinest_prior, n_params,
                        outputfiles_basename=basename,
                        importance_nested_sampling=False, multimodal=True,
                        resume=False, verbose=True, n_live_points=1000,
                        sampling_efficiency='parameter')

        # save parameter names
        json.dump(parameters, open(basename+'params.json', 'w'))
        
