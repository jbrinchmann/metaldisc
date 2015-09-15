from collections import OrderedDict
import json

import numpy as np
from scipy.special import gammaln

import matplotlib.pyplot as plt
from matplotlib import gridspec

import pymultinest


def linear_prior(x, low, upp):
    """Linear prior

    Maps x [0-1] uniformly between low and upp

    Parameters
    ----------
    x : float [between 0 and 1]
        percentile
    low : float
        lower limit of range
    upp : float
        upper limit of range

    Returns
    -------
    y : float

    """

    y = (upp-low) * x + low
    return y


def logarithmic_prior(x, low, upp):
    """Logarithmic prior

    Maps x [0-1] logarithmically between low and upp

    Parameters
    ----------
    x : float [between 0 and 1]
        percentile
    low : float
        lower limit of range
    upp : float
        upper limit of range

    Returns
    -------
    y : float

    """

    l_low = np.log10(low)
    l_upp = np.log10(upp)
    y = 10. ** ((l_upp-l_low) * x + l_low)
    return y


class MultinestFitting(object):

    def __init__(self, lines, flux, var, obssim, model_err, likelihood='student',
            **kwargs):
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
        

        self.obs_flux = flux
        self.obs_var = var

        self.obssim = obssim


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
        lines_flat = []
        coadd = []
        
        for i_out, line in enumerate(lines):
            if type(line) in [list, tuple]:
                lines_flat += line
                coadd += [i_out]*len(line)
            elif type(line) == str:
                lines_flat.append(line)
                coadd.append(i_out)
            else:
                raise Exception("Line names should be strings")

        lines_flat = tuple(lines_flat) #convert list to tuple
        mapping = np.zeros([len(lines),len(lines_flat)], dtype=float)
        for j_in, i_out in enumerate(coadd):
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

    
    def multinest_prior(self, cube, ndim, nparams):
        #SFRtotal
        cube[0] = logarithmic_prior(cube[0], 0.01, 100.)

        #r_d
        cube[1] = linear_prior(cube[1], 0., 2.)

        #get fluxgrid range in logZ and logU
        fluxgrid = self.obssim.galaxy.fluxgrid
        logZ_min = fluxgrid.logZ_min
        logZ_max = fluxgrid.logZ_max
        logZ_solar = fluxgrid.logZ_solar

        logU_min = fluxgrid.logU_min
        logU_max = fluxgrid.logU_max
        
        #Z_in
        cube[2] = linear_prior(cube[2], logZ_min, logZ_max)

        #Z_out
        cube[3] = linear_prior(cube[3], logZ_min, logZ_max)

        #logU_0
#        print "CHECK THIS"
        logU_0_max = logU_max + 0.8 * (logZ_max-logZ_solar)
        logU_0_min = logU_min + 0.8 * (logZ_min-logZ_solar)

#        print logZ_min, logZ_max
#        print logU_min, logU_max
#        print logU_0_min, logU_0_max
        cube[4] = linear_prior(cube[4], logU_0_min, logU_0_max)

        #tauV_in
        cube[5] = linear_prior(cube[5], 0., 4.)

        #tauV_out
        cube[6] = linear_prior(cube[6], 0., 4.)


    @staticmethod
    def cube_to_params(cube):
        params = OrderedDict()
        params['SFRtotal'] = cube[0]
        params['r_d'] = cube[1]
        params['Z_in'] = cube[2]
        params['Z_out'] = cube[3]
        params['logU_0'] = cube[4]
        params['tauV_in'] = cube[5]
        params['tauV_out'] = cube[6]
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
        parameters = ["SFRtotal", "r_d", "Z_in", 'Z_out', "logU_0",
                      "tauV_in", "tauV_out"]
        n_params = len(parameters)

        pymultinest.run(self.multinest_loglike, self.multinest_prior, n_params,
                        outputfiles_basename=basename,
                        importance_nested_sampling=False, multimodal=True,
                        resume=False, verbose=True, n_live_points=1000,
                        sampling_efficiency='parameter')

        # save parameter names
        json.dump(parameters, open(basename+'params.json', 'w'))
        
