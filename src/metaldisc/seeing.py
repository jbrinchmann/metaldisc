from __future__ import absolute_import
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import dblquad
from scipy.optimize import brentq


class CircularPSF(object):

    def _get_args(self, wave):
        """Get arguments for _func [to be implemented]

        Parameters
        ----------
        wave : array of floats
            wavelength

        Returns
        -------
        args : tuple
            arguments for corresponding _func method

        """
            
        msg = "Method '_get_args' not implemented in class {0}"
        msg = msg.format(self.__class__)

        raise NotImplementedError(msg)

    def _func(self, r, *args):
        """PSF function [to be implemented]

        Parameters
        ----------
        r : array of floats
            radius
        *args: extra arguments defined in _get_args

        Returns
        -------
        y : array of floats
            function at each radius

        """

        msg = "Method '_func' not implemented in class {0}"
        msg = msg.format(self.__class__)

        raise NotImplementedError(msg)

    def _integral(self, r, *args):
        """2D polar integral of function

        Parameters
        ----------
        r : array of floats
            radius
        *args: extra arguments defined in _get_args

        Returns
        -------
        y : array of floats
            integrated function from radius=0 to radius=r

        """
    
        gfun = lambda x: -np.sqrt(r**2. - x**2.)
        hfun = lambda x: np.sqrt(r**2. - x**2.)

        #NB dblquad expects dx and dy to be swapped
        integrand = lambda dy, dx: self._func(np.sqrt(dx**2. + dy**2.), *args)

        y, _ = dblquad(integrand, -r, r, gfun, hfun)

        return y

    def __call__(self, r, wave):
        """Calcualate the PSF at a given radii and wavelength
        
        Parameters
        ----------
        r : array of floats
            radii [arcsec]
        wave : float
            wavelength [Angstrom]

        Returns
        -------
        profile : array of floats
            Normalized PSF computed at each radius

        """

        args = self._get_args(wave)
        profile = self._func(r, *args)

        return profile

    def flux_enclosed(self, r, wave):
        """Calculate PSF enclosed within a given radius
        
        Parameters
        ----------
        r : float
            radius
        wave : float
            wavelength

        Returns
        -------
        y : fraction of flux enclosed (total = 1)

        """

        args = self._get_args(wave)
        y = self._integral(r, *args)

        return y

    def radius_enclosing(self, fraction, wave):
        """Calculate radius that encloses a given fraction of total flux
        
        Parameters
        ----------
        fraction : float
            fraction of flux enclosed (normalized to 1)
        wave : float
            wavelength

        Returns
        -------
        r : radius enclosing a given fraction of flux

        Raises
        ------
        RuntimeError : if enlosed radius is very large

        """
        
        args = self._get_args(wave)

        f = lambda r, x: self._integral(r, *args) - x
        
        try:
            r = brentq(f, 0., 20., args=(fraction,))
        except ValueError:
            raise RuntimeError("PSF is very broad") # if necessary increase upperbound in brentq function

        return r


class NonCircularPSF(object):

    def _get_args(self, wave):
        """Get arguments for _func [to be implemented]

        Parameters
        ----------
        wave : array of floats
            wavelength

        Returns
        -------
        args : tuple
            arguments for corresponding _func method

        """
            
        msg = "Method '_get_args' not implemented in class {0}"
        msg = msg.format(self.__class__)

        raise NotImplementedError(msg)


    def _func(self, dx, dy, *args):
        """PSF function [to be implemented]

        Parameters
        ----------
        dx : array of floats
            x-axis displacement
        dy : array of floats
            y-axis displacement
        *args: extra arguments defined in _get_args

        Returns
        -------
        y : array of floats
            function at each radius

        """

        msg = "Method '_func' not implemented in class {0}"
        msg = msg.format(self.__class__)

        raise NotImplementedError(msg)


    def _integral(self, r, *args):
        """2D polar integral of function

        Parameters
        ----------
        r : float
            radius
        *args: extra arguments defined in _get_args

        Returns
        -------
        y : float
            integrated function from radius=0 to radius=r

        """
    
        gfun = lambda x: -np.sqrt(r**2. - x**2.)
        hfun = lambda x: np.sqrt(r**2. - x**2.)

        #NB dblquad expects dx and dy to be swapped
        integrand = lambda dy, dx: self._func(dx, dy, *args)

        y, _ = dblquad(integrand, -r, r, gfun, hfun)

        return y

    def __call__(self, dx, dy, wave):
        """Calcualate the PSF at a given radii and wavelength
        
        Parameters
        ----------
        dx : array of floats
            x-axis displacement
        dy : array of floats
            y-axis displacement
        wave : float
            wavelength [Angstrom]

        Returns
        -------
        profile : array of floats
            Normalized PSF computed at each radius

        """

        args = self._get_args(wave)
        profile = self._func(dx, dy, *args)

        return profile

    def flux_enclosed(self, r, wave):
        """Calculate PSF enclosed within a given radius
        
        Parameters
        ----------
        r : float
            radius
        wave : float
            wavelength

        Returns
        -------
        y : fraction of flux enclosed (total = 1)

        """

        args = self._get_args(wave)
        y = self._integral(r, *args)

        return y

    def radius_enclosing(self, fraction, wave):
        """Calculate radius that encloses a given fraction of total flux
        
        Parameters
        ----------
        fraction : float
            fraction of flux enclosed (normalized to 1)
        wave : float
            wavelength

        Returns
        -------
        r : radius enclosing a given fraction of flux

        Raises
        ------
        RuntimeError : if enlosed radius is very large

        """
        
        args = self._get_args(wave)

        f = lambda r, x: self._integral(r, *args) - x
        
        try:
            r = brentq(f, 0., 20., args=(fraction,))
        except ValueError:
            raise RuntimeError("PSF is very broad") # if necessary increase upperbound in brentq function

        return r


#FUTURE:
#class VariablePSF(object):
#
#    def __call__(self, dx, dy, x0, y0, wave):
#        """Calcualate the PSF at a given distance, position and wavelength"""
#
#        msg = "Method '__call__' not implemented in class {0}".format(self.__class__)
#        raise NotImplementedError(msg)
#
#    def radius_enclosing(self, fraction, x0, y0, wave):
#        """Calculate radius that encloses a given fraction of total flux at a given position"""
#
#        msg = "Method 'radius_enclosing' not implemented in class {0}".format(self.__class__)
#        raise NotImplementedError(msg)


def gauss_fwhm_to_sigma(fwhm):
    """Convert FWHM to standard deviation
    
    Parameters
    ----------
    fwhm : float
        Full-Width Half Max

    Returns
    -------
    sigma : float
        Standard deviation of Gaussian

    """
    sigma = fwhm / (2. * np.sqrt(2. * np.log(2.)))
    return sigma


def moffat_fwhm_to_alpha(fwhm, beta):
    """Convert FWHM to Moffat alpha parameter
    
    Parameters
    ----------
    fwhm : float
        Full-Width Half Max
    beta : float
        Moffat beta parameter

    Returns
    -------
    alpha : float
        Corresponding Moffat alpha parameter

    """
    alpha = fwhm / (2. * np.sqrt(2.**(1./beta) - 1.))
    return alpha


class GaussianPSF(CircularPSF):

    def __init__(self, wave, fwhm):

        """Gaussian point spread function

        Performs linear interpolation of FWHM parameter as a function of
        wavelength

        Parameters
        ----------
        wave : array of floats
            wavelength [Angstrom], independent varible for interpolation
        fwhm : array of floats
            full-width half max [arcsec], dependent varible for interpolation
        
        """
        self.wave = wave
        self.fwhm = fwhm
    
        self.interp_fwhm = interp1d(self.wave, self.fwhm, copy=True)


    def _get_args(self, wave):

        fwhm = self.interp_fwhm(wave)
        sigma = gauss_fwhm_to_sigma(fwhm)

        args = (sigma,)

        return args


    def _func(self, r, sigma):
        """Gaussian function

        Parameters
        ----------
        r : array of floats
            radius
        sigma : float
            Standard deviation

        Returns
        -------
        y : array of floats
            Gaussian function at each radius

        """

        norm = 1. / (2. * np.pi * sigma**2.)
        y = norm * np.exp(-0.5 * (r/sigma)**2.)

        return y


    # overload _integral because can be calculated analytically
    def _integral(self, r, sigma):
        """2D polar integral of Gaussian function

        Parameters
        ----------
        r : array of floats
            radius
        sigma : float
            Standard deviation

        Returns
        -------
        y : array of floats
            integrated Gaussian function from r=0 to r=radius

        """
    
        y =  1. - np.exp(-0.5 * (r/sigma)**2.)

        return y


class MoffatPSF(CircularPSF):

    def __init__(self, wave, fwhm, beta):
        """Moffat point spread function

        Performs linear interpolation of FWHM and Beta parameters as a function
        of wavelength

        Parameters
        ----------
        wave : array of floats
            wavelength [Angstrom], independent varible for interpolation
        fwhm : array of floats
            full-width half max [arcsec], dependent varible for interpolation
        beta : array of floats
            Moffat beta parameter, dependent varible for interpolation
        
        """
        self.wave = wave
        self.fwhm = fwhm
        self.beta = beta
    
        self.interp_fwhm = interp1d(self.wave, self.fwhm, copy=True)
        self.interp_beta = interp1d(self.wave, self.beta, copy=True)

    def _get_args(self, wave):

        fwhm = self.interp_fwhm(wave)
        beta = self.interp_beta(wave)

        alpha = moffat_fwhm_to_alpha(fwhm, beta)

        args = (alpha, beta)

        return args


    def _func(self, r, alpha, beta):
        """Moffat function

        Parameters
        ----------
        r : array of floats
            radius
        alpha : float
            Corresponding Moffat alpha parameter
        beta : float
            Moffat beta parameter

        Returns
        -------
        y : array of floats
            Moffat function at each radius

        """

        y = (beta-1.) / (np.pi*alpha**2.) * (1. + (r/alpha)**2.)**(-beta)

        return y


    # overload _integral because can be calculated analytically
    def _integral(self, r, alpha, beta):
        """2D polar integral of Moffat function

        Parameters
        ----------
        r : array of floats
            radius
        alpha : float
            Corresponding Moffat alpha parameter
        beta : float
            Moffat beta parameter

        Returns
        -------
        y : array of floats
            integrated Moffat function from r=0 to r=radius

        """

        #Thanks WolframAlpha!
        y = 1. - ((1.+(r/alpha)**2.) / (1. + (r/alpha)**2.)**beta)

        return y


class EllipticalGaussianPSF(NonCircularPSF):

    def __init__(self, wave, fwhm_a, fwhm_b, pa):

        """Elliptical Gaussian point spread function

        Performs linear interpolation of FWHM parameter as a function of
        wavelength

        Parameters
        ----------
        wave : array of floats
            wavelength [Angstrom], independent varible for interpolation
        fwhm_a : array of floats
            major axis full-width half max [arcsec], dependent varible for
            interpolation
        fwhm_b : array of floats
            minor axis full-width half max [arcsec], dependent varible for
            interpolation
        pa : array of floats
            position angle of major axis (North = 0, East = 90) [degrees],
            dependent varible for interpolation
        
        """
        self.wave = wave
        self.fwhm_a = fwhm_a
        self.fwhm_b = fwhm_b
        self.pa = pa
    
        self.interp_fwhm_a = interp1d(self.wave, self.fwhm_a, copy=True)
        self.interp_fwhm_b = interp1d(self.wave, self.fwhm_b, copy=True)
        self.interp_pa = interp1d(self.wave, self.pa, copy=True)


    def _get_args(self, wave):

        fwhm_a = self.interp_fwhm_a(wave)
        fwhm_b = self.interp_fwhm_b(wave)

        sigma_a = gauss_fwhm_to_sigma(fwhm_a)
        sigma_b = gauss_fwhm_to_sigma(fwhm_b)

        pa = self.interp_pa(wave)

        args = (sigma_a, sigma_b, pa)

        return args


    def _func(self, dx, dy, sigma_a, sigma_b, pa):
        """Elliptical Gaussian function

        Parameters
        ----------
        dx : array of floats
            x-axis displacement
        dy : array of floats
            y-axis displacement
        sigma_a : float
            major axis standard deviation
        sigma_b : float
            minor axis standard deviation
        pa : float
            position angle of major axis
            

        Returns
        -------
        y : array of floats
            Gaussian function at each dx, dy position

        """

        theta = np.radians(90. + pa)
        da = dx * np.cos(theta) - dy * np.sin(theta)
        db = dx * np.sin(theta) + dy * np.cos(theta)

        norm_a = 1. / np.sqrt(2. * np.pi * sigma_a**2.)
        norm_b = 1. / np.sqrt(2. * np.pi * sigma_b**2.)
        y = (norm_a * norm_b *
             np.exp(-0.5 * ((da/sigma_a)**2. + (db/sigma_b)**2.)))

        return y
