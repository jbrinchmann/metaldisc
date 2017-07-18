import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import brentq

class CircularPSF(object):

    def __call__(self, r, wave):
        """Calcualate the PSF at a given radii and wavelength"""

        msg = "Method '__call__' not implemented in class {0}".format(self.__class__)
        raise NotImplementedError(msg)

    def radius_enclosing(self, fraction, wave):
        """Calculate radius that encloses a given fraction of total flux"""

        msg = "Method 'radius_enclosing' not implemented in class {0}".format(self.__class__)
        raise NotImplementedError(msg)


class EllipticalPSF(object):

    def __call__(self, dx, dy, wave):
        """Calcualate the PSF at a given distance and wavelength"""

        msg = "Method '__call__' not implemented in class {0}".format(self.__class__)
        raise NotImplementedError(msg)

    def dist_enclosing(self, fraction, wave):
        """Calculate distance that encloses a given fraction of total flux"""

        msg = "Method 'dist_enclosing' not implemented in class {0}".format(self.__class__)
        raise NotImplementedError(msg)


class VariablePSF(object):

    def __call__(self, dx, dy, x0, y0, wave):
        """Calcualate the PSF at a given distance, position and wavelength"""

        msg = "Method '__call__' not implemented in class {0}".format(self.__class__)
        raise NotImplementedError(msg)

    def dist_enclosing(self, fraction, x0, y0, wave):
        """Calculate distance that encloses a given fraction of total flux at a given position"""

        msg = "Method 'dist_enclosing' not implemented in class {0}".format(self.__class__)
        raise NotImplementedError(msg)


def fwhm_to_sigma(fwhm):
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


class GaussianSeeing(CircularPSF):

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


    def __call__(self, r, wave):
        """Calcualate the Gaussian PSF at a given radii and wavelength
        
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

        fwhm = self.interp_fwhm(wave)
        sigma = fwhm_to_sigma(fwhm)

        profile = self._gaussian_func(r, sigma)
        return profile


    @staticmethod


    @staticmethod
    def _gaussian_func(r, sigma):
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


    @staticmethod
    def _gaussian_integral(r, sigma):
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

        fwhm = self.interp_fwhm(wave)
        sigma = fwhm_to_sigma(fwhm)

        y = self._gaussian_integral(r, sigma)
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
        fwhm = self.interp_fwhm(wave)
        sigma = fwhm_to_sigma(fwhm)
        
        f = lambda r, sigma, x: self._gaussian_integral(r,sigma) - x
        
        try:
            r = brentq(f, 0., 20., args=(sigma, fraction))
        except ValueError:
            raise RuntimeError("PSF is very broad") # if necessary increase upperbound in brentq function

        return r


class MoffatSeeing(CircularPSF):
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


    def __call__(self, r, wave):
        """Calcualate the Moffat PSF at a given radii and wavelength
        
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

        fwhm = self.interp_fwhm(wave)
        beta = self.interp_beta(wave)
        alpha = self.fwhm_to_alpha(fwhm, beta)

        profile = self._moffat_func(r, alpha, beta)
        return profile


    @staticmethod
    def fwhm_to_alpha(fwhm, beta):
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


    @staticmethod
    def _moffat_func(r, alpha, beta):
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


    @staticmethod
    def _moffat_integral(r, alpha, beta):
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
        #Thanks Wolfram!
        y = 1. - ((1.+(r/alpha)**2.) / (1. + (r/alpha)**2.)**beta)
        return y


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

        fwhm = self.interp_fwhm(wave)
        beta = self.interp_beta(wave)
        alpha = self.fwhm_to_alpha(fwhm, beta)

        y = self._moffat_integral(r, alpha, beta)
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
        fwhm = self.interp_fwhm(wave)
        beta = self.interp_beta(wave)
        alpha = self.fwhm_to_alpha(fwhm, beta)
        
        f = lambda r, alpha, beta, x: self._moffat_integral(r,alpha,beta) - x
        
        try:
            r = brentq(f, 0., 20., args=(alpha, beta, fraction))
        except ValueError:
            raise RuntimeError("PSF is very broad") # if necessary increase upperbound in brentq function

        return r


class EllipticalGaussianSeeing(EllipticalPSF):

    def __init__(self, wave, fwhm_a, fwhm_b, pa):

        """Gaussian point spread function

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


    def __call__(self, dx, dy, wave):
        """Calcualate the Gaussian PSF at a given radii and wavelength
        
        Parameters
        ----------
        dx : array of floats
            x-axis displacement [arcsec]
        dy : array of floats
            y-axis displacement [arcsec]
        wave : float
            wavelength [Angstrom]

        Returns
        -------
        profile : array of floats
            Normalized PSF computed at each radius

        """

        fwhm_a = self.interp_fwhm_a(wave)
        fwhm_b = self.interp_fwhm_b(wave)
        pa = self.interp_pa(wave)

        sigma_a = fwhm_to_sigma(fwhm)
        sigma_b = fwhm_to_sigma(fwhm)

        profile = self._gaussian_func(dx, dy, sigma_a, sigma_b, pa)
        return profile


    @staticmethod
    def _gaussian_func(dx, dy, sigma_a, sigma_b, pa):
        """Gaussian function

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
            Gaussian function at each radius

        """

        theta = np.radians(90. + pa)
        da = dx * np.cos(theta) - dy * np.sin(theta)
        db = dx * np.sin(theta) + dy * np.cos(theta)

        norm_a = 1. / (2. * np.pi * sigma_a**2.)
        norm_b = 1. / (2. * np.pi * sigma_b**2.)
        y = (norm_a * norm_b *
             np.exp(-0.5 * ((da/sigma_a)**2. + (db/sigma_b)**2.)))

        return y


    @staticmethod
    def _gaussian_integral(r, sigma):
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

        fwhm = self.interp_fwhm(wave)
        sigma = fwhm_to_sigma(fwhm)

        y = self._gaussian_integral(r, sigma)
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
        fwhm = self.interp_fwhm(wave)
        sigma = fwhm_to_sigma(fwhm)
        
        f = lambda r, sigma, x: self._gaussian_integral(r,sigma) - x
        
        try:
            r = brentq(f, 0., 20., args=(sigma, fraction))
        except ValueError:
            raise RuntimeError("PSF is very broad") # if necessary increase upperbound in brentq function

        return r

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    wave = np.array([4750., 7000., 9300.])
    fwhm = np.array([0.76, 0.66, 0.61])
    beta = np.array([2.6, 2.6, 2.6])
    seeing = MoffatSeeing(wave, fwhm, beta)
#    seeing = GaussianSeeing(wave, fwhm)

    r = np.arange(0, 10., 0.01)
    y_5000 = [seeing.flux_enclosed(i, 5000) for i in r]
    y_7000 = [seeing.flux_enclosed(i, 7000) for i in r]
    y_9000 = [seeing.flux_enclosed(i, 9000) for i in r]

    plt.plot(r, y_5000, 'k')
    plt.plot(r, y_7000, 'k') 
    plt.plot(r, y_9000, 'k')
    plt.axhline(0.995)
    plt.axvline(seeing.radius_enclosing(0.995, 5000))
    plt.axvline(seeing.radius_enclosing(0.995, 7000))
    plt.axvline(seeing.radius_enclosing(0.995, 9000))

    plt.show()
    
