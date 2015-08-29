import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import newton

class MoffatSeeing(object):

    def __init__(self, wave, fwhm, beta):
        self.wave = wave
        self.fwhm = fwhm
        self.beta = beta
    
        self.interp_fwhm = interp1d(self.wave, self.fwhm, copy=True)
        self.interp_beta = interp1d(self.wave, self.beta, copy=True)

    def __call__(self, r, wave):
        fwhm = self.interp_fwhm(wave)
        beta = self.interp_beta(wave)
        alpha = self.fwhm_to_alpha(fwhm, beta)

        profile = self._moffat_func(r, alpha, beta)
        return profile

    @staticmethod
    def fwhm_to_alpha(fwhm, beta):
        alpha = fwhm / (2. * np.sqrt(2.**(1./beta) - 1.))
        return alpha

    @staticmethod
    def _moffat_func(r, alpha, beta):
        y = (beta-1.) / (np.pi*alpha**2.) * (1. + (r/alpha)**2.)**(-beta)
        return y

    @staticmethod
    def _moffat_integral(r, alpha, beta):
        #2d polar integral of moffat function, thanks Wolfram!
        y = 1. - ((1.+(r/alpha)**2.) / (1. + (r/alpha)**2.)**beta)
        return y

    def flux_enclosed(self, r, wave):
        fwhm = self.interp_fwhm(wave)
        beta = self.interp_beta(wave)
        alpha = self.fwhm_to_alpha(fwhm, beta)

        y = self._moffat_integral(r, alpha, beta)
        return y

    def radius_enclosing(self, fraction, wave):
        fwhm = self.interp_fwhm(wave)
        beta = self.interp_beta(wave)
        alpha = self.fwhm_to_alpha(fwhm, beta)
        
        f = lambda r, alpha, beta, x: self._moffat_integral(r,alpha,beta) - x
        
        r = newton(f, 2., args=(alpha, beta, fraction))
        return r


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    wave = np.array([4750., 7000., 9300.])
    fwhm = np.array([0.76, 0.66, 0.61])
    beta = np.array([2.6, 2.6, 2.6])
    moffat = MoffatSeeing(wave, fwhm, beta)

    r = np.arange(0, 10., 0.01)
    y_5000 = [moffat.flux_enclosed(i, 5000) for i in r]
    y_7000 = [moffat.flux_enclosed(i, 7000) for i in r]
    y_9000 = [moffat.flux_enclosed(i, 9000) for i in r]

    plt.plot(r, y_5000, 'k')
    plt.plot(r, y_7000, 'k') 
    plt.plot(r, y_9000, 'k')
    plt.axhline(0.995)
    plt.axvline(moffat.radius_enclosing(0.995, 5000))
    plt.axvline(moffat.radius_enclosing(0.995, 7000))
    plt.axvline(moffat.radius_enclosing(0.995, 9000))
    plt.show()
    
