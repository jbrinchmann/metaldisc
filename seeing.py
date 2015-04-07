import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad

class MoffatSeeing(object):

    def __init__(self, wave, fwhm, beta):
        self.wave = wave
        self.fwhm = fwhm
        self.beta = beta
    
        self.interp_fwhm = interp1d(self.wave, self.fwhm, copy=False)
        self.interp_beta = interp1d(self.wave, self.beta, copy=False)

    def __call__(self, r, wave):
        fwhm = self.interp_fwhm(wave)
        beta = self.interp_beta(wave)
        alpha = self.fwhm_to_alpha(fwhm, beta)

        profile = self._func(r, alpha, beta)
        return profile

    @staticmethod
    def fwhm_to_alpha(fwhm, beta):
        alpha = fwhm / (2. * np.sqrt(2.**(1./beta) - 1.))
        return alpha

    @staticmethod
    def _func(r, alpha, beta):
        norm = (beta-1.) / (np.pi*alpha**2.)
        profile = (1. + (r/alpha)**2.)**-beta
        return norm * profile

    def __int_func(self, r, alpha, beta):
        return 2. * np.pi * r * self._func(r, alpha, beta)

    def flux_enclosed(self, r, wave):
        fwhm = self.interp_fwhm(wave)
        beta = self.interp_beta(wave)
        alpha = self.fwhm_to_alpha(fwhm, beta)

        res = quad(self.__int_func, 0, r, args=(alpha, beta))
        return res[0]

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    wave = np.array([4750., 7000., 9300.])
    fwhm = np.array([0.76, 0.66, 0.61])
    beta = np.array([2.6, 2.6, 2.6])
    moffat = MoffatSeeing(wave, fwhm, beta)

    r = np.arange(0, 3., 0.05)
    y_5000 = [moffat.flux_enclosed(i, 5000) for i in r]
    y_7000 = [moffat.flux_enclosed(i, 7000) for i in r]
    y_9000 = [moffat.flux_enclosed(i, 9000) for i in r]

    plt.plot(r, y_5000) 
    plt.plot(r, y_7000) 
    plt.plot(r, y_9000) 
    plt.show()
    
