import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import interp1d
import h5py

class BaseFluxGrid(object):

    @property
    def logZ_min(self):
        """Min logZ value spanned by grid [12+log10(O/H)]"""
        raise NotImplementedError("Subclasses should provide logZ_min attribute")

    @property
    def logZ_max(self):
        """Max logZ value spanned by grid [12+log10(O/H)]"""
        raise NotImplementedError("Subclasses should provide logZ_max attribute")

    @property
    def logU_min(self):
        """Min logU value spanned by grid"""
        raise NotImplementedError("Subclasses should provide logU_min attribute")

    @property
    def logU_min(self):
        """Max logU value spanned by grid"""
       
class CL01LineFlux(object):
    def __init__(self, filename, lines=[], wave=[]):
    
        fh = h5py.File(filename, 'r')
        
        index = []
        for line in lines:
            try:
                index.append(np.argwhere(fh['lines'][:] == line)[0,0])
            except IndexError:
                raise Exception('line %s unknown' % line)

        self.lines = lines
        self.wave = np.array(wave)
        self.logZ = fh['dim0'][:]
        self.logU = fh['dim1'][:]
        self.tauV = fh['dim2'][:]
        self.Z_solar = fh['dim0'].attrs['Z_solar']

        self.flux = fh['flux'][:][...,index]
        self.var = fh['var'][:][...,index]
    
        dims = [self.logZ, self.logU, self.tauV]
        self.intep_flux = RegularGridInterpolator(
                dims, self.flux, method='linear',
                bounds_error=True, fill_value=None)
        self.intep_var = RegularGridInterpolator(
                dims, self.var, method='linear',
                bounds_error=True, fill_value=None)


    def __call__(self, SFR, metal, logU_Zsolar, tauV):

        logZ = metal - self.Z_solar
        logZ = np.clip(logZ, self.logZ[0], self.logZ[-1])

        logU = -0.8 * logZ + logU_Zsolar
        logU = np.clip(logU, self.logU[0], self.logU[-1])
        tauV = np.clip(tauV, self.tauV[0], self.tauV[-1])

        x = np.column_stack([logZ, logU, tauV])

        flux = self.intep_flux(x)
        var = self.intep_var(x)

        mask = flux < 0
        flux[mask] = 0.
        var[mask] = 0.
        var[:] = 0.

        lum = SFR * 3.826e33 # L_sun per M_sun/yr

        flux = flux * lum[:,None]
        var = var * lum[:,None] ** 2.

        dt = np.dtype([('line', np.str_, max([len(i) for i in self.lines])),
                       ('wave', float),
                       ('flux', float, flux.shape[0]),
                       ('var', float, flux.shape[0])])
        out = np.zeros(flux.shape[1], dtype=dt)
        out['line'] = self.lines
        out['wave'] = self.wave
        out['flux'] = flux.T
        out['var'] = var.T

        return out

class D13LineFlux(object):
    def __init__(self, filename, lines=[], wave=[]):
    
        fh = h5py.File(filename, 'r')
        
        index = []
        for line in lines:
            try:
                index.append(np.argwhere(fh['lines'][:] == line)[0,0])
            except IndexError:
                raise Exception('line %s unknown' % line)

        self.lines = lines
        self.wave = wave
        self.logZ = fh['dim0'][:]
        self.logU = fh['dim1'][:]
        self.Z_solar = fh['dim0'].attrs['Z_solar']

        self.flux = fh['flux'][:][...,index]
    
        dims = [self.logZ, self.logU]
        self.intep_flux = RegularGridInterpolator(
                dims, self.flux, method='linear',
                bounds_error=True, fill_value=None)


    def __call__(self, SFR, metal, logU_Zsolar, tauV):

        logZ = metal - self.Z_solar
        logZ = np.clip(logZ, self.logZ[0], self.logZ[-1])

        logU = -0.8 * logZ + logU_Zsolar
        logU = np.clip(logU, self.logU[0], self.logU[-1])


        x = np.column_stack([logZ, logU])

        flux = self.intep_flux(x)
        var = 0.04 * flux

        ext = np.exp(-tauV[:,None] * (self.wave/5500.) ** -1.3)
        np.clip(ext, 0., 1., out=ext)
        flux *= ext
        var *= ext ** 2.

        lum = SFR / 7.9e-42 #1998ARA&A..36..189K

        flux = flux * lum[:,None]
        var = var * lum[:,None]**2.

        dt = np.dtype([('line', np.str_, max([len(i) for i in self.lines])),
                       ('wave', float),
                       ('flux', float, flux.shape[0]),
                       ('var', float, flux.shape[0])])
        out = np.zeros(flux.shape[1], dtype=dt)
        out['line'] = self.lines
        out['wave'] = self.wave
        out['flux'] = flux.T
        out['var'] = var.T

        return out


class EmpiricalLineFlux(object):
    def __init__(self, filename, lines=[], wave=[]):
    
        fh = h5py.File(filename, 'r')
        
        index = []
        for line in lines:
            try:
                index.append(np.argwhere(fh['lines'][:] == line)[0,0])
            except IndexError:
                raise Exception('line %s unknown' % line)

        self.lines = lines
        self.wave = wave
        self.OH = fh['OH'][:]
        self.flux = fh['flux'][:][:,index]
        self.var = fh['covar'][:][:,index,index]
    
        self.intep_flux = interp1d(self.OH, self.flux, copy=False, axis=0)
        self.intep_var = interp1d(self.OH, self.var, copy=False, axis=0)


    def __call__(self, SFR, metal):
        L_Ha = SFR / 7.9e-42 #1998ARA&A..36..189K

        flux = self.intep_flux(metal)

        flux = 10. ** flux
        flux = flux * L_Ha[:,None]

        var = self.intep_var(metal)
        var = (flux * np.log(10.)) ** 2. * var

        dt = np.dtype([('line', np.str_, max([len(i) for i in self.lines])),
                       ('wave', float),
                       ('flux', float, flux.shape[0]),
                       ('var', float, flux.shape[0])])
        out = np.zeros(flux.shape[1], dtype=dt)
        out['line'] = self.lines
        out['wave'] = self.wave
        out['flux'] = flux.T
        out['var'] = var.T

        return out



if __name__ == '__main__':
    filename = '/data2/MUSE/metallicity_calibration/flux_cal_singlevar.h5'
   
    lineFlux = EmpiricalLineFlux(filename, ['OIII_5007', 'OII'],
                                 ['5007.', 3727.])
    SFR = np.ones(11, dtype=float)
    metal = np.linspace(7.5, 9.5, 11)
    l = lineFlux(SFR, metal)
    print l['flux'][0] / l['flux'][1]


