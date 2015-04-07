import numpy as np
from scipy.interpolate import interp1d
import h5py

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


