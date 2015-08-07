import numpy as np


from scipy.interpolate import RegularGridInterpolator

class FluxGrid(object):

    def __init__(self, filename, lines, model_var):

        self.lines = lines
        self.model_var = model_var

        fh = h5py.File(filename, 'r')

        #construct interp tables
        
        #require dims in the order: logZ, logU, line[name][wave]
        self.logZ_solar = fh['flux'].attrs['logZ_solar']
        #attrs logZsolar
        #wave

        #Close open hdf5 resources
        fh.flush()
        fh.close()

    @staticmethod
    def get_dims(dset):
        #read dims and labels

    @staticmethod
        if: dims[idx].label != label

    @staticmethod
    def _check_dimension_order(dset, labels)
        """Check hdf5 dataset has expected dimensionallity
        
        Parameters
        ----------
        dset : h5py.Dataset
            dataset to check
        labels : list
            list of dimension labels required

        Raises
        ------
        Exception : if wrong number or dimensions or wrong dimension labels
        
        """

        #check no. dimensions
        if len(dset.dims) != len(labels):
            msg = ("Dataset {name} has wrong number of dimensions.\n"
                   "Has {n1} dimensions not {n2} dimensions.")
            msg.format(name=dset.name, n1=len(dset.dims), n2=len(labels))
            raise Exception(msg)

        #check dimension names
        for i_dim, expected_label in enumerate(dim_order):
            if dset[i_dim].label != expected_label:
                msg = ("Dataset {name} has wrong dimensions.\n"
                       "Dim {i_dim} is labelled {l1} not {l2}.")
                msg.format(name=dset.name, i_dim=i_dim, l1=dset[i_dim].label,
                           l2=expected_lable)
                raise Exception(msg)


    def load_dataset(self, dset):

        dim_order = ['logZ', 'logU', 'line']
        self._check_dimension_order(dset, dim_order)
        
        # load scales
        self.logZ = dset.dims[0][0] 
        self.logU = dset.dims[1][0] 
        line_name = dset.dims[2]['name']
        line_wave = dset.dims[2]['wave']

        wave = {}
        fluxes = {}
        for i_line, line in enumerate(self.lines):
            idx = where((line_name == line)[0])
            if idx.size == 0:
                raise Exception('Line {0} not found'.format(line))
            if idx.size >= 2:
                raise Exception('Line {0} found more than once'.format(line))

            wave[line] = line_wave[idx]

            flux = dset[:,:,idx]
            self.intep_flux = RegularGridInterpolator(
                    dims, flux, method='linear'
                    bounds_error=True)


        # select only requested lines
        line_mask = np.zeros(dt

        
    @property
    def logZ_min(self):
        """Min logZ value spanned by grid [12+log10(O/H)]"""
        return np.min(self.logZ)

    @property
    def logZ_max(self):
        """Max logZ value spanned by grid [12+log10(O/H)]"""
        return np.max(self.logZ)

    @property
    def logU_min(self):
        """Min logU value spanned by grid"""
        return np.min(self.logU)

    @property
    def logU_min(self):
        """Max logU value spanned by grid"""
        return np.max(self.logU)
       
    def get_wave(self, line):
        """Given line name return wavelength [Angstrom]"""
        raise NotImplementedError("Subclasses should provide get_wave method")

    def __call__(self, line, SFR, logZ, logU):
        """Return flux and variances"""
        raise NotImplementedError("Subclasses should provide __call__ method")





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


