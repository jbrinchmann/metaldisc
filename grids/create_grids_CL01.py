from collections import OrderedDict
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import h5py
from pyhdf.SD import SD, SDC

def get_dims(file_):
    
    fh = SD(file_, SDC.READ)

    dims = OrderedDict()
    dim_names = ['log Z', 'Log Mu', 'Tau_V', 'Xsi', 'Log U']
    for dim_name in dim_names:
        dims[dim_name] = fh.select(dim_name)[:]

    fh.end()
    return dims

class Interpolator(object):
    def __init__(self, file_):
        fh = SD(file_, SDC.READ)

        dims = get_dims(file_)
        self.dim_min = np.array([np.min(dim) for dim in dims.itervalues()])
        self.dim_max = np.array([np.max(dim) for dim in dims.itervalues()])

        values = fh.select('Time   1')[:].astype(float)

        self.interp = RegularGridInterpolator(
                dims.values(), values, method='linear',
                bounds_error=True, fill_value=None)

        fh.end()

    def __call__(self, xi, method=None):

        xi_clipped = xi.copy()
        for i in xrange(5):
            np.clip(xi_clipped[...,i], self.dim_min[i], self.dim_max[i],
                    out=xi_clipped[...,i])

        y = self.interp(xi_clipped, method) #linear interpolation
        
        return y

if __name__ == '__main__':

    hdffile_dir = '/disks/galpop2/jarle/HDFFiles/'

    fh_out = h5py.File('model_CL01_fluxes.h5', 'w')


    dims = get_dims(hdffile_dir + 'n1p3_LHa_6563_20p0.hdf')
    new_dims = [dims['log Z'], # logZ
                np.array([-0.50267535]), # Log Mu
                dims['Tau_V'], # tauV
                np.linspace(0.1, 0.5, 100), # xi
                dims['Log U']] # logU
    len_new_dims = [i.size for i  in new_dims]
    new_dims = np.meshgrid(*new_dims, indexing='ij')
    new_dims = np.column_stack([i.flat for i in new_dims])
    new_dims = new_dims.reshape(len_new_dims + [5])

    fh_out['dim0'] = dims['log Z']
    fh_out['dim1'] = dims['Log U']
    fh_out['dim2'] = dims['Tau_V']

    fh_out['dim0'].attrs['name'] = 'logZ'
    fh_out['dim1'].attrs['name'] = 'logU'
    fh_out['dim2'].attrs['name'] = 'tauV'
    fh_out['dim0'].attrs['Z_solar'] = 8.82

    lines = ['OII_3727', 'NeIII_3869', 'Hd_4101', 'Hg_4340', 'OIII_4363',
             'Hb_4861', 'OIII_4959', 'OIII_5007', 'OI_6300', 'Ha_6563',
             'NII_6584', 'SII_6716', 'SII_6724', 'SII_6731']

    fh_out['lines'] = lines

    out_shape = [len_new_dims[0], len_new_dims[4], len_new_dims[2], len(lines)]
    fh_out.create_dataset('flux', shape=out_shape, dtype=float,
                          fillvalue=np.nan)
    fh_out.create_dataset('var', shape=out_shape, dtype=float,
                          fillvalue=np.nan)

    for i, line in enumerate(lines):
        print line

        interp = Interpolator(hdffile_dir + 'n1p3_L' + line + '_20p0.hdf')
        flux = interp(new_dims)
        flux_mean = np.mean(flux, 3)
        flux_var = np.std(flux, 3, ddof=1) ** 2.

        flux_mean = np.squeeze(flux_mean)
        flux_var = np.squeeze(flux_var)
        flux_mean = np.swapaxes(flux_mean, 1, 2)
        flux_var = np.swapaxes(flux_var, 1, 2)

        fh_out['flux'][:,:,:,i] = flux_mean
        fh_out['var'][:,:,:,i] = flux_var

    fh_out.flush()
    fh_out.close()
