import numpy as np
import pyfits
import pywcs
from scipy.integrate import dblquad
from scipy import sparse
from scipy import spatial


class ObsSim(object):
        
    def __init__(self, galaxy, ra, dec, segmap, seeing):
        self.galaxy = galaxy
        galaxy.register_observer(self)

        self.seeing = seeing
        
        self.segmap, hdr = pyfits.getdata(segmap, header=True)
#        self.segmap.flat = np.arange(self.segmap.size)+1
    
        shape = self.segmap.shape + (2,)
        self.grid_centre = self.grid_centres(hdr, ra, dec).reshape(shape)

        shape = self.segmap.shape + (4,2)
        corners = self.grid_corners(hdr, ra, dec).reshape(shape)
        self.grid_area = self.calc_area(corners)

        mask = (self.segmap >= 1)
        self.bin_id = self.segmap[mask]
        self.bin_centre = self.grid_centre[mask]
        self.bin_area = self.grid_area[mask]

        self.coords = spatial.cKDTree(self.bin_centre)
        self.create_bins_to_samples_area_map()
        self.calc_samples_to_points_dist()

        self.redist = {}
 
    def notify(self, observable, *args, **kwargs):
        if bins_changed:
            self.calc_samples_to_points_dist()


    @staticmethod
    def _dist_from(wcs, pix_x, pix_y, ra, dec):
        coords = np.column_stack([pix_x, pix_y])

        new_coords = wcs.all_pix2sky(coords, 0)
        sky_x = new_coords[:,0]
        sky_x -= np.round((sky_x - ra) / 360.) * 360.
        sky_y = new_coords[:,1]

        dist_x = (sky_x - ra) * np.cos(np.radians(dec)) * 3600.
        dist_y = (sky_y - dec) * 3600.

        return dist_x, dist_y 

    def grid_centres(self, hdr, ra, dec):
        n_x = hdr['NAXIS1']
        n_y = hdr['NAXIS2']
        wcs = pywcs.WCS(hdr)

        x = np.tile(np.arange(n_x, dtype=float), n_y).ravel()
        y = np.repeat(np.arange(n_y, dtype=float), n_x).ravel()

        x, y = self._dist_from(wcs, x, y, ra, dec)
        centres = np.column_stack([x, y])
        return centres

    def grid_corners(self, hdr, ra, dec):
        n_x = hdr['NAXIS1']
        n_y = hdr['NAXIS2']
        wcs = pywcs.WCS(hdr)

        x = np.tile(np.arange(n_x, dtype=float), n_y).ravel()
        y = np.repeat(np.arange(n_y, dtype=float), n_x).ravel()
        
        corners = np.zeros((n_x*n_y,4,2), dtype=float)
        corners[:,0,0], corners[:,0,1] = self._dist_from(wcs, x-0.5, y-0.5,
                                                         ra, dec)
        corners[:,1,0], corners[:,1,1] = self._dist_from(wcs, x+0.5, y-0.5,
                                                         ra, dec)
        corners[:,2,0], corners[:,2,1] = self._dist_from(wcs, x-0.5, y+0.5,
                                                         ra, dec)
        corners[:,3,0], corners[:,3,1] = self._dist_from(wcs, x+0.5, y+0.5,
                                                         ra, dec)
        return corners

    @staticmethod
    def calc_area(corners):
        AC = corners[:,:,3,:] - corners[:,:,0,:]
        BD = corners[:,:,2,:] - corners[:,:,1,:]
        area = 0.5 * np.abs(AC[:,:,0] * BD[:,:,1] - BD[:,:,0] * AC[:,:,1])
        return area

    def create_bins_to_samples_area_map(self):
        #construct map from bins to sample points - do this once
        uniq_bin_id = np.unique(self.bin_id)
        n_bins = len(uniq_bin_id)
        n_pixels = len(self.bin_id)
        area_mapper = np.zeros((n_bins, n_pixels), dtype=float)
        for i, id_ in enumerate(uniq_bin_id):
            mask = (self.bin_id == id_)
            area_mapper[i][mask] = self.bin_area[mask] / np.sum(mask)

        self.area_mapper = sparse.csr_matrix(area_mapper)

    def calc_samples_to_points_dist(self):
        #calc dist from sample cooords to galaxy coords (less that 3")
        dist = self.coords.sparse_distance_matrix(galaxy.coordKDTree, 3.)
        self.dist_matrix = dist.tocsr()
   

    def model(self):
        data = self.galaxy.model()

        uniq_bin_id = np.unique(self.bin_id)
        out_dt = np.dtype(data.dtype.descr[0:2] + 
                          [('flux', float, uniq_bin_id.size),
                           ('var', float, uniq_bin_id.size)])
        out = np.zeros(len(data), dtype=out_dt)

        for i in xrange(len(data)):
            line = data[i]['line']
            wave = data[i]['wave']

            out['line'][i] = line
            out['wave'][i] = wave

            try:
                redist_flux, redist_var = self.redist[line]

            except KeyError:
                seeing_redist = self.dist_matrix.copy()

                seeing_redist.data[:] = self.seeing(seeing_redist.data, wave)
                redist_flux = self.area_mapper * seeing_redist

                redist_var = redist_flux.copy()
                redist_var.data[:] = redist_var.data[:] ** 2.

                self.redist[line] = (redist_flux, redist_var)

            out['flux'][i] = redist_flux.dot(data[i]['flux'])
            out['var'][i] = redist_var.dot(data[i]['var'])

        return out
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from galaxy import Galaxy
    from SFdensity import ExpDisc
    from metallicity import LinearMetallicity
    from lineflux import EmpiricalLineFlux
    from seeing import MoffatSeeing

    from astropy.cosmology import FlatLambdaCDM

    sf_density = ExpDisc(.1, 0.1)
    metallicity = LinearMetallicity(1, 8.5, 8.5)
    filename = '/data2/MUSE/metallicity_calibration/flux_cal_singlevar.h5'
    lineflux = EmpiricalLineFlux(filename, ['OIII_5007', 'H_BETA', 'OII'],
                                 ['5007.', 4861., '3727.'])

    cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

    galaxy = Galaxy(0.564497, cosmo, 0.78042, 76.6-90., 18.94, 45,
                    sf_density, metallicity, lineflux)

    wave = np.array([4750., 7000., 9300.])
    fwhm = np.array([0.76, 0.66, 0.61])
    beta = np.array([2.6, 2.6, 2.6])
    moffat = MoffatSeeing(wave, fwhm, beta)

    obsSim = ObsSim(galaxy, 338.2239, -60.560436, '/data2/MUSE/metallicity_analysis/flux_extraction/spectra_extraction/binmap1.fits', moffat)
    print obsSim.model()


    plt.imshow(im, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.show()
    
