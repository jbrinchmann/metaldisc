import numpy as np
import pyfits
import pywcs
from scipy.integrate import dblquad
from scipy import sparse
from scipy import spatial


class ObsSim(object):
        
    def __init__(self, ra, dec, segmap, seeing):
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
   

    def model(self, galaxy):
        data = galaxy.model()


        uniq_bin_id = np.unique(self.bin_id)
        n_bins = len(uniq_bin_id)
        n_pixels = len(self.bin_id)
        area_mapper = np.zeros((n_bins, n_pixels), dtype=float)
        for i, id_ in enumerate(uniq_bin_id):
            mask = (self.bin_id == id_)
            area_mapper[i][mask] = self.bin_area[mask]

        area_mapper = sparse.csr_matrix(area_mapper)

        dist = self.coords.sparse_distance_matrix(galaxy.coordKDTree, 3.)
        seeing_blur = dist.tocsr()
        seeing_blur.data[:] = self.seeing(seeing_blur.data, 7000.)

        redist = area_mapper * seeing_blur

        bin_flux = redist.dot(data['flux'][1])
        
        bins = np.zeros_like(self.segmap, dtype=float)
        for i, id_ in enumerate(uniq_bin_id):
            mask = (self.segmap == id_)
            bins[mask] = bin_flux[i]
#        seg_ids = np.unique(self.segmap)
#        for seg_id in seg_ids:
#            print seg_id
#            if seg_id <= 0:
#                continue
#            mask = (self.segmap == seg_id)
#            dx = data['x'] - self.grid_centre[mask,0][:,None]
#            dy = data['y'] - self.grid_centre[mask,1][:,None]
#            r = np.sqrt(dx**2. + dy**2.)
#
#            for line in lines:
#                frac = self.moffat(r, a, b)
#                bins[mask] = np.sum(frac * data[line['name']+'_flux'])
        return bins, bin_flux
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from galaxy import Galaxy
    from SFdensity import ExpDisc
    from metallicity import LinearMetallicity
    from lineflux import EmpiricalLineFlux
    from seeing import MoffatSeeing

    sf_density = ExpDisc(.1, 0.1)
    metallicity = LinearMetallicity(1, 8.5, 8.5)
    filename = '/data2/MUSE/metallicity_calibration/flux_cal_singlevar.h5'
    lineflux = EmpiricalLineFlux(filename, ['OIII_5007', 'H_BETA'],
                                 ['5007.', 4861.])

    cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'omega_k_0': 0.0, 'h':0.7}

    galaxy = Galaxy(0.564497, cosmo, 0.78042, 76.6-90., 18.94, 45,
                    sf_density, metallicity, lineflux)

    wave = np.array([4750., 7000., 9300.])
    fwhm = np.array([0.76, 0.66, 0.61])
    beta = np.array([2.6, 2.6, 2.6])
    moffat = MoffatSeeing(wave, fwhm, beta)

    obsSim = ObsSim(338.2239, -60.560436, '/data2/MUSE/metallicity_analysis/flux_extraction/spectra_extraction/binmap5.fits', moffat)
    im, bin_flux = obsSim.model(galaxy)
    print bin_flux.sum()
    print im.sum()

    plt.imshow(im, interpolation='nearest', origin='lower')
    plt.colorbar()
    plt.show()
    
