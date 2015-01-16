import numpy as np
import pyfits
import pywcs
from scipy.integrate import dblquad

class ObsSim(object):
        
    def __init__(self, ra, dec, segmap):
        self.segmap, hdr = pyfits.getdata(segmap, header=True)

        shape = self.segmap.shape + (2,)
        self.grid_centre = self.grid_centres(hdr, ra, dec).reshape(shape)
        shape = self.segmap.shape + (4,2)
        corners = self.grid_corners(hdr, ra, dec).reshape(shape)
        self.grid_area = self.calc_area(corners)

    @staticmethod
    def _dist_from(wcs, pix_x, pix_y, ra, dec):
        coords = np.column_stack([pix_x, pix_y])

        new_coords = wcs.all_pix2sky(coords, 0)
        sky_x = new_coords[:,0]
        sky_x -= np.round((sky_x - ra) / 360.) * 360.
        print sky_x
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
   
    @staticmethod
    def moffat(r, a, b):
        return (b-1.) / (np.pi*a**2.) / (1. + (r/a)**2.)**b

    def model(self, data, lines):
        a = 0.6
        b = 2.6

        bins = np.zeros_like(self.segmap, dtype=float)
        seg_ids = np.unique(self.segmap)
        for seg_id in seg_ids:
            print seg_id
            if seg_id <= 0:
                continue
            mask = (self.segmap == seg_id)
            dx = data['x'] - self.grid_centre[mask,0][:,None]
            dy = data['y'] - self.grid_centre[mask,1][:,None]
            r = np.sqrt(dx**2. + dy**2.)

            for line in lines:
                frac = self.moffat(r, a, b)
                bins[mask] = np.sum(frac * data[line['name']+'_flux'])
        return bins
        

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import galaxy
    from sfr_profile import ExponentialSFRProfile
    from metallicity_profile import LinearMetallicityProfile
    from line_flux import PlaceholderLineFlux

    lines = ['OII_3726', 'OII_3729', 'H_delta', 'H_gamma', 'H_beta',
             'OIII_4959', 'OIII_5007']

    gal = galaxy.Galaxy(ExponentialSFRProfile, LinearMetallicityProfile,
                        PlaceholderLineFlux, lines=lines)
    
    params = {'z': 0.5,
              'SFRDensity_centre': 0.1,
              'SFRDensity_thres': 1e-4,
              'r0': 1.0,
              'inc': 40.,
              'PA': 30.0,
              'Z_in': 9.0,
              'Z_out': 8.0,}
    data, lines = gal.model(params)

    obsSim = ObsSim(180., 0., 'test_binmap3.fits')
    im = obsSim.model(data, lines)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    im = ax.imshow(im, interpolation='nearest', origin='lower', cmap='RdYlBu_r')

    plt.colorbar(im)
    plt.show()
