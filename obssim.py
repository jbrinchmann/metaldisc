import numpy as np
from astropy import wcs
from scipy import sparse
from scipy.spatial import cKDTree

#base obs sim with mapping code
#make image from binmap
#make image from spec ra dec, pixsize, etc.
    
#restore notify and register observers

class BaseObsSim(object):

    def __init__(self, galaxy, seeing, conserve_flux=False):
        """Base class of observation simulations

        Parameters
        ----------
        galaxy : instance of BaseGalaxy or subclass
            galaxy to simulate observations of
        seeing : instance of BaseSeeing or subclass
            seeing model to be used in simulation
        conserve_flux : bool [default=False]
            if flag set, bins conserve flux, rather than intensity
            i.e. if False values normalized to number of pixels per bin

        """
        #Read only params
        self._galaxy = galaxy
        self._seeing = seeing

        # Required attributes
        self.__pixel_id = None
        self.__pixel_area = None

        #special
        self.__pixel_coord = None
        self.__pixel_coord_tree = None

        self.conserve_flux = conserve_flux
        self.mapping_matrix = {} #store for flux mappings

    # Make sure setup is read only as flux redistribution depends on them being fixed
    @property
    def galaxy(self):
        """Get galaxy being simulated"""
        return self._galaxy

    @property
    def seeing(self):
        """Get seeing used in simulation"""
        return self._seeing
    #END readonly

    @property
    def pixel_id(self):
        """Array of image pixel id numbers"""
        if self.__pixel_id is None:
            raise NotImplementedError("Subclasses should provide pixel_id attribute")
        else:
            return self.__pixel_id

    @pixel_id.setter
    def pixel_id(self, value):
        self.__pixel_id = value

    @property
    def pixel_area(self):
        """Array of image pixel areas [arcsec^2]"""
        if self.__pixel_area is None:
            raise NotImplementedError("Subclasses should provide pixel_area attribute")
        else:
            return self.__pixel_area

    @pixel_area.setter
    def pixel_area(self, value):
        self.__pixel_area = value


    #require subclasses to supply pixel_coords
    @property
    def pixel_coord(self):
        """Get Nx2 array relative image pixel coords (for N pixels) [arcsec]"""
        if self.__pixel_coord is None:
            raise NotImplementedError("Subclasses should provide bin_coord attribute")
        else:
            return self.__pixel_coord

    #automatically compute bin_coordTree
    @pixel_coord.setter
    def pixel_coord(self, value):
        self.__pixel_coord = value
        self.__pixel_coord_tree = cKDTree(value) #construct cKDTree


    @property
    def pixel_coord_tree(self):
        """Get scipy.cKDTree representation of pixel_coord"""
        if self.__pixel_coord_tree is None:
            raise NotImplementedError("Subclasses should provide pixel_coord attribute")
        else:
            return self.__pixel_coord_tree


    def _calc_pixel2bin_matrix(self):
        """Constructs matrix mapping pixel area to bins

        Returns
        -------
        m : CSR sparse matrix

        """

        uniq_pixel_id = np.unique(self.pixel_id)

        n_bins = len(uniq_pixel_id)
        n_pixels = len(self.pixel_id)
        area_mapper = np.zeros((n_bins, n_pixels), dtype=float)

        for i, id_ in enumerate(uniq_pixel_id):
            mask = (self.pixel_id == id_)
            area_mapper[i][mask] = self.pixel_area[mask]
            if not self.conserve_flux: 
               area_mapper[i][mask] /= np.sum(mask) #conserve intensity

        m = sparse.csr_matrix(area_mapper)
        return m

    def _calc_psf2pixel_matrix(self, wave):
        maxdist = self.seeing.radius_enclosing(0.995, wave)

        #get distance between image pixels and galaxy bins
        m = self.pixel_coord_tree.sparse_distance_matrix(
                self.galaxy.bin_coord_tree, maxdist)
        m = m.tocsr()

        #calc PSF at distances
        m.data[:] = self.seeing(m.data, wave)

        return m
    

    def calc_mapping_matrix(self, line):

        wave = self.galaxy.get_obs_wave(line) #observed wavelength

        #matrix mapping image pixels to image bins multiplied by pixel area
        pixel2bin = self._calc_pixel2bin_matrix()

        #matrix mapping galaxy bins to image pixels redistributed according to
        #seeing
        psf2pixel = self._calc_psf2pixel_matrix(wave)

        mapping_matrix = pixel2bin * psf2pixel
        return mapping_matrix


    def get_mapping_matrix(self, line):

        try:
            #check if already computed
            m = self.mapping_matrix[line]
        except KeyError:
            #if not, compute now
            m = self.calc_mapping_matrix(line)
            self.mapping_matrix[line] = m

        return m

    def __call__(self, lines, params):

        uniq_pixel_id = np.unique(self.pixel_id)
        n_bins = len(uniq_pixel_id)
        n_lines = len(lines)

        flux = np.full((n_bins, n_lines), np.nan, dtype=float)
        var = np.full((n_bins, n_lines), np.nan, dtype=float)

        gal_flux, gal_var = self.galaxy(lines, params)

        for i_line, line in enumerate(lines):
            mapping = self.get_mapping_matrix(line)
            flux[:,i_line] = mapping.dot(gal_flux[:,i_line])
            var[:,i_line] = mapping.dot(gal_var[:,i_line])

        return flux, var
            



class BinmapObsSim(BaseObsSim):

    def __init__(self, binmap, galaxy, seeing, conserve_flux=False):
        """Simulates observations given a segmentation map image

        Notes
        -----
        The segmentation map is a integer array where values represent:
            >= 1: represents bin numbers
            <= 0: pixels are ignored

        Parameters
        ----------
        binmap : astropy.io.fits.ImageHDU instance
            segmentation map image with WCS info in header 
        galaxy : instance of BaseGalaxy or subclass
            galaxy to simulate observations of
        seeing : instance of BaseSeeing or subclass
            seeing model to be used in simulation
        conserve_flux : bool [default=False]
            if flag set, bins conserve flux, rather than intensity
            i.e. if False values normalized to number of pixels per bin
            
        """

        super(BinmapObsSim, self).__init__(galaxy, seeing, conserve_flux)

        mask = (binmap.data.flatten() >= 1)

        self.pixel_id = binmap.data.flatten()[mask]

        x, y = self.get_pixel_coord(binmap.header)
        self.pixel_coord = np.column_stack([x[mask], y[mask]])
        
        self.pixel_area = self.get_pixel_area(binmap.header)[mask]


    def calc_rel_coords(self, ra, dec):
        """Calculates the relative sky coordinates given sky coordinates

        Parameters
        ----------
        ra : array of floats
            right ascention [deg]
        dec : array of floats
            declination [deg]

        Returns
        -------
        x : array of floats
            x-axis distances from galaxy centre [arcsec]
        y : array of floats
            y-axis distances from galaxy centre [arcsec]

        """

        x = ra - self.galaxy.ra
        y = dec - self.galaxy.dec

        x -= np.round(x/360.) * 360. # wrap to interval (-180,180)

        x *= np.cos(np.radians(self.galaxy.dec)) * 3600. #deg to arcsec
        y *= 3600. #deg to arcsec

        return x, y 


    def get_pixel_coord(self, hdr, x_offset=0., y_offset=0.):
        """Get relative pixel coordinates

        Notes
        -----
        x_offset=-0.5, y_offset=-0.5 yields coordinates of pixels' lower-left corners

        Parameters
        ----------
        hdr : astropy.io.fits.Header instance
            FITS header including WCS info
        x_offset : [optional]float
            x-axis pixel coord offsets
        y_offset : [optional]float
            y-axis pixel coord offsets

        Returns
        -------
        x : array of floats
            x coords of relative pixel centres [arcsec]
        y : array of floats
            y coords of relative pixel centres [arcsec]

        """
        #get pixel centres [image coords]
        n_x = hdr['NAXIS1']
        n_y = hdr['NAXIS2']
        pix_x = np.tile(np.arange(n_x, dtype=float), n_y) + x_offset
        pix_y = np.repeat(np.arange(n_y, dtype=float), n_x) + y_offset

        #convert pixel centres to sky coords
        wcsobj = wcs.WCS(hdr)
        coords = np.column_stack([pix_x, pix_y])
        new_coords = wcsobj.all_pix2world(coords, 0) #[[ra], [dec]]

        #get pixel distances from galaxy [arcsec]
        x, y = self.calc_rel_coords(new_coords[:,0], new_coords[:,1])

        return x, y


    def get_pixel_area(self, hdr):
        """Set pixel areas

        Parameters
        ----------
        hdr : astropy.io.fits.Header instance
            FITS header including WCS info

        Returns
        -------
        area : array of floats
            pixel areas [arcsec^2]

        """

        #get pixel corners [arcsec]  (clockwise)
        A_x, A_y = self.get_pixel_coord(hdr, x_offset=-0.5, y_offset=-0.5)
        B_x, B_y = self.get_pixel_coord(hdr, x_offset=-0.5, y_offset=+0.5)
        C_x, C_y = self.get_pixel_coord(hdr, x_offset=+0.5, y_offset=+0.5)
        D_x, D_y = self.get_pixel_coord(hdr, x_offset=+0.5, y_offset=-0.5)

        #get vectors between opposite corners
        AC_x = C_x - A_x
        AC_y = C_y - A_y

        BD_x = D_x - B_x
        BD_y = D_y - B_y

        area = 0.5 * np.abs((AC_x * BD_y) - (BD_x * AC_y))

        return area


#class ObsSim(object):
#        
#    def __init__(self, galaxy, ra, dec, segmap, seeing):
#        self.galaxy = galaxy
#
#        self.seeing = seeing
#        
#        self.segmap, hdr = pyfits.getdata(segmap, header=True)
##        self.segmap.flat = np.arange(self.segmap.size)+1
#    
#        shape = self.segmap.shape + (2,)
#        self.grid_centre = self.grid_centres(hdr, ra, dec).reshape(shape)
#
#        shape = self.segmap.shape + (4,2)
#        corners = self.grid_corners(hdr, ra, dec).reshape(shape)
#        self.grid_area = self.calc_area(corners)
#
#        mask = (self.segmap >= 1)
#        self.bin_id = self.segmap[mask]
#        self.bin_centre = self.grid_centre[mask]
#        self.bin_area = self.grid_area[mask]
#
#        self.coords = spatial.cKDTree(self.bin_centre)
#        self.create_bins_to_samples_area_map()
#        self.calc_samples_to_points_dist()
#
#        self.redist = {}
# 
#
#    @staticmethod
#    def _dist_from(wcs, pix_x, pix_y, ra, dec):
#        coords = np.column_stack([pix_x, pix_y])
#
#        new_coords = wcs.all_pix2sky(coords, 0)
#        sky_x = new_coords[:,0]
#        sky_x -= np.round((sky_x - ra) / 360.) * 360.
#        sky_y = new_coords[:,1]
#
#        dist_x = (sky_x - ra) * np.cos(np.radians(dec)) * 3600.
#        dist_y = (sky_y - dec) * 3600.
#
#        return dist_x, dist_y 
#
#    def grid_centres(self, hdr, ra, dec):
#        n_x = hdr['NAXIS1']
#        n_y = hdr['NAXIS2']
#        wcs = pywcs.WCS(hdr)
#
#        x = np.tile(np.arange(n_x, dtype=float), n_y).ravel()
#        y = np.repeat(np.arange(n_y, dtype=float), n_x).ravel()
#
#        x, y = self._dist_from(wcs, x, y, ra, dec)
#        centres = np.column_stack([x, y])
#        return centres
#
#    def grid_corners(self, hdr, ra, dec):
#        n_x = hdr['NAXIS1']
#        n_y = hdr['NAXIS2']
#        wcs = pywcs.WCS(hdr)
#
#        x = np.tile(np.arange(n_x, dtype=float), n_y).ravel()
#        y = np.repeat(np.arange(n_y, dtype=float), n_x).ravel()
#        
#        corners = np.zeros((n_x*n_y,4,2), dtype=float)
#        corners[:,0,0], corners[:,0,1] = self._dist_from(wcs, x-0.5, y-0.5,
#                                                         ra, dec)
#        corners[:,1,0], corners[:,1,1] = self._dist_from(wcs, x+0.5, y-0.5,
#                                                         ra, dec)
#        corners[:,2,0], corners[:,2,1] = self._dist_from(wcs, x-0.5, y+0.5,
#                                                         ra, dec)
#        corners[:,3,0], corners[:,3,1] = self._dist_from(wcs, x+0.5, y+0.5,
#                                                         ra, dec)
#        return corners
#
#    @staticmethod
#    def calc_area(corners):
#        AC = corners[:,:,3,:] - corners[:,:,0,:]
#        BD = corners[:,:,2,:] - corners[:,:,1,:]
#        area = 0.5 * np.abs(AC[:,:,0] * BD[:,:,1] - BD[:,:,0] * AC[:,:,1])
#        return area
#
#    def create_bins_to_samples_area_map(self):
#        #construct map from bins to sample points - do this once
#        uniq_bin_id = np.unique(self.bin_id)
#        n_bins = len(uniq_bin_id)
#        n_pixels = len(self.bin_id)
#        area_mapper = np.zeros((n_bins, n_pixels), dtype=float)
#        for i, id_ in enumerate(uniq_bin_id):
#            mask = (self.bin_id == id_)
#            area_mapper[i][mask] = self.bin_area[mask] / np.sum(mask)
#
#        self.area_mapper = sparse.csr_matrix(area_mapper)
#
#    def calc_samples_to_points_dist(self):
#        #calc dist from sample cooords to galaxy coords (less that 3")
#        dist = self.coords.sparse_distance_matrix(self.galaxy.coordKDTree, 3.)
#        self.dist_matrix = dist.tocsr()
#   
#
#    def model(self):
#        data = self.galaxy.model()
#
#        uniq_bin_id = np.unique(self.bin_id)
#        out_dt = np.dtype(data.dtype.descr[0:2] + 
#                          [('flux', float, uniq_bin_id.size),
#                           ('var', float, uniq_bin_id.size)])
#        out = np.zeros(len(data), dtype=out_dt)
#
#        for i in xrange(len(data)):
#            line = data[i]['line']
#            wave = data[i]['wave']
#
#            out['line'][i] = line
#            out['wave'][i] = wave
#
#            try:
#                redist_flux, redist_var = self.redist[line]
#
#            except KeyError:
#                seeing_redist = self.dist_matrix.copy()
#
#                seeing_redist.data[:] = self.seeing(seeing_redist.data, wave)
#                redist_flux = self.area_mapper * seeing_redist
#
#                redist_var = redist_flux.copy()
#                redist_var.data[:] = redist_var.data[:] ** 2.
#
#                self.redist[line] = (redist_flux, redist_var)
#
#            out['flux'][i] = redist_flux.dot(data[i]['flux'])
#            out['var'][i] = redist_var.dot(data[i]['var'])
#
#        return out
        

if __name__ == '__main__':
    from astropy.io import fits
    from astropy.cosmology import FlatLambdaCDM

    from galaxy import GalaxyDisc
    from fluxgrid import FluxGrid
    from seeing import MoffatSeeing


    lines = ['O2-3727', 'O2-3729', 'Hg', 'Hb', 'O3-5007']
    fluxgrid = FluxGrid('grids/grid_Dopita13_kappa=inf.h5', lines, 0.04)

    cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
    gal = GalaxyDisc(338.24124, -60.563644, 0.427548, 0., 0., 3., 45, cosmo, fluxgrid)

    wave = np.array([4750., 7000., 9300.])
    fwhm = np.array([0.76, 0.66, 0.61])
    beta = np.array([2.6, 2.6, 2.6])
    moffat = MoffatSeeing(wave, fwhm, beta)

    binmap = fits.open('/data2/MUSE/metallicity_analysis/spectra_extraction/v03p1/0020/binmap-image.fits')[1]
    obssim = BinmapObsSim(binmap, gal, moffat, conserve_flux=False)

    params = {
        'SFdensity_0': 0.647860980069 / 31.2604018452, #total SFR = 1
        'r_d': 0.5,
        'Z_in': 9.0,
        'Z_out': 8.9,
        'logU_0': -3.,
        'tauV_in': 0.2,
        'tauV_out': 0.1,
        }

    flux, var = obssim(lines, params)
    
    logZ, logU = gal.bin_logZ_logU(params)

    import matplotlib.pyplot as plt

    for i in xrange(len(lines)):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        im = ax.imshow(flux[:,i].reshape(binmap.data.shape), interpolation='nearest',
                       origin='lower')
        ax.set_title(lines[i])
        plt.colorbar(im)
    plt.show()
