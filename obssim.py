import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from astropy import wcs
from astropy import constants as const
from astropy import units

    
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
        self.conserve_flux = conserve_flux

        #Read only params
        self._galaxy = galaxy
        self._seeing = seeing

        # Required attributes
        self.__pixel_id = None
        self.__pixel_area = None

        #special
        self.__pixel_coord = None
        self.__pixel_coord_tree = None
        
        #store for flux mappings
        self._gal2pix = {}
        self._pix2bin = None

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


#    def _calc_pixel2bin_matrix(self):
#        """Constructs matrix mapping pixel to bins mulitplied by pixel area
#
#        Notes
#        -----
#        if self.conserving_flux=False then bins are normalized to number of pixels contributing to the bin
#
#        Returns
#        -------
#        m : CSR sparse matrix
#
#        """
#
#        uniq_pixel_id = np.unique(self.pixel_id)
#
#        n_bins = len(uniq_pixel_id)
#        n_pixels = len(self.pixel_id)
#        area_mapper = np.zeros((n_bins, n_pixels), dtype=float)
#
#        for i, id_ in enumerate(uniq_pixel_id):
#            mask = (self.pixel_id == id_)
#            area_mapper[i][mask] = self.pixel_area[mask]
#            if not self.conserve_flux: 
#               area_mapper[i][mask] /= np.sum(mask) #conserve intensity
#
#        m = sparse.csr_matrix(area_mapper)
#        return m
#
#
#    def _calc_psf2pixel_matrix(self, wave):
#        """Constructs matrix mapping galaxy sample elements image pixels
#
#        Calculates element -> pixel distances and computes PSF for each distance
#
#        Parameters
#        ----------
#        wave : float
#            wavelength [Angstrom] at which PSF is computed
#
#        Returns
#        -------
#        m : CSR sparse matrix
#
#        """
#        maxdist = self.seeing.radius_enclosing(0.995, wave)
#
#        #get distance between image pixels and galaxy bins
#        m = self.pixel_coord_tree.sparse_distance_matrix(
#                self.galaxy.bin_coord_tree, maxdist)
#        m = m.tocsr()
#
#        #calc PSF at distances
#        m.data[:] = self.seeing(m.data, wave)
#
#        return m
#
#
#    def calc_mapping_matrix(self, line):
#        """Constructs matrix mapping galaxy sample elements output bins
#        
#        Parameters
#        ----------
#        line : string
#            emission line identification
#
#        Returns
#        -------
#        m : CSR sparse matrix
#
#        """
#        wave = self.galaxy.get_obs_wave(line) #observed wavelength
#
#        #matrix mapping image pixels to image bins multiplied by pixel area
#        pixel2bin = self._calc_pixel2bin_matrix()
#
#        #matrix mapping galaxy bins to image pixels redistributed according to
#        #seeing
#        psf2pixel = self._calc_psf2pixel_matrix(wave)
#
#        mapping_matrix = pixel2bin * psf2pixel
#        return mapping_matrix
#
#
#    def get_mapping_matrix(self, line):
#        """Returns a matrix mapping galaxy sample elements output bins
#
#        Stores matrix for future iterations.
#        
#        Parameters
#        ----------
#        line : string
#            emission line identification
#
#        Returns
#        -------
#        m : CSR sparse matrix
#            flux transform
#        m_var : CSR sparse matrix
#            variance transform
#
#        """
#        try:
#            #check if already computed
#            m = self.mapping_matrix[line]
#            m_var = self.mapping_matrix[line+'_var']
#        except KeyError:
#            #if not, compute now
#            m = self.calc_mapping_matrix(line)
#            m_var = m.copy() #create var matrix
#            m_var.data[:] = m_var.data ** 2.
#            self.mapping_matrix[line] = m
#            self.mapping_matrix[line+'_var'] = m_var
#
#        return m, m_var


    def _calc_gal2pix(self, line):
        """Calculate matrix mapping flux from galaxy elements to sampling pixels

        Calculates element -> pixel distances and computes PSF for each distance

        Parameters
        ----------
        line : string
            names identifying emission line

        Returns
        -------
        map_flux : CSR sparse matrix
            flux transform

        """

        wave = self.galaxy.get_obs_wave(line) #observed wavelength
        maxdist = self.seeing.radius_enclosing(0.995, wave)

        #get distance between image pixels and galaxy bins
        map_flux = self.pixel_coord_tree.sparse_distance_matrix(
                         self.galaxy.bin_coord_tree, maxdist)
        map_flux = map_flux.tocsr()

        #calc PSF at distances
        map_flux.data[:] = self.seeing(map_flux.data, wave)

        return map_flux


    def _calc_pix2bin(self):
        """Calculate matrix mapping flux from sample pixels to output bins

        Notes
        -----
        if self.conserving_flux=False then bins are normalized to number of
        pixels contributing to the bin

        Returns
        -------
        map_flux : CSR sparse matrix
            flux transform

        """

        uniq_pixel_id = np.unique(self.pixel_id)

        n_bins = len(uniq_pixel_id)
        n_pixels = len(self.pixel_id)
        map_flux = np.zeros((n_bins, n_pixels), dtype=float)

        for i, id_ in enumerate(uniq_pixel_id):
            mask = (self.pixel_id == id_)
            map_flux[i][mask] = self.pixel_area[mask]
            if not self.conserve_flux: 
               #conserve intensity, normalize by number of pixels contributing
               map_flux[i][mask] /= np.sum(mask)

        map_flux = sparse.csr_matrix(map_flux)

        return map_flux


    def get_gal2pix(self, line):
        """Matrix mapping flux from galaxy elements to sampling pixels

        Parameters
        ----------
        line : string
            names identifying emission line

        Returns
        -------
        map_flux : CSR sparse matrix
            flux transform

        """

        
        try: #if already computed
            map_flux = self._gal2pix[line]

        except KeyError: #if not, compute now
            map_flux = self._calc_gal2pix(line)
        
        return map_flux


    def get_pix2bin(self):
        """Matrices mapping sample pixels to output bins

        Returns both mapping for both flux

        Returns
        -------
        map_flux : CSR sparse matrix
            flux transform

        """
        #check if already computed
        if self._pix2bin is not None:
            map_flux = self._pix2bin

        else: #if not, compute now
            map_flux = self._calc_pix2bin()
            self._pix2bin = map_flux #store for next time

        return map_flux
    
    
    def __call__(self, lines, params):
        """Calculate line fluxes for a set of emission lines
        
        Parameters
        ----------
        lines : list of strings
            names identifying emission lines
        params : dict
            dictionary of model parameters

        Returns:
        flux : array of floats, shape:(a,b)
            emission line fluxes, a:#bins b:#lines [erg/s/cm^2]

        """

        n_lines = len(lines)

        if len(model_err) != n_lines:
            raise ValueError('length of model_err should match that of lines')

        map_pix2bin  = self.get_pix2bin()
        n_pix = map_pix2bin_flux.shape[1]

        #initalize intermediate arrays
        pix_flux = np.full((n_pix, n_lines), np.nan, dtype=float)

        #calc galaxy model
        gal_flux = self.galaxy(lines, params)

        for i_line, line in enumerate(lines):
            map_gal2pix = self.get_gal2pix(line)
            pix_flux[:,i_line] = map_gal2pix.dot(gal_flux[:,i_line])

        bin_flux = map_pix2bin_flux.dot(pix_flux)

        return bin_flux
            

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


class ImageObsSim(BaseObsSim):
    def __init__(self, shape, pix_size, galaxy, seeing):
        """Create a simulated galaxy image, centred on galaxy centre

        Parameters
        ----------
        shape : 2-tuple of integers
            (#y-pixels, #x-pixels)
        pix_size : float
            pixel size [arcsec]
        galaxy : instance of BaseGalaxy or subclass
            galaxy to simulate observations of
        seeing : instance of BaseSeeing or subclass
            seeing model to be used in simulation
            
        """

        super(ImageObsSim, self).__init__(galaxy, seeing, conserve_flux=True)

        self.shape = shape
        n_y, n_x = shape
        
        self.pixel_id = np.arange(n_x*n_y)+1

        x = np.arange(n_x, dtype=float) - ((n_x-1.) / 2.)
        y = np.arange(n_y, dtype=float) - ((n_y-1.) / 2.)
        
        x *= -pix_size #east left
        y *= pix_size

        x = np.tile(x, n_y)
        y = np.repeat(y, n_x)
        self.pixel_coord = np.column_stack([x, y])
        
        self.pixel_area = np.full((n_x*n_y), pix_size**2., dtype=float)


    def __call__(self, lines, params):
        """Calculate line fluxes for a set of emission lines
        
        Parameters
        ----------
        lines : list of strings
            names identifying emission lines
        params : dict
            dictionary of model parameters

        Returns:
        flux : array of floats, shape:(nx,ny,nl)
            emission line fluxes, nx,ny=shape nl:#lines [erg/s/cm^2]

        """

        flux = super(ImageObsSim, self).__call__(lines, params)
        flux = flux.reshape(self.shape + (len(lines),))

        return flux






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
        'SFRtotal': 1.,
        'r_d': 0.5,
        'Z_in': 9.0,
        'Z_out': 8.9,
        'logU_0': -3.,
        'tauV_in': 0.2,
        'tauV_out': 0.1,
        }

    flux = obssim(lines, params)
    
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
