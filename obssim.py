import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from astropy import constants as const
from astropy import units

from seeing import CircularPSF, NonCircularPSF
import wcs_utils

    
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
        self._pix2bin = None
        self._gal2pix = {}
        self._gal2bin = {}

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
            msg = "Attribute 'pixel_id' not implemented in class {0}".format(self.__class__)
            raise NotImplementedError(msg)
        else:
            return self.__pixel_id

    @pixel_id.setter
    def pixel_id(self, value):
        self.__pixel_id = value

    @property
    def pixel_area(self):
        """Array of image pixel areas [arcsec^2]"""
        if self.__pixel_area is None:
            msg = "Attribute 'pixel_area' not implemented in class {0}".format(self.__class__)
            raise NotImplementedError(msg)
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
            msg = "Attribute 'pixel_coord' not implemented in class {0}".format(self.__class__)
            raise NotImplementedError(msg)
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
            msg = "Attribute 'pixel_coord' not implemented in class {0}".format(self.__class__)
            raise NotImplementedError(msg)
        else:
            return self.__pixel_coord_tree


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

        #check if already computed
        if self._pix2bin is not None:
            map_flux = self._pix2bin

        else: #if not, compute now
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
            self._pix2bin = map_flux #store for next time

        return map_flux


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
        map_flux.sort_indices()

        if isinstance(self.seeing, CircularPSF):
            #calc PSF at radii
            r = map_flux.data
            y = self.seeing(r, wave)

        elif isinstance(self.seeing, NonCircularPSF):
            #calc PSF at x, y pos
            
            #get indicies works for csr and csc matrices
            major_dim, minor_dim = map_flux._swap(map_flux.shape)
            minor_indices = map_flux.indices
            major_indices = np.empty(len(minor_indices),
                                dtype=map_flux.indices.dtype)
            sparse._sparsetools.expandptr(major_dim, map_flux.indptr,
                                    major_indices)
            idx_pix, idx_gal_bin = map_flux._swap(
                                        (major_indices, minor_indices))

            dist = (self.pixel_coord[idx_pix] -
                    self.galaxy.bin_coord[idx_gal_bin])
                
            dx = dist[:,0]
            dy = dist[:,1]
            y = self.seeing(dx, dy, wave)

        else:
            msg = ("PSF {0} not supported, must be subclass of "
                   "CircularPSF or EllipticalPSF")
            msg = msg.format(self.seeing.__class__)
            raise Exception(msg)

        map_flux.data[:] = y

        return map_flux


    def get_gal2bin(self, line):

        try: #if already computed
            map_flux = self._gal2bin[line]

        except KeyError: #if not, compute now
            pix2bin = self._calc_pix2bin()
            gal2pix = self._calc_gal2pix(line)
            map_flux = pix2bin.dot(gal2pix)
            self._gal2bin[line] = map_flux
        
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

        map_pix2bin = self._calc_pix2bin()
        n_bins = map_pix2bin.shape[0]

        #initalize intermediate arrays
        bin_flux = np.full((n_bins, n_lines), np.nan, dtype=float)

        #calc galaxy model
        gal_flux = self.galaxy(lines, params)

        for i_line, line in enumerate(lines):
            map_gal2bin = self.get_gal2bin(line)
            bin_flux[:,i_line] = map_gal2bin.dot(gal_flux[:,i_line])

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

        mask = (binmap.data >= 1)

        self.pixel_id = binmap.data[mask]

        ra_centre = self.galaxy.ra
        dec_centre = self.galaxy.dec

        x, y = wcs_utils.get_pixel_coord(binmap.header, ra_centre, dec_centre)
        self.pixel_coord = np.column_stack([x[mask], y[mask]])
        
        area = wcs_utils.get_pixel_area(binmap.header, ra_centre, dec_centre)
        self.pixel_area = area[mask]


#        mask = (binmap.data.flatten() >= 1)
#
#        self.pixel_id = binmap.data.flatten()[mask]
#
#        x, y = self.get_pixel_coord(binmap.header)
#        self.pixel_coord = np.column_stack([x[mask], y[mask]])
#        
#        self.pixel_area = self.get_pixel_area(binmap.header)[mask]


#    def calc_rel_coords(self, ra, dec):
#        """Calculates the relative sky coordinates given sky coordinates
#
#        Parameters
#        ----------
#        ra : array of floats
#            right ascention [deg]
#        dec : array of floats
#            declination [deg]
#
#        Returns
#        -------
#        x : array of floats
#            x-axis distances from galaxy centre [arcsec]
#        y : array of floats
#            y-axis distances from galaxy centre [arcsec]
#
#        """
#
#        x = ra - self.galaxy.ra
#        y = dec - self.galaxy.dec
#
#        x -= np.round(x/360.) * 360. # wrap to interval (-180,180)
#
#        x *= np.cos(np.radians(self.galaxy.dec)) * 3600. #deg to arcsec
#        y *= 3600. #deg to arcsec
#
#        return x, y 
#
#
#    def get_pixel_coord(self, hdr, x_offset=0., y_offset=0.):
#        """Get relative pixel coordinates
#
#        Notes
#        -----
#        x_offset=-0.5, y_offset=-0.5 yields coordinates of pixels' lower-left corners
#
#        Parameters
#        ----------
#        hdr : astropy.io.fits.Header instance
#            FITS header including WCS info
#        x_offset : [optional]float
#            x-axis pixel coord offsets
#        y_offset : [optional]float
#            y-axis pixel coord offsets
#
#        Returns
#        -------
#        x : array of floats
#            x coords of relative pixel centres [arcsec]
#        y : array of floats
#            y coords of relative pixel centres [arcsec]
#
#        """
#        #get pixel centres [image coords]
#        n_x = hdr['NAXIS1']
#        n_y = hdr['NAXIS2']
#        pix_x = np.tile(np.arange(n_x, dtype=float), n_y) + x_offset
#        pix_y = np.repeat(np.arange(n_y, dtype=float), n_x) + y_offset
#
#        #convert pixel centres to sky coords
#        wcsobj = wcs.WCS(hdr)
#        coords = np.column_stack([pix_x, pix_y])
#        new_coords = wcsobj.all_pix2world(coords, 0) #[[ra], [dec]]
#
#        #get pixel distances from galaxy [arcsec]
#        x, y = self.calc_rel_coords(new_coords[:,0], new_coords[:,1])
#
#        return x, y
#
#
#    def get_pixel_area(self, hdr):
#        """Set pixel areas
#
#        Parameters
#        ----------
#        hdr : astropy.io.fits.Header instance
#            FITS header including WCS info
#
#        Returns
#        -------
#        area : array of floats
#            pixel areas [arcsec^2]
#
#        """
#
#        #get pixel corners [arcsec]  (clockwise)
#        A_x, A_y = self.get_pixel_coord(hdr, x_offset=-0.5, y_offset=-0.5)
#        B_x, B_y = self.get_pixel_coord(hdr, x_offset=-0.5, y_offset=+0.5)
#        C_x, C_y = self.get_pixel_coord(hdr, x_offset=+0.5, y_offset=+0.5)
#        D_x, D_y = self.get_pixel_coord(hdr, x_offset=+0.5, y_offset=-0.5)
#
#        #get vectors between opposite corners
#        AC_x = C_x - A_x
#        AC_y = C_y - A_y
#
#        BD_x = D_x - B_x
#        BD_y = D_y - B_y
#
#        area = 0.5 * np.abs((AC_x * BD_y) - (BD_x * AC_y))
#
#        return area


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

        self.shape = tuple(shape)
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
