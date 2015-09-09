import numpy as np
from scipy.spatial import cKDTree

from utils import vactoair

class BaseGalaxy(object):

    def __init__(self, ra, dec, z, cosmo, fluxgrid):
        """Base class of galaxy models
        
        Parameters
        ----------
        ra : float
            Right Ascention of galaxy centre [deg]
        dec : float
            Declination of galaxy centre [deg]
        z : float
            Redshift of galaxy
        cosmo: astropy.cosmology object
            cosmology to use, e.g. for calculating luminosity distance
        fluxgrid : metaldisc.fluxgrid object
            fluxgrid object specifying the line-ratio physics

        """

        # Read only params
        self._ra = ra
        self._dec = dec
        self._fluxgrid = fluxgrid
        if (z < 0.):
            raise Exception("Reshift must be greater than zero")
        self._z = z

        # Required attributes
        self.__radius = None
        self.__theta = None
        self.__disc_x = None
        self.__disc_y = None
        self.__bin_area = None

        self.cosmo = cosmo

        #special
        self.__bin_coord = None
        self.__bin_coord_tree = None



    # Make some attributes read only
    @property
    def ra(self):
        """Get right ascention of disc"""
        return self._ra

    @property
    def dec(self):
        """Get declination of disc"""
        return self._dec

    @property
    def z(self):
        """Get redshift of disc"""
        return self._z

    @property
    def fluxgrid(self):
        """Get fluxgrid used in galaxy"""
        return self._fluxgrid


    #require subclasses to have the following attributes
    @property
    def radius(self):
        """Radius of galaxy bins in plane of disc [arcsec]"""
        if self.__radius is None:
            raise NotImplementedError("Subclasses should provide radius attribute")
        else:
            return self.__radius

    @radius.setter
    def radius(self, value):
        self.__radius = value


    @property
    def theta(self):
        """Azimuthal angle of galaxy bins in plane of disc [radians]"""
        if self.__theta is None:
            raise NotImplementedError("Subclasses should provide theta attribute")
        else:
            return self.__theta

    @theta.setter
    def theta(self, value):
        self.__theta = value


    @property
    def disc_x(self):
        """Get x coord of galaxy bins in plane of disc [arcsec]"""
        if self.__disc_x is None:
            raise NotImplementedError("Subclasses should provide disc_x attribute")
        else:
            return self.__disc_x

    @disc_x.setter
    def disc_x(self, value):
        self.__disc_x = value


    @property
    def disc_y(self):
        """y coord of galaxy bins in plane of disc [arcsec]"""
        if self.__disc_y is None:
            raise NotImplementedError("Subclasses should provide disc_y attribute")
        else:
            return self.__disc_y

    @disc_y.setter
    def disc_y(self, value):
        self.__disc_y = value

    @property
    def bin_area(self):
        """Array of galaxy sample bins areas [arcsec^2]"""
        if self.__bin_area is None:
            raise NotImplementedError("Subclasses should provide bin_area attribute")
        else:
            return self.__bin_area

    @bin_area.setter
    def bin_area(self, value):
        self.__bin_area = value
    

    #require subclasses to supply bin_coords
    @property
    def bin_coord(self):
        """Get Nx2 array galaxy sample coords (for N bins) [arcsec]"""
        if self.__bin_coord is None:
            raise NotImplementedError("Subclasses should provide bin_coord attribute")
        else:
            return self.__bin_coord


    #automatically compute bin_coordTree
    @bin_coord.setter
    def bin_coord(self, value):
        self.__bin_coord = value
        self.__bin_coord_tree = cKDTree(value) #construct cKDTree


    @property
    def bin_coord_tree(self):
        """Get scipy.cKDTree representation of bin_coord"""
        if self.__bin_coord_tree is None:
            raise NotImplementedError("Subclasses should provide bin_coord attribute")
        else:
            return self.__bin_coord_tree

    def get_obs_wave(self, lines):
        """Given line name return observed wavelength [Angstrom]

        Notes
        -----
        Redshifts and then converts to air wavelength
        
        Parameters
        ----------
        lines : string or list of strings of line names

        Returns
        -------
        wave : float or array of floats
            wavelength [Angstrom]

        """
        wave = self.fluxgrid.get_wave(lines)

        wave *= (1.+self.z)

        if np.isscalar(wave):
            wave = vactoair(np.array([wave]))[0]
        else:
            wave = vactoair(wave)
        
        return wave


    def incline_rotate(self, x, y):
        """Maps disc x, y coords to observed x, y coords, accounting for
        inclination and position angle effects

        Parameters
        ----------
        x : float array
            disc x-coords [arcsec]
        y : float array
            disc y-coords [arcsec]

        Return
        ------
        new_x : float array
            right ascention [arcsec]
        new_y : float array
            declination [arcsec]

        """

        coords = np.column_stack([x, y])
                
        inc = np.radians(self._inc)
        pa = np.radians(90.+self._pa)

        #rotate about x-axis (inc) then...
        #rotate about z-axis (90+PA) anti-clockwise then...
        #flip horizontal (so that East is Left)
        rot_matrix = np.array([[-np.cos(pa), np.sin(pa)*np.cos(inc)],
                               [np.sin(pa), np.cos(pa)*np.cos(inc)]])

        new_coords = (np.dot(rot_matrix, coords.T)).T

        new_x = new_coords[:,0]
        new_y = new_coords[:,1]

        return new_x, new_y


    def bin_SFR(self, params):
        """Return SFR of galaxy bins [M_sun/yr]"""
        raise NotImplementedError("Subclasses should provide bin_SFR method")


    def _bin_logZ(self, params):
        """Generates metallicity for each bin, given model parameters
        
        Produces an axisymmetric linear metallicity gradient.

        Parameters
        ----------
        params : dict
            Dictionary containing the following:
            Z_in : float
                metallicity at galaxy centre
            Z_out : float
                metallicity at one arcsec from galaxy centre

        Returns
        -------
        logZ : array of floats
            Oxygen abundance of each bin [12 + log10(O/H)]
        
        """

        try:
            Z_in = params['Z_in']
            Z_out = params['Z_out']
        except KeyError, e:
            print "Parameter '{0}' not found".format(e.message)
            raise

        logZ = (Z_out-Z_in) * self.radius + Z_in
        return logZ


    def _bin_logU(self, logZ, params):
        """Generates ionization parameter for each bin, given model parameters
        
        Produces the ionization parameter, adopting an anti-correlation between
        metallicity and ionization parameter

        Parameters
        ----------
        logZ : array of floats
            metallicity of each bin
        params : dict
            Dictionary containing the following:
            logU_0 : float
                ionization paramter at solar metallicity

        Returns
        -------
        logU : array of floats
            Dimensionless Ionization parameter of each bin
        
        """

        try:
            logU_0 = params['logU_0']
        except KeyError, e:
            print "Parameter '{0}' not found".format(e.message)
            raise

        logZ_0 = self.fluxgrid.logZ_solar
        logU = -0.8 * (logZ-logZ_0) + logU_0

        return logU


    def bin_logZ_logU(self, params):
        """Generates metallicity and ionization parameter for each bin, 
        given model parameters

        Metallcity and ionization parameter are clipped between min and max
        of flux grid

        Parameters
        ----------
        params : dict
            dictionary of model parameters
        
        Returns
        -------
        logZ : array of floats
            Oxygen abundance of each bin [12 + log10(O/H)]
        logU : array of floats
            Dimensionless Ionization parameter of each bin
        
        """

        logZ = self._bin_logZ(params)
        logZ = np.clip(logZ, self.fluxgrid.logZ_min, self.fluxgrid.logZ_max)
        logU = self._bin_logU(logZ, params)
        logU = np.clip(logU, self.fluxgrid.logU_min, self.fluxgrid.logU_max)

        return logZ, logU


    def _bin_tauV(self, params):
        """Generates dust attenuation tauV for each bin, given model parameters
        
        Produces an axisymmetric linear profile.

        Parameters
        ----------
        params : dict
            Dictionary containing the following:
            tauV_in : float
                V-band dust extinction at galaxy centre
            tauV_out : float
                V-band dust extinction at one arcsec from galaxy centre

        Returns
        -------
        tauV : array of floats
            Oxygen abundance of each bin [12 + log10(O/H)]
        
        """
        try:
            tauV_in = params['tauV_in']
            tauV_out = params['tauV_out']
        except KeyError, e:
            print "Parameter '{0}' not found".format(e.message)
            raise

        tauV = (tauV_out-tauV_in) * self.radius + tauV_in

        tauV = np.clip(tauV, 0., None) # no negative values

        return tauV


    def apply_bin_extinction(self, lines, flux, var, params):
        """Add dust attenuation effects to fluxes and variances
        

        Parameters
        ----------
        lines : list of strings N
            strings identifing emission lines
        flux : array of floats
            emission line fluxes
        var : array of floats
            corresponding emission line variances
        params : dict
            dictionary of model parameters

        Returns
        -------
        flux_ext : array of floats
            array of fluxes with extinction applied
        var_ext : array of floats
            array of variances with extinction applied
        
        """

        #get wavelength of emission line
        wave = self.fluxgrid.get_wave(lines)

        #get tauV
        tauV = self._bin_tauV(params)
        tauV = np.clip(tauV, 0., None) # no negative values
        
        # Charlot and Fall 2000 model for HII regions   (slope=1.3)
        attenu = np.exp(-tauV[:,None] * (wave/5500.)**-1.3)
        attenu = np.clip(attenu, 0., 1.) # restrict effects to between 0 and 1


        flux_ext = flux * attenu
        var_ext = var * attenu**2.

        return flux_ext, var_ext


    def scale_flux_for_distance(self, flux, var):
        """Accounts flux annuation for distance (reshift)
        
        Parameters
        ----------
        flux : array of floats
            intrinic flux [erg/s]
        var : array of floats
            coresponding variance

        Returns
        -------
        flux_attenu : array of floats
            fluxes attenated fluxs  [erg/s/cm^2]
        var_attenu : array of floats
            coresponding variance
        
        """
        d_lum = self.cosmo.luminosity_distance(self.z).cgs.value #cm
        norm = 4.*np.pi * d_lum**2.
        flux_attenu = flux / norm
        var_attenu = var / norm**2.
        return flux_attenu, var_attenu
   

    def __call__(self, lines, params):
        """Calculate line fluxes and variances for a set of emission lines
        
        Parameters
        ----------
        lines : list of strings
            names identifying emission lines
        params : dict
            dictionary of model parameters

        Returns:
        flux : array of floats, shape:(a,b)
            emission line fluxes, a:#bins b:#lines [erg/s/cm^2]
        var : array of floats, shape:(a,b)
            corresponding variances

        """

        #get metallicity and ionization parameter on bins
        logZ, logU = self.bin_logZ_logU(params)

        #gett SFR of bins
        SFR = self.bin_SFR(params)

        #get line flux
        flux, var = self.fluxgrid(lines, SFR, logZ, logU)
        #apply dust extinction
        flux, var = self.apply_bin_extinction(lines, flux, var, params)

        #apply distance correction
        flux, var = self.scale_flux_for_distance(flux, var)

        return flux, var




class GalaxyDisc(BaseGalaxy):
    def __init__(self, ra, dec, z, pa, inc, r_max, n_annuli, cosmo, fluxgrid):  
        """2D Galaxy Disc model
        
        Create a galaxy disc model with a fixed specified geometry
        Model simulates a galaxy as a set of annular segments
        
        Parameters
        ----------
        ra : float
            Right Ascention of galaxy centre [deg]
        dec : float
            Declination of galaxy centre [deg]
        z : float
            Redshift of galaxy
        pa : float
            postition angle of galaxy disc [deg], North=0, East=90
        inc : float
            inclination of galaxy disc [deg]
        r_max : float
            max radius of galaxy disc [arcsec]
        n_annuli : int
            number of annular bins used to describe galaxy
        cosmo: astropy.cosmology object
            cosmology to use, e.g. for calculating luminosity distance
        fluxgrid : metaldisc.fluxgrid object
            fluxgrid object specifying the line-ratio physics
        
        """
       
        super(GalaxyDisc, self).__init__(ra, dec, z, cosmo, fluxgrid)

        #protect params from being overwritten without updating geometry
        self._pa = pa
        self._inc = inc
        self._r_max = r_max
        self._n_annuli = n_annuli

        self._set_bin_coords() #initialize geometry


    # Make sure bin coords are read only
    #position angle
    @property
    def pa(self):
        """Get position angle of disc"""
        return self._pa

    #inclination
    @property
    def inc(self):
        """Get inclination of disc"""
        return self._inc

    #max radius
    @property
    def r_max(self):
        """Get max radius of disc"""
        return self._r_max

    #no. annular bins
    @property
    def n_annuli(self):
        """Get number of annuli in disc model"""
        return self._n_annuli


    @staticmethod
    def calc_bin_positions(r_max, n_annuli):
        """Calculate annular segment bins
        
        Generate N=n_annuli annular between r=0 and r=r_max

        Innermost bin has 6 azziumthal samples, the next has 12, then 18 etc..
        This pattern gives the radial and transverse directions similar
        separataion.

        Parameters
        ----------
        r_max : float
            max radius of disc [e.g. arcsec]
        n_annuli : int
            number of annular bins

        Returns
        -------
        x : float 1D-array
            x coord of bin centre
        y : float 1D-array
            y coord of bin centre
        radius : float 1D-array
            radial coord of bin centre
        theta : float 1D-array
            azimuthal coord of bin centre
        r_in : float 1D-array
            inner radius of bin edge
        r_out : float 1D-array
            outer radius of bin edge
        d_theta: float 1D-array
            azimuthal angle spanned by bin


        Notes
        -----
        Returns a flat array of bins in sequence spiraling outwards

        """

        # radius of annuli
        r_edge =  np.linspace(0., r_max, n_annuli+1, dtype=float)

        # bin centre
        r = (r_edge[:-1] + r_edge[1:]) / 2.

        # inner and outer edges of annuli
        r_in = r_edge[:-1]
        r_out = r_edge[1:]
        
        n = 6 * (np.arange(n_annuli, dtype=int) + 1)# no. bins per annulus

        radius = np.repeat(r, n) # bin radii
        N = np.repeat(n, n) # number of bins at radius
        r_in = np.repeat(r_in, n)
        r_out = np.repeat(r_out, n)

        # bin ID number within each annulus
        I = np.concatenate([np.arange(i, dtype=float) for i in n])

        d_theta = 2. * np.pi / N
        theta = I * d_theta  #angle to bin

        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        return x, y, radius, theta, r_in, r_out, d_theta


    def _set_bin_coords(self):   
        """Set bin coordinates for model sampling.

        """

        #calculate bin geometry
        out = self.calc_bin_positions(self._r_max, self._n_annuli)
        x, y, radius, theta, r_in, r_out, d_theta = out

        self.disc_x = x
        self.disc_y = y
        self.radius = radius
        self.theta = theta #x,y=(0,0) theta = 0 and x,y=(0,1) theta=pi/2
        #area of annular segments
        self.bin_area = d_theta / 2. * (r_out ** 2. - r_in ** 2.)

        #calculate projected bin coords on sky
        x, y = self.incline_rotate(x, y)
        self.bin_coord = np.column_stack([x, y])


    def bin_SFR(self, params):
        """Calculate the SFR of galaxy bins

        Produces an exponential SF disc

        Parameters
        ----------
        params : dict
            Dictionary containing the following:
            SFRtotal : float
                Total star formation rate of model [M_sun/yr]
            r_d : float
                Disc scale length

        Returns
        -------
        SFR : array of floats
            SFR of bins [M_sun/yr]

        """

        try:
            SFRtotal = params['SFRtotal']
            r_d = params['r_d']
        except KeyError, e:
            print "Parameter '{0}' not found".format(e.message)
            raise

        #get central normalization of SFR
        const = (2.*np.pi*r_d) * (r_d - (np.exp(-r_max/r_d) * (r_d+r_max)))
        SFRdensity_0 = SFRtotal / const

        #exponetial disc
        SFRdensity = SFRdensity_0 * np.exp(-self.radius/r_d) # M_sun/yr/arcsec^2

        #account for bin size
        SFR = SFdensity * self.bin_area # M_sun/yr

        return SFR


#class GalaxyMap(BaseGalaxy):
#    def __init__(self, sfrmap, ra, dec, z, pa, inc, oversample, cosmo, fluxgrid):  
#        """2D Galaxy model using a SFR map
#        
#        Create a galaxy disc model with a fixed SFR
#        
#        Parameters
#        ----------
#        sfrmap: astropy.io.fits.ImageHDU
#            SFR map with WCS header [M_sun/yr]
#        ra : float
#            Right Ascention of galaxy centre [deg]
#        dec : float
#            Declination of galaxy centre [deg]
#        z : float
#            Redshift of galaxy
#        pa : float
#            postition angle of galaxy disc [deg], North=0, East=90
#        inc : float
#            inclination of galaxy disc [deg]
#        oversample : int
#            factor by which to oversample the input SFR map
#        cosmo: astropy.cosmology object
#            cosmology to use, e.g. for calculating luminosity distance
#        fluxgrid : metaldisc.fluxgrid object
#            fluxgrid object specifying the line-ratio physics
#        
#        """
#
#        super(GalaxyDisc, self).__init__(ra, dec, z, cosmo, fluxgrid)
#
#        
#
#        #protect params from being overwritten without updating geometry
#        self._pa = pa
#        self._inc = inc
#        self._oversample = oversample
#
#        self._set_SFR_map(sfrmap) #initialize geometry
#
#
#    # Make sure bin coords are read only
#    #position angle
#    @property
#    def pa(self):
#        """Get position angle of disc"""
#        return self._pa
#
#    #inclination
#    @property
#    def inc(self):
#        """Get inclination of disc"""
#        return self._inc
#
#    #oversample factor
#    @property
#    def oversample(self):
#        """Get max radius of disc"""
#        return self._oversample
#
#
#    def _set_SFR_map(self, sfrmap):
#
#        #get map coords and pixel area
#        
#        self.disc_x = x
#        self.disc_y = y
#        self.radius = np.sqrt(x**2. + y**2.)
#        self.theta = np.arctan2(y,x) % (2.*np.pi) #interval [0, 2*pi)
#        self.bin_area = 
#
#        #calculate projected bin coords on sky
#        x, y = self.incline_rotate(x, y)
#        self.bin_coord = np.column_stack([x, y])
#
#
#    def bin_SFR(self, params):
#        """Calculate the SFR of galaxy bins using an SFR map
#
#        Parameters
#        ----------
#        params : dict
#            Dictionary need not contain anything
#
#        Returns
#        -------
#        SFR : array of floats
#            SFR of bins [M_sun/yr]
#
#        """
#
#        SFR
#        
#        return SFR




if __name__ == '__main__':
    
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=70., Om0=0.3)
    from fluxgrid import FluxGrid
    lines = ['O2-3727', 'O2-3729', 'O3-5007']
    fluxgrid = FluxGrid('grids/grid_Dopita13_kappa=inf.h5', lines, 0.04)
    gal = GalaxyDisc(338.24124, -60.563644, 0.427548, 15., 85., 3., 45, cosmo, fluxgrid)

    params = {
            'SFRtotal': 1.,
            'r_d': 0.5,
            'Z_in': 9.0,
            'Z_out': 8.9,
            'logU_0': -3.,
            'tauV_in': 0.2,
            'tauV_out': 0.1,
            }

    flux, var = gal(lines, params)
    logZ, logU = gal.bin_logZ_logU(params)
    tauV = gal._bin_tauV(params)
    coords = gal.bin_coord

    import matplotlib.pyplot as plt
    plt.scatter(coords[:,0], coords[:,1], c=flux[:,2])
    plt.colorbar()
    
    plt.show()
