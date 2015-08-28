import numpy as np
from scipy.spatial import cKDTree

import utils

class BaseGalaxy(object):

    def __init__(self, z, cosmo, fluxgrid):
        """Base class of galaxy models
        
        Parameters
        ----------
        z : float
            Redshift of galaxy
        cosmo: astropy.cosmology object
            cosmology to use, e.g. for calculating luminosity distance
        fluxgrid : metaldisc.fluxgrid object
            fluxgrid object specifying the line-ratio physics

        """
        if (z < 0.):
            raise Exception("Reshift must be greater than zero")
        self.z = z
        self.cosmo = cosmo
        self.fluxgrid = fluxgrid

        self._bin_coord = None
        self._bin_coord_tree = None

        self.__observers = []


    def register_observer(self, observer):
        """Record objects that create views of the model"""
        self.__observers.append(observer)
    
    def notify_observers(self, *args, **kwargs):
        """Notify objects that create views that the model has changed config"""
        for observer in self.__observers:
            observer.notify(self, *args, **kwargs)

    #require subclasses to have the following attributes
    @property
    def radius(self):
        """Radius of galaxy bins in plane of disc [arcsec]"""
        raise NotImplementedError("Subclasses should provide radius attribute")

    @property
    def theta(self):
        """Azimuthal angle of galaxy bins in plane of disc [deg]"""
        raise NotImplementedError("Subclasses should provide theta attribute")

    @property
    def disc_x(self):
        """x coord of galaxy bins in plane of disc [arcsec]"""
        raise NotImplementedError("Subclasses should provide disc_x attribute")

    @property
    def disc_y(self):
        """y coord of galaxy bins in plane of disc [arcsec]"""
        raise NotImplementedError("Subclasses should provide disc_y attribute")

    @property
    def bin_area(self):
        """Array of galaxy sample bins areas [arcsec^2]"""
        raise NotImplementedError("Subclasses should provide bin_area attribute")
    

    #require subclasses to supply bin_coords and trigger updates as required
    @property
    def bin_coord(self):
        """Get Nx2 array galaxy sample coords (for N bins) [deg]"""
        if self._bin_coord is None:
            raise NotImplementedError("Subclasses should provide bin_coord attribute")
        else:
            return self._bin_coord

    #automatically compute bin_coordTree on update
    @bin_coord.setter
    def bin_coord(self, value):
        """Set Nx2 array galaxy sample coords (for N bins) [deg]
        
        Forces update of cKDTree repesentation and notifies observers
        """
        self._bin_coord = value
        self._bin_coord_tree = cKDTree(value) #construct cKDTree
        self.notify_observers(bins_changed=True) #trigger notification


    @property
    def bin_coord_tree(self):
        """Get scipy.cKDTree representation of bin_coord"""
        if self._bin_coordTree is None:
            raise NotImplementedError("Subclasses should provide bin_coord attribute")
        else:
            return self._bin_coord_tree


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
            r_d : float
                disc scale length
            Z_in : float
                metallicity at galaxy centre
            Z_out : float
                metallicity at one disc scale length

        Returns
        -------
        logZ : array of floats
            Oxygen abundance of each bin [12 + log10(O/H)]
        
        """

        try:
            r_d = params['r_d']
            Z_in = params['Z_in']
            Z_out = params['Z_out']
        except KeyError, e:
            print "Parameter '{0}' not found".format(e.message)
            raise

        logZ = (Z_out-Z_in) * (self.radius/r_d) + Z_in
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

        logU = -0.8 * logZ + logU_0

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
            r_d : float
                disc scale length
            tauV_in : float
                V-band dust extinction at galaxy centre
            tauV_out : float
                V-band dust extinction at one disc scale length

        Returns
        -------
        tauV : array of floats
            Oxygen abundance of each bin [12 + log10(O/H)]
        
        """
        try:
            r_d = params['r_d']
            tauV_in = params['tauV_in']
            tauV_out = params['tauV_out']
        except KeyError, e:
            print "Parameter '{0}' not found".format(e.message)
            raise

        tauV = (tauV_out-tauV_in) * (self.radius/r_d) + tauV_in

        return tauV


    def apply_bin_extinction(self, line, flux, var, params)
        """Add dust attenuation effects to fluxes and variances
        

        Parameters
        ----------
        lines : list of strings
            strings identifing emission line
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
        wave = [self.fluxgrid.get_wave(l) for l in lines]

        #get tauV
        tauV = self._bin_tauV(params)
        tauV = np.clip(tauV, 0., None) # no negative values
        
        # Charlot and Fall 2000 model for HII regions   (slope=1.3)
        ext = np.exp(-tauV * (wave/5500.)**-1.3)
        ext = np.clip(ext, 0., 1.) # restrict effects to between 0 and 1

        flux_ext = flux * ext
        var_ext = var * ext**2.

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
        var_attenu var / norm**2.
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
       
        super(GalaxyDisc, self).__init__(z, cosmo, fluxgrid)

        #protect params from being overwritten without updating geometry
        self._ra = ra
        self._dec = dec
        self._pa = pa
        self._inc = inc
        self._r_max = r_max
        self._n_annuli = n_annuli

        self.geometry_changed = True #bins require updating
        self._update_bin_coords() #initialize geometry


# Make sure bin coords recomputed if following geometry parameters are changed
    #right ascention
    @property
    def ra(self):
        """Get right ascention of disc"""
        return self._ra

    @ra.setter
    def ra(self, value):
        """Set right ascention of disc"""
        self._ra = value
        self.geometry_changed = True #bins require updating

    #declination
    @property
    def dec(self):
        """Get declination of disc"""
        return self._dec

    @dec.setter
    def dec(self, value):
        """Set declination of disc"""
        self._dec = value
        self.geometry_changed = True #bins require updating

    #position angle
    @property
    def pa(self):
        """Get position angle of disc"""
        return self._pa

    @pa.setter
    def pa(self, value):
        """Set position angle of disc"""
        self._pa = value
        self.geometry_changed = True #bins require updating

    #inclination
    @property
    def inc(self):
        """Get inclination of disc"""
        return self._inc

    @inc.setter
    def inc(self, value):
        """Set inclination of disc"""
        self._inc = value
        self.geometry_changed = True #bins require updating

    #max radius
    @property
    def r_max(self):
        """Get max radius of disc"""
        return self._r_max

    @r_max.setter
    def r_max(self, value):
        """Set max radius of disc"""
        self._r_max = value
        self.geometry_change = True #bins require updating

    #no. annular bins
    @property
    def n_annuli(self):
        """Get number of annuli in disc model"""
        return self._n_annuli

    @n_annuli.setter
    def n_annuli(self, value):
        """Set number of annuli in disc model"""
        self._n_annuli = value
        self.geometry_changed = True #bins require updating

# END of geometry parameters

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
        #multiply SFR of bins
        SFR = self.bin_SFR()

        return x, y, radius, theta, r_in, r_out, d_theta


    def transform_to_sky(self, x, y):
        """Maps disc x and y coords to sky RA and Dec

        Includes inclination and position angle effects

        Parameters
        ----------
        x : float array
            disc x-coords [arcsec]
        y : float array
            disc y-coords [arcsec]

        Return
        ------
        ra : float array
            right ascention [deg]
        dec : float array
            declination [deg]

        """

        coords = np.column_stack([x, y])
                
        inc = np.radians(self._inc)
        pa = np.radians(90.+self._pa)

        #rotate about x-axis (inc) then...
        #rotate about z-axis (90+PA) anti-clockwise then...
        #flip horizontal (so that East is Left)
        rot_matrix = np.array([[-np.cos(pa), np.sin(pa)*np.cos(inc)],
                               [np.sin(pa), np.cos(pa)*np.cos(inc)]])

        new_coords = np.dot(rot_matrix, coords.T)

        ra, dec = utils.separation_to_radec(new_coords[:,0], new_coords[:,1],
                                            self._ra, self._dec)
        return ra, dec


    def _update_bin_coords(self):   
        """Set bin coordinates for model sampling
        
        Notes
        -----
        Work is only done if (self.geometry_changed == True)

        Return
        ------
        flag : bool
            True if recomputation was performed. False if skipped

        #multiply SFR of bins
        SFR = self.bin_SFR()
        """

        #only recompute if geometry of system has changed
        if self.geometry_changed:

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
            ra, dec = self.transform_to_sky(x, y)
            self.bin_coords = np.column_stack([ra, dec])

            #Reset flag
            self.geometry_changed = False
            return True

        else:
            return False


    def bin_SFR(self, params):
        """Calculate the SFR of galaxy bins

        Produces an exponential SF disc

        Parameters
        ----------
        params : dict
            Dictionary containing the following:
            SFdensity_0 : float
                Central star formation density [M_sun/yr/kpc^2]
            r_d : float
                Disc scale length

        Returns
        -------
        SFR : array of floats
            SFR of bins [M_sun/yr]
        """

        try:
            SFdensity_0 = params['SFdensity_0']
            r_d = params['r_d']
        except KeyError, e:
            print "Parameter '{0}' not found".format(e.message)
            raise


        #exponetial disc with central scaling
        SFdensity = SFdensity_0 * np.exp(-self.radius/r_d) # M_sun/yr/kpc^2

        kpc_per_arcsec = 1. / self.cosmo.arcsec_per_kpc_proper(self.z).value
        area_kpc = self.bin_area * kpc_per_arcsec**2. # kpc^2

        SFR = SFdensity * area_kpc # M_sun / yr

        return SFR
