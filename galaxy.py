import numpy as np
from cosmolopy import cd


def Galaxy(object):
    def __init__(self, r_max, pa, inc, n_annuli):

        self._r_max = r_max
        self._pa = pa
        self._inc = inc
        self._n_annuli = n_annuli
        self._set_bin_coords()
    
    #make auto update bins on parameter changes
    @property
    def r_max(self):
        return self._r_max

    @r_max.setter
    def r_max(self, value):
        self._r_max = value
        self._set_bin_coords(self.r_max, self.pa, self.inc, self.n_annuli)

    @property
    def r_max(self):
        return self._pa

    @pa.setter
    def pa(self, value):
        self._pa = value
        self._set_bin_coords(self.r_max, self.pa, self.inc, self.n_annuli)

    @property
    def inc(self):
        return self._inc

    @inc.setter
    def inc(self, value):
        self._inc = value
        self._set_bin_coords(self.r_max, self.pa, self.inc, self.n_annuli)

    @property
    def n_annuli(self):
        return self._n_annuli

    @n_annuli.setter
    def n_annuli(self, value):
        self._n_annuli = value
        self._set_bin_coords(self.r_max, self.pa, self.inc, self.n_annuli)

    def _set_coords(self, r_max, pa, inc, n_annuli):   
        x, y, radius, theta, r_in, r_out, d_theta \
            = self.calc_bin_positions(r_max, n_annuli)

        x, y = self.incline_rotate(x, y, inc, pa)
        self.x = x
        self.y = y
        self.radius = radius
        self.theta = theta
        self.radius_inner = r_in
        self.radius_outer = r_out
        self.d_theta = d_theta

    def calc_bin_positions(self, r_max, n_annuli):
        # radius of annuli
        r, dr = np.linspace(0., r_max, n_annuli, retstep=True dtype=float)

        #calc inner and outer edges of annuli
        r_in = r - dr / 2.
        r_in[0] = 0.
        r_out = r + dr / 2.
        
        
        n = 6 * np.arange(n_annuli, dtype=int) # no. bins per annulus
        n[0] = 1

        R = np.repeat(r, n) # bin radii
        N = np.repeat(r, n) # number of bins at radius
        r_in = np.repeat(r_in, n)
        r_out = np.repeat(r_out, n)
        # bin number within annulus
        I = np.concatenate([np.arange(i, dtype=float) for i in n])

        d_theta = 2. * np.pi / N
        theta = I * d_theta  #angle to bin

        x = R * np.cos(theta)
        y = R * np.sin(theta)

        return x, y, R, theta, r_in, r_out, d_theta

    @staticmethod
    def incline_rotate(x, y, inc, pa):
        inc = np.radians(inc)
        pa = np.radians(pa)

        rot_matrix = np.array([[np.cos(inc) * np.cos(pa), np.sin(pa)],
                               [-np.cos(inc) * np.sin(pa), np.cos(pa)]])

        coords = np.row_stack([x.ravel(), y.ravel()])
        new_coords = np.dot(rot_matrix, coords)
        new_x = new_coords[0].reshape(x.shape)
        new_y = new_coords[1].reshape(y.shape)

        return new_x, new_y


def create_xy_grid(x_sampling, y_sampling, x_max, y_max):
    """Creates regualarly sampled grid centred on (0, 0)

    Args:
        x_sampling (float): spacing between samples in the x dirction
        y_sampling (float): spacing between samples in the y dirction
        x_max (float): defines limits of grid in x direction [-x_max, x_max]
        y_max (float): defines limits of grid in y_dircetion [-y_max, y_max]

    Returns:
        x_grid (array): grid of x coords
        y_grid (array): grid of y coords
    """
    x = np.arange(0., x_max, x_sampling)
    x = np.concatenate([-x[::-1], x[1:]]) #make symmetric about 0

    y = np.arange(0., y_max, y_sampling)
    y = np.concatenate([-y[::-1], y[1:]]) #make symmetric about 0

    x_grid, y_grid = np.meshgrid(x, y)
    
    return x_grid, y_grid


class oldGalaxy(object):
    """Galaxy model producing undegraded 2D emission line maps"""
    def __init__(self, sfrProfile, metallicityProfile, lineFlux, lines=[],
                 xy_sampling=0.05, xy_max=10.):
        """Initialize galaxy

        Kwargs:
            xy_sampling (float): spatial sampling of model (arcsec)
            xy_max (float): max limit of spatial sampling (arcsec)

        Raises:
            ValueError
        """
    
        #Check spatial sampling is positive
        if xy_sampling <= 0.:
            t = ("Illegal input: xy_sampling (spatial sampling) cannot be less "
                 "than 0. Value given z={0:.3d}".format(xy_sampling))
            raise ValueError(t)

        #Check spatial max limit is positive
        if xy_max <= 0.:
            t = ("Illegal input: xy_max (spatial limit) cannot be less "
                 "than 0. Value given z={0:.3d}".format(xy_sampling))
            raise ValueError(t)

        if xy_sampling >= xy_max:
           t = ("Illegal input: xy_max should be greater than xy_sampling")
           raise ValueError(t)

        #create and store store spatial grids (arcsec)
        grids = create_xy_grid(xy_sampling, xy_sampling, xy_max, xy_max)
        self.x_grid = grids[0]
        self.y_grid = grids[1]
        self.r_grid = np.sqrt(self.x_grid ** 2. + self.y_grid ** 2.)

        self.x = self.x_grid.ravel()
        self.y = self.y_grid.ravel()
        self.r = self.r_grid.ravel()

        self.pix_area = xy_sampling ** 2. #(arcsec^2)
        
        #store grid info (arcsec)
        self.xy_sampling = xy_sampling
        self.xy_max = xy_max
        
        self.sfrProfile = sfrProfile(self)
        self.metallicityProfile = metallicityProfile(self)
        self.lineFlux = lineFlux(self, lines)

        self.cosmo = {'omega_M_0':0.3, 'omega_lambda_0':0.7, 'h':0.70}
        self.cosmo = cd.set_omega_k_0(self.cosmo)
        

    def kpc_per_arcsec(self, z):
        """Angular diameter distance in units of kpc per arcsec
        
        Args:
            z - redshift
        """
        d_a = cd.angular_diameter_distance(z, **self.cosmo) #Mpc/rad
        d_a *= 1e3 / (np.degrees(1.) * 3600.) #kpc/arcsec
        return d_a

    def luminosity_distance(self, z):
        """Luminosity distance in units of cm
        
        Args:
            z - redshift
        """
        D_L = cd.angular_diameter_distance(z, **self.cosmo) #Mpc
        D_L *= 3.08567758e24 #cm
        return D_L

    def model(self, params):
        """Produces emission line maps of galaxy"""
        inc = params['inc'] 
        pa = params['PA'] 
        
        SFR = self.sfrProfile(params)
        Z = self.metallicityProfile(params)
        
        mask = ~np.isnan(SFR)
        SFR = SFR[mask]
        Z = Z[mask]
        lineflux, lines = self.lineFlux(SFR, Z, params)

        x, y = self.incline_rotate(self.x[mask], self.y[mask], inc, pa)

        dt = np.dtype([('x', float), ('y', float)] + lineflux.dtype.descr)
        out = np.zeros(x.size, dtype=dt)
        out['x'] = x
        out['y'] = y
        for name, _ in lineflux.dtype.descr:
            out[name] = lineflux[name]

        return out, lines

     
    @staticmethod
    def incline_rotate(x, y, inc, pa):
        inc = np.radians(inc)
        pa = np.radians(pa)

        rot_matrix = np.array([[np.cos(inc) * np.cos(pa), np.sin(pa)],
                               [-np.cos(inc) * np.sin(pa), np.cos(pa)]])

        coords = np.row_stack([x.ravel(), y.ravel()])
        new_coords = np.dot(rot_matrix, coords)
        new_x = new_coords[0].reshape(x.shape)
        new_y = new_coords[1].reshape(y.shape)

        return new_x, new_y


    def _exponential_disc_SFR(self, z, Sigma_SFR_centre, r_s):
        """Exponential SFR profile

        Args:
            z - redshift
            Sigma_SFR_centre - SFR density at galaxy centre (M_sol/yr/kpc^2)
            r_s - scale radius of disc (arcsec)
        """
        pix_area_kpc = self.pix_area * self.kpc_per_arcsec(z) ** 2.
        norm = Sigma_SFR_centre * pix_area_kpc
        profile = np.exp(-self.r_grid / r_s) #exp profile

        return norm * profile # M_sol/yr

    def _linear_metallicity_profile(self, r_s, Z_in, Z_out):
        """Linear Metallicity_Profile

        Args:
            r_s - scale radius of disc (arcsec)
            Z_in - metallicity at centre of galaxy (12 + log(O/H))
            Z_out - metallicity at one scale radius from centre (12 + log(O/H))
        """
        Z = (Z_out - Z_in) * (self.r_grid / r_s) + Z_in
        return Z


    def _placeholder_calc_line_flux(self, Z, SFR, line):
        #SFR(Halpha) (M_sol/yr) = 7.9x10^-42 L(Halpha) (erg/s)
        flux = SFR / 7.9e-42 #erg/s
        if line == 'H_BETA':
            flux /= 2.7
        var = flux / 10.
        return flux, var
    
    #return lines, wave, pixtable(x,y,flux1,err1,flux2,err2)


#PARAMS
#ra - ra center of galaxy
#dec - dec center of galaxy
#z - redshift
#R_s - radial scale length of disc (arcsec)
#SFR - central SFR
#Z_in - central metallicity
#Z_out - metallicity at scale length
#inc - inclination
#PA - PA of major axis


#seeing - FWHM seeing at a wavelength

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    gal = Galaxy()
    params = {'z': 0.5,
              'Sigma_SFR_centre': 0.1,
              'r_s': 1.0,
              'inc': 70.,
              'PA': 10.0,
              'Z_in': 9.0,
              'Z_out': 8.0,}
    out = gal.model(params)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal')
    s = ax.hexbin(out['x'], out['y'], C=np.log10(out['H_ALPHA']), cmap='RdYlBu_r')
    plt.colorbar(s)

    plt.show()


