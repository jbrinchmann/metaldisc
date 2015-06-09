import numpy as np
from scipy.spatial import cKDTree

class Galaxy(object):
    def __init__(self, z, cosmo, r_max, pa, inc, n_annuli, sf_density,
                 metallicity, lineflux, extinction):
        self.__observers = []

        self.z = z
        self.cosmo = cosmo

        self._r_max = r_max
        self._pa = pa
        self._inc = inc
        self._n_annuli = n_annuli

        self._set_bin_coords(r_max, pa, inc, n_annuli)

        self.sf_density = sf_density
        self.metallicity = metallicity
        self.lineflux = lineflux
        self.extinction = extinction


#        sf_density.galaxy = self
#        metallicity.galaxy = self

    def register_observer(self, observer):
        self.__observers.append(observer)
    
    def notify_observers(self, *args, **kwargs):
        for observer in self.__observers:
            observer.notify(self, *args, **kwargs)
 

    @property
    def r_max(self):
        return self._r_max

    @property
    def pa(self):
        return self._pa

    @property
    def inc(self):
        return self._inc

    @property
    def n_annuli(self):
        return self._n_annuli


    def _set_bin_coords(self, r_max, pa, inc, n_annuli):   
        x, y, radius, theta, r_in, r_out, d_theta \
            = self.calc_bin_positions(r_max, n_annuli)

        coords = np.column_stack([x.ravel(), y.ravel()])
        coords = self.incline_rotate(coords, inc, pa)

        self.coords = coords
        self.radius = radius
        self.theta = theta
        self.radius_inner = r_in
        self.radius_outer = r_out
        self.d_theta = d_theta
        self.area = d_theta / 2. * (r_out ** 2. - r_in ** 2.)

        self.coordKDTree = cKDTree(self.coords)
        self.notify_observers(bins_changed=True)

    def calc_bin_positions(self, r_max, n_annuli):
        # radius of annuli
#        r, dr = np.linspace(0., r_max, n_annuli, retstep=True, dtype=float)
        r_edge =  np.linspace(0., r_max, n_annuli+1, dtype=float)

        r = (r_edge[:-1] + r_edge[1:]) / 2.

        #calc inner and outer edges of annuli
#        r_in = r - dr / 2.
#        r_in[0] = 0.
#        r_out = r + dr / 2.
        r_in = r_edge[:-1]
        r_out = r_edge[1:]
        
#        n = 6 * np.arange(n_annuli, dtype=int) # no. bins per annulus
        n = 6 * (np.arange(n_annuli, dtype=int) + 1)# no. bins per annulus
#        n[0] = 1

        R = np.repeat(r, n) # bin radii
        N = np.repeat(n, n) # number of bins at radius
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
    def incline_rotate(coords, inc, pa):
        inc = np.radians(inc)
        pa = np.radians(pa-90.)

        rot_matrix = np.array([[np.cos(pa), np.cos(inc) * np.sin(pa)],
                               [-np.sin(pa), np.cos(inc) * np.cos(pa)]])

        new_coords = np.dot(rot_matrix, coords.T)
        return new_coords.T


    def luminosity_distance(self, z):
        """Luminosity distance in units of cm"""
        d_lum = self.cosmo.luminosity_distance(self.z) #Mpc
        d_lum *= 3.08567758e24 #cm
        return d_lum

    def model(self):
        sf_density = self.sf_density(self.radius, self.r_max)
        sfr = sf_density * self.area

        metal = self.metallicity(self.radius)
        np.clip(metal, 7.5, 9.5, out=metal)

        lines = self.lineflux(sfr, metal)

        tauV = self.extinction(self.radius)
        ext = np.exp(-tauV * (lines['wave'][:,None]/5500.) ** -1.3)
        np.clip(ext, 0., 1., out=ext)
        lines['flux'] *= ext
        lines['var'] *= ext ** 2.

        lines['wave'] *= (1. + self.z)

        d_lum = self.luminosity_distance(self.z)
        norm = 4. * np.pi * d_lum ** 2.
        lines['flux'] /= norm
        lines['var'] /= norm ** 2
        


        return lines

#coords class
#    - update params simulatenously
#    - yields k-cluster?
#SFR class
#    - update SFR, r_d params
#    - call at radius, returns SFdensity
#metallicity model class
#    - update Z_in, Z_out
#    - call at radius
#line physics
#    - call with SFR and Z
#encpuslateing galaxy class
#    - init with redshift, cosmology and classes
#    - use reshift to calc flux
#    - returns coords/k-cluster?, fluxes, vars


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

    from SFdensity import ExpDisc
    from metallicity import LinearMetallicity
    from lineflux import EmpiricalLineFlux

    from astropy.cosmology import FlatLambdaCDM

    sf_density = ExpDisc(1., 1.)
    metallicity = LinearMetallicity(1, 8.5, 8.0)
    filename = '/data2/MUSE/metallicity_calibration/flux_cal_singlevar.h5'
    lineflux = EmpiricalLineFlux(filename, ['H_ALPHA', 'H_BETA'],
                                 ['6562.', 4861.])

    cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

    galaxy = Galaxy(0.5, cosmo, 1., 10., 60., 45, sf_density, metallicity,
                    lineflux)

    fig = plt.figure()
    ax = fig.add_subplot(111, aspect='equal', adjustable='datalim')
    s = ax.scatter(galaxy.coords[:,0], galaxy.coords[:,1], c=galaxy.model()[0]['flux'], s=40, cmap='RdYlBu_r')
    s.set_edgecolor('none')
    plt.colorbar(s)

#    for i in np.arange(-2, 2.1, 0.2):
#        ax.axhline(i, color='grey', zorder=-1)
#        ax.axvline(i, color='grey', zorder=-1)
    

    plt.show()

