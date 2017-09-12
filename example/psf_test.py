import numpy as np
import matplotlib.pyplot as plt

from astropy.cosmology import FlatLambdaCDM

from metaldisc.fluxgrid import FluxGrid
from metaldisc.galaxy import GalaxyPointSource
from metaldisc.seeing import GaussianPSF, EllipticalGaussianPSF
from metaldisc.obssim import ImageObsSim

cosmo = FlatLambdaCDM(H0=70., Om0=0.3)

ra = 10.
dec = 10.
z = 0.1

lines = ['Ha']
grid = '../grids/grid_Dopita13_kappa=inf.h5'
fluxgrid = FluxGrid(grid, lines)

galaxy = GalaxyPointSource(ra, dec, z, cosmo, fluxgrid)


fwhm_a = 1.0
fwhm_b = 0.9
pa = 90.
seeing1 = EllipticalGaussianPSF([4500., 10000.], [fwhm_a, fwhm_a],
                [fwhm_b, fwhm_b], [pa, pa])
seeing2 = GaussianPSF([4500., 10000.], [fwhm_a, fwhm_a])

obssim1 = ImageObsSim([101,101], 0.05, galaxy, seeing1)
obssim2 = ImageObsSim([101,101], 0.05, galaxy, seeing2)

params = {
    'SFRtotal': 1.,
    'logZ_0': 0.,
    'dlogZ': 0.,
    'logU_sol': -3.,
    'tauV_0': 0.,
    }

flux1 = obssim1(lines, params)
flux2 = obssim2(lines, params)

image1 = flux1[:,:,0]
image2 = flux2[:,:,0]

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal')

im = ax.imshow(image1, origin='lower')
plt.colorbar(im)

plt.show()

