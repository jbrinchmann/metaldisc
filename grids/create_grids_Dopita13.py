from collections import OrderedDict
import numpy as np
import h5py
from astropy.table import Table




def load_data(kappa):
    """Loads a data table for a specified kappa

    Parameters
    ----------
    kappa : 10, 20, 50, 'inf'
    
    Returns
    -------
    logZ : M array of floats
        log10(metallicity) [Z_solar]
    logq : N array of floats
        ionization parameter [s/cm]
    lines : P list of strings
        emission line names
    data : MxNxP array of floats
        line fluxes [normalized to Hb=1]

    """

    data_blue = Table.read('raw/Dopita13/table4.dat', format='cds',
                           readme="raw/Doptita13/ReadMe")
    data_red = Table.read('raw/Dopita13/table5.dat', format='cds',
                          readme="raw/Doptita13/ReadMe")

    #check tables correspond 1:1
    x_blue = data_blue[['Z', 'kappa', 'f_kappa', 'logq']]
    x_red = data_blue[['Z', 'kappa', 'f_kappa', 'logq']]
    assert np.all(x_blue == x_red), "Red and Blue table rows do not match"

    if kappa == 'inf':
        mask_kappa = data_blue['f_kappa'] == 'i'
    elif kappa in [10, 20, 50]:
        mask_kappa = data_blue['kappa'] == kappa
    else:
        raise Exception("Invalid value for kappa")

    logZ = np.array([0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 3.0, 5.0])
    logq = np.array([6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5])

    lines_blue = data_blue.colnames[4:]
    lines_red = data_red.colnames[4:]
    lines = lines_blue + lines_red

# 8.69 = solar

#convert Halpha to SFR -> 7.9e-42
#normalize by SFR 
#convert fluxes to solar luminosity 3.846e33 erg/s





def create_grid:


data_blue = Table.read('raw/Dopita13/table4.dat', format='cds', readme="raw/Doptita13/ReadMe")
data_red = Table.read('raw/table5.dat', format='cds', readme="raw/ReadMe")

data_blue = data_blue[data_blue['kappa'] == 20]
data_red = data_red[data_red['kappa'] == 20]

lines_blue = data_blue.colnames[4:]
lines_red = data_red.colnames[4:]
coadd_lines = OrderedDict()
