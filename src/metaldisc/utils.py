from __future__ import absolute_import
import numpy as np

def vactoair(wave_vac):
    """Convert vacuum wavelengths to air wavelengths

    Based on IDL Astronomy User's Library vactoair
    Formula from Ciddor 1996  Applied Optics , 35, 1566

    Notes
    -----
    Wavelength values below 2000A will not be altered.

    Parameters
    ----------
    wave_vac : array of floats
        vacuum wavelenths to be converted
    
    Results
    -------
    wave_air :array of floats
        air wavelenths

    """
    wave_vac = np.array(wave_vac, dtype=np.float64, copy=True)

    mask = np.argwhere(wave_vac >= 2000.) #only modify above 2000A
    
    wavenum2 = (1e4 / wave_vac)**2. # wavenumber squared
    
    fact = (1.0 +  5.792105e-2 / (238.0185-wavenum2) + 
            1.67917e-3 / (57.362-wavenum2))

    wave_air = wave_vac.copy()
    wave_air[mask] /= fact[mask]

    if wave_vac.ndim == 0:
        return wave_air[0]
    else:
        return wave_air
