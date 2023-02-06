# Forward modelling of galaxy metallicity profiles

This code provides a method for recovering gas-phase metallicity
gradients from integral field spectroscopic (IFS) observations of
barely resolved galaxies.

The approach here is based on [Carton et
al. 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.2140C and
employs a forward modelling approach to fit the observed spatial
distribution of emission-line fluxes, accounting for the degrading
effects of seeing and spatial binning.  The method is flexible and is
not limited to particular emission lines or instruments, nor one set
specific set of photoionization models.

While this model can be fit in many ways, we adopt a Bayesian approach
with a robust likelihood.


## Installation

It is advisable to install this in a separate virtual environment. The
code currently requires

* `numpy`
* `scipy`
* `pymultinest`
* `h5py`
* `astropy`

and it is advisable to also install `getdist` for visualisation and
`jupyter` to use the notebooks.

If you use `conda`, a possible installation method would be

```
  > conda create -n metaldisc  numpy scipy h5py astropy jupyter pip
   <... Various output ...>
  > conda activate metaldisc
  > pip install pymultinest
  > pip install getdist
  > git clone https://github.com/jbrinchmann/metaldisc.git
  > cd metaldisc
  > pip install . 
  OR
  > pip install -e . 
```

Note in particular that `pymultinest` and `getdist` are most easily
installed using `pip` so that is what is indicated here.

## Examples of use

In the `example` directory, there are two scripts:
 - `fit_from_sfrmap.py` - shows how to fit the example data
 - `model_from_sfrmap.py` - provides a visualisation of the model
   output
 
The `example` directory also contains two notebook versions of
this. The fitting notebook uses `getdist` to
visualise the results in contrast to `fit_from_sfrmap.py` which uses
the code distributed with `pymultinest`.

## Citing
If you use this code, you should cite [Carton et al. 2017](https://ui.adsabs.harvard.edu/abs/2017MNRAS.468.2140C), where the method is explained.

You may also wish to cite [Carton et al. 2018](https://ui.adsabs.harvard.edu/abs/2018MNRAS.478.4293C), where we provide further discussion and improvements.


## GNU GPL v3
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
