from __future__ import absolute_import
import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator

class FluxGrid(object):

    dim_order = ['logZ', 'logU', 'line']

    def __init__(self, filename, lines):
        """Fluxgrid interpolator

        Parameters
        ----------
        filename : sting
            file path to fluxgrid
        lines : list of strings
            emission line names to be used

        """

        fh = h5py.File(filename, 'r')

        if len(lines) != len(np.unique(lines)):
            raise Exception("Lines should be a unique set of names")
        self.lines =  np.array(lines)

        #wave
        self.wave = self._load_wave(fh['flux'], lines)

        #load fluxes and construct interp table
        dims, fluxes = self._load_data(fh['flux'], self.lines)
        self.flux_intp = RegularGridInterpolator(dims[:2], fluxes,
                method='linear', bounds_error=True)

        self.OH_solar = fh['OH_solar'][()] #12 + log(O/H) solar value

        #Close open hdf5 resources
        fh.flush()
        fh.close()


    def _check_dimension_order(self, dset):
        """Check hdf5 dataset has expected dimensionallity
        
        Parameters
        ----------
        dset : h5py.Dataset
            dataset to check

        Raises
        ------
        Exception : if wrong number or dimensions or wrong dimension labels
        
        """

        dim_order = self.dim_order

        #check no. dimensions
        if len(dset.dims) != len(dim_order):
            msg = ("Dataset {name} has wrong number of dimensions.\n"
                   "Has {n1} dimensions not {n2} dimensions.")
            msg.format(name=dset.name, n1=len(dset.dims), n2=len(dim_order))
            raise Exception(msg)

        #check dimension names
        for i_dim, expected_label in enumerate(dim_order):
            if dset.dims[i_dim].label != expected_label:
                msg = ("Dataset {name} has wrong dimensions.\n"
                       "Dim {i_dim} is labelled {l1} not {l2}.")
                msg.format(name=dset.name, i_dim=i_dim,
                           l1=dset.dims[i_dim].label, l2=expected_lable)
                raise Exception(msg)

    def _get_line_names(self, dset):
        """Minor helping routine to handle changes in layout"""
        line_name = [s.decode('ascii') for s in dset.dims[2]['name'][:]]
        return line_name

    def _load_data(self, dset, lines):
        """
        Given a set of emission line names return subcube of dataset

        Parameters
        ----------
        dset : h5py.Dataset
            dataset to check
        lines : array-like list of strings
            list of emission lines

        Returns
        ------
        dims : list of arrays
            dimension scales of each axis
        data : array of floats
            datagrid corresponding to lines
        
        """

        self._check_dimension_order(dset)
        
        # load scales
        self.logZ = dset.dims[0]['logZ'][:]
        self.logU = dset.dims[1]['logU'][:] 
        line_name = self._get_line_names(dset)

        dims = (self.logZ, self.logU, lines)
        shape = [len(i) for i in dims]
        data = np.full(shape, np.nan, dtype=float)
        for i_line, line in enumerate(self.lines):
            # See _load_wave on the change of logic.
            try:
                idx = line_name.index(line)
            except:
                idx = -1
            if idx < 0:
                raise Exception('Line {0} not found'.format(line))

            data[:,:,i_line] = np.squeeze(dset[:,:,idx])
        return dims, data


    
    def _load_wave(self, dset, lines):
        """
        Given a set of emission line names lookup wavelength in dataset

        Parameters
        ----------
        dset : h5py.Dataset
            dataset to check
        lines : array-like list of strings
            list of emission lines

        Returns
        ------
        wave : array of floats
            wavelength corresponding to lines [Angstrom]
            

        """
        self._check_dimension_order(dset)

        line_name = self._get_line_names(dset)
        line_wave = dset.dims[2]['wave'][:]

#        print("Lines known = ", line_name)
#        print("Waves known = ", line_wave)
        
        wave = np.full(len(lines), np.nan, dtype=float)

        for i_line, line in enumerate(lines):
            # Logic now altered as the code now uses a string array. This
            # then does _not_ check for multiple matches which is
            # a breaking change relative to earlier versions but hopefully
            # a very unlikely one.
            try:
                idx = line_name.index(line)
            except:
                idx = -1
            #            idx = np.where(line_name == line)[0]
            #            if idx.size == 0:
            if idx < 0:
                raise Exception('Line {0} not found'.format(line))
            #            if idx.size >= 2:
            #                raise Exception('Line {0} found more than once'.format(line))

            wave[i_line] = line_wave[idx]

        return wave

        
    @property
    def logZ_min(self):
        """Min logZ value spanned by grid [relative to solar]"""
        return np.min(self.logZ)

    @property
    def logZ_max(self):
        """Max logZ value spanned by grid [relative to solar]"""
        return np.max(self.logZ)

    @property
    def logU_min(self):
        """Min logU value spanned by grid"""
        return np.min(self.logU)

    @property
    def logU_max(self):
        """Max logU value spanned by grid"""
        return np.max(self.logU)
       
    def get_wave(self, line):
        """Given line name return wavelength [Angstrom]
        
        Parameters
        ----------
        line : string or list of strings of line names

        Returns
        -------
        wave : float or array of floats
            wavelength [Angstrom]

        """
        if np.isscalar(line):
            try:
                ind = np.where(self.lines == line)[0][0]
            except IndexError:
                raise Exception("Line {0} not found by FluxGrid".format(line))
        else:
            try:
                ind = [np.where(self.lines == l)[0][0] for l in line]
            except IndexError:
                raise Exception("Line {0} not found by FluxGrid".format(l))
        
        wave = self.wave[ind] 
        return wave


    def __call__(self, lines, SFR, logZ, logU):
        """Get flux for a given line or lines
        
        Parameters
        ----------
        lines : string or array of N strings
            representing line names
        SFR : array of M floats
            Star Formation rate in [M_sun / yr]
        logZ : array of M floats
            Metallicity [12 + log(O/H)]
        logU : array of M floats
            dimensionless ionization parameter

        Returns
        -------
        flux : MxN array of floats
            line fluxes [erg/s]

        """

        if (len(SFR) != len(logZ)) or (len(SFR) != len(logU)):
            raise Exception("SFR, logZ and logU should all have the same length")

        if np.isscalar(lines):
            try:
                ind = [np.where(self.lines == lines)[0][0]]
            except IndexError:
                raise Exception("Line {0} not found by FluxGrid".format(line))
        else:
            try:
                ind = [np.where(self.lines == l)[0][0] for l in lines]
            except IndexError:
                raise Exception("Line {0} not found by FluxGrid".format(l))

        x = np.column_stack([logZ, logU])
        flux = self.flux_intp(x)[:,ind]
        flux *= SFR[:,None]

        return flux
