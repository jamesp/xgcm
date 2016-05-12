# python 3 compatiblity
from __future__ import print_function, division

# make some functions for taking divergence
from dask import array as da
import xarray
import numpy as np

def _append_to_name(array, append):
    try:
        return array.name + "_" + append
    except TypeError:
        return append


class GCMDataset(object):
    """Representation of GCM (General Circulation Model) output data, numerical
    grid information, and operations related to finite-volume analysis.
    """

    def __init__(self, ds):
        """Initialize GCM object.

        Parameters
        ----------
        ds : xarray Dataset
            A template dataset to determine coordinates in use.
        """

        # check that needed variables are present
        for v in self.needed_vars:
            if v not in ds:
                raise KeyError('Needed variable %s not found in dataset' % v)

        self.ds = ds

    # silly functions

    def _get_coords_from_dims(self, dims, replace=None):
        """Utility function for quickly fetching coordinates from parent
        dataset.
        """
        dims = list(dims)
        if replace:
            for k in replace:
                dims[dims.index(k)] = replace[k]
        return {dim: self.ds[dim] for dim in dims}, dims

    def _get_hfac_for_array(self, array):
        """Figure out the correct hfac given array dimensions."""
        hfac = None
        if 'X' in array.dims and 'Y' in array.dims and 'HFacC' in self.ds:
            hfac = self.ds.HFacC
        if 'Xp1' in array.dims and 'Y' in array.dims and 'HFacW' in self.ds:
            hfac = self.ds.HFacW
        if 'X' in array.dims and 'Yp1' in array.dims and 'HFacW' in self.ds:
            hfac = self.ds.HFacS
        return hfac

    ### Vertical Differences, Derivatives, and Interpolation ###

    def get_vertical_coord_name(self, array):
        """Find the vertical coord this is aligned to."""
        zs = self._vertical_coords.intersection(array.coords)
        if len(zs) != 1:
            raise ValueError('Array %r has no vertical coordinate (or more than one).' % array.name)
        return zs.pop()

    def pad_vertical(self, array, fill_value=0.0, new_coord=None):
        """Pad an array located to be aligned to a different set
        of vertical coordinates.
         An additional fill value is required for the bottom point.

        Parameters
        ----------
        array : xarray DataArray
            The array to difference.
        fill_value : number, optional
            The value to be used at the bottom point.
        new_coord : str, optional
            The name of the other vertical coordinate.
            If not given and there are only two vertical coordinates
            specified in the GCMDataset, use the other one.

        Returns
        -------
        padded : xarray DataArray
            Padded array with vertical coordinate zp1.
        """
        # if there are only two vertical coordinate axes, assuming
        # padding to the other
        orig_coord = self.get_vertical_coord_name(array)

        if new_coord is None:
            if len(self._vertical_coords) == 2:
                new_coord = self._vertical_coords.difference([orig_coord])
            else:
                raise ValueError("No new_coord specified, but there are more than two vertical coordinates to choose from.")

        coords, dims = self._get_coords_from_dims(array.dims)
        zdim = dims.index(orig_coord)
        # shape of the new array to concat at the bottom
        shape = list(array.shape)
        shape[zdim] = 1
        # replace Zl with the bottom level
        coords[orig_coord] = np.atleast_1d(self.ds[new_coord][-1].data)
        # an array of zeros at the bottom
        # need different behavior for numpy vs dask
        if array.chunks:
            chunks = list(array.data.chunks)
            chunks[zdim] = (1,)
            zarr = fill_value * da.ones(shape, dtype=array.dtype, chunks=chunks)
            zeros = xarray.DataArray(zarr, coords, dims).chunk()
        else:
            zarr = np.zeros(shape, array.dtype)
            zeros = xarray.DataArray(zarr, coords, dims)
        newarray = xarray.concat([array, zeros], dim=orig_coord).rename({orig_coord: new_coord})
        if newarray.chunks:
            # this assumes that there was only one chunk in the vertical to begin with
            # how can we do that better
            return newarray.chunk({new_coord: len(newarray[new_coord])})
        else:
            return newarray
