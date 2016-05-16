# python 3 compatiblity
from __future__ import print_function, division

# make some functions for taking divergence
from dask import array as da
import xarray
import numpy as np
from . import manifests

def _append_to_name(array, append):
    try:
        return array.name + "_" + append
    except TypeError:
        return append


class GCMDataset(object):
    """Representation of GCM (General Circulation Model) output data, numerical
    grid information, and operations related to finite-volume analysis.
    """
    _coord_map = {}
    _y_coords = set()
    _x_coords = set()
    _z_coords = set()

    def __init__(self, ds, manifest=None):
        """Initialize GCM object.

        Parameters
        ----------
        ds : xarray Dataset
            A template dataset to determine coordinates in use.
        """


        # if manifest is not given, see if we can work it out from metadata
        if manifest is None:
            title = ds.attrs.get('title', None)
            if title:
                if title.startswith('FMS Model results'):
                    manifest = manifests.gfdl
            # TODO more ways of auto-detecting which model was used
        if manifest is None:
            raise ValueError("No manifest given, and it can't be determined from the template file.")
        self.manifest = manifest.copy()

        self.ds = ds
        self.coords = ds.coords.to_dataset()

        for coord, attr in manifest.items():
            label = attr['label']
            # map labels e.g. `x_centre` to underlying dataset coordinate names.
            self._coord_map[label] = coord
            if attr['periodic']:
                px = self.make_periodic_left(self.coords[coord])
                dx = px.diff(coord)
            else:
                dx = self.coords[coord].diff(coord)
            self.coords['d%s' % coord] = dx


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

    # helper functions
    def _get_coords_from_dims(self, dims, replace=None):
        """Utility function for quickly fetching coordinates from parent
        dataset.
        """
        dims = list(dims)
        if replace:
            for k in replace:
                dims[dims.index(k)] = replace[k]
        return {dim: self.ds[dim] for dim in dims}, dims

    def _get_coord_name(self, label):
        """For a given label, return the underlying data coord.

        e.g. for GFDL, vertical centres are called 'phalf'
        >>> dom._get_coord_name('z_centre')
        'phalf'

        Returns a string.  If label is not found, returns the label as-is.
        """
        s = self._coord_map.get(label, None)
        if s is None:
            return label
        else:
            return s


    ### Horizontal differencing

    def is_periodic(self, coord):
        coord = self._get_coord_name(coord)
        if coord in self.manifest:
            return self.manifest[coord].get('periodic', False)
        else:
            return False

    def make_periodic_left(self, array, coord=None):
        """Add the rightmost "column" of data to the left of array.
        Parameters
        ----------
        array : xarray DataArray
            The array to make periodic.
        coord : str
            The name of the periodic coordinate. Default: the only periodic coord on the array

        Returns
        -------
        periodic : xarray DataArray
            Array with periodic dimension extended by one data point.
        """
        if coord is None:
            pcoords = [c for c in array.coords if self.is_periodic(c)]
            cs = set(pcoords).intersection(array.coords)
            if len(cs) != 1:
                raise ValueError('Array %r has no periodic coordinate (or more than one). Periodic Coords: %r.' % (array.name, cs))
            else:
                coord = cs.pop()
        else:
            coord = self._get_coord_name(coord)

        if not self.is_periodic(coord):
            raise ValueError("Coordinate %s is not periodic" % coord)
        coords, dims = self._get_coords_from_dims(array.dims)

        # get the last "column" of data
        coords[coord] = np.atleast_1d(array[coord][-1].data)

        xt = array.isel(**{coord:-1})
        # make the coordinate an array if it isn't already
        xt.coords[coord] = np.atleast_1d(xt.coords[coord].data)
        # concat to the left
        pt = xarray.concat([xt, array], dim=coord)
        # amend the new coordinate value (assume equal x spacing for now).
        pt[coord].values[0] = pt[coord].values[1] - pt[coord].values[2]

        return pt


    ### Vertical Differences, Derivatives, and Interpolation ###
    def _dim_coords(self, dim):
        """Return the names of coordinates that have the appropriate dimension.

        e.g. For a gfdl dataset domain:
        >>> dom._dim_coords('z')
        {'phalf', 'pfull'}

        Returns a set of coord names.
        """
        return set(v for k, v in self._coord_map.items() if k.startswith(dim))

    def get_vertical_coord_name(self, array):
        """Find the vertical coord this is aligned to."""
        zs = self._dim_coords('z').intersection(array.coords)
        if len(zs) != 1:
            raise ValueError('Array %r has no vertical coordinate (or more than one) %r.' % (array.name, zs))
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
            if len(self._z_coords) == 2:
                new_coord = self._z_coords.difference([orig_coord])
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
