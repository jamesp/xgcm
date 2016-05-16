from xgcm import GCMDataset
from xgcm.datasets import manifests

import numpy as np
import xarray as xr
import pytest

import pytest
import os
import tarfile

import xgcm

_TESTDATA_FILENAME = 'test_gfdl.tar.gz'

@pytest.fixture(scope='module')
def gfdl_datadir(tmpdir_factory, request):
    filename = request.module.__file__
    datafile = os.path.join(os.path.dirname(filename), _TESTDATA_FILENAME)
    if not os.path.exists(datafile):
        raise IOError('Could not find data file %s' % datafile)
    # tmpdir_factory returns LocalPath objects
    # for stuff to work, has to be converted to string
    target_dir = str(tmpdir_factory.mktemp('gfdl_data'))
    tar = tarfile.open(datafile)
    tar.extractall(target_dir)
    tar.close()
    return target_dir


def test_gfdl_manifest(gfdl_datadir):
    ds = xr.open_dataset(os.path.join(gfdl_datadir, 'test_gfdl.nc'), decode_times=False)
    dom = GCMDataset(ds)
    assert dom.manifest == manifests.gfdl

def test_periodic_left(gfdl_datadir):
    ds = xr.open_dataset(os.path.join(gfdl_datadir, 'test_gfdl.nc'), decode_times=False)
    dom = GCMDataset(ds)
    ptemp = dom.make_periodic_left(ds.temp)
    assert len(ptemp.coords['lon']) == 129
    assert np.all(ptemp.isel(lon=0) == ptemp.isel(lon=-1))

def test_vertical_diff(gfdl_datadir):
    ds = xr.open_dataset(os.path.join(gfdl_datadir, 'test_gfdl.nc'), decode_times=False)
    dom = GCMDataset(ds)
    # fake a constant lapse rate from surface temp upwards
    lapse_rate = 10.0
    ds['const_lapse'] = (ds.temp.sel(pfull=ds.pfull.max())
                            - lapse_rate*(ds.pfull))
    dTdz = dom.diff(ds.const_lapse, 'z')
    assert np.allclose(ds.const_lapse.diff('pfull') / ds.pfull.diff('pfull'), -lapse_rate)

def test_multiple_dim_coords(gfdl_datadir):
    ds = xr.open_dataset(os.path.join(gfdl_datadir, 'test_gfdl.nc'), decode_times=False)
    dom = GCMDataset(ds)
    ds['single_height'] = ds.pfull * 100.0
    assert dom.get_primary_dim_coord_name('z', ds.single_height) == 'pfull'

    ds['double_height'] = (ds.pfull + ds.phalf)
    with pytest.raises(ValueError):
        dom.get_primary_dim_coord_name('z', ds.double_height)

    ds['no_height'] = ds.lat * 30.0
    with pytest.raises(ValueError):
        dom.get_primary_dim_coord_name('z', ds.no_height)