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