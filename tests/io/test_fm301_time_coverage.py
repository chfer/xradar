#!/usr/bin/env python
# Copyright (c) 2026, openradar developers.
# Distributed under the MIT License. See LICENSE for more info.

"""Tests for FM301 time-coverage root variables."""

import numpy as np
import pytest

import xradar as xd

# One backend per way the root group is built:
# - shared path (_get_required_root_dataset -> _assign_root): odim, gamic,
#   furuno, rainbow, iris, datamet, hpl, metek
# - direct _assign_root callers: nexradlevel2, uf
# - own root implementation: cfradial1, cfradial2
# - IMD root implementation
ENGINES = [
    pytest.param(("odim", "odim_file"), id="odim"),
    pytest.param(("gamic", "gamic_file"), id="gamic"),
    pytest.param(("furuno", "furuno_scn_file"), id="furuno"),
    pytest.param(("rainbow", "rainbow_file"), id="rainbow"),
    pytest.param(("iris", "iris0_file"), id="iris"),
    pytest.param(("datamet", "datamet_file"), id="datamet"),
    pytest.param(("hpl", "hpl_file"), id="hpl"),
    pytest.param(("metek", "metek_ave_gz_file"), id="metek"),
    pytest.param(("nexradlevel2", "nexradlevel2_file"), id="nexradlevel2"),
    pytest.param(("uf", "uf_file_1"), id="uf"),
    pytest.param(("cfradial1", "cfradial1_file"), id="cfradial1"),
    pytest.param(("cfradial2", "cfradial2_file"), id="cfradial2"),
    pytest.param(("imd", "imd_file"), id="imd"),
]


# Return the root dataset and close its owning DataTree after each test.
@pytest.fixture(params=ENGINES)
def root_ds(request):
    engine, fixture_name = request.param
    filepath = request.getfixturevalue(fixture_name)
    dtree = xd.open_datatree(filepath, engine=engine)
    try:
        yield dtree.to_dataset(inherit=False)
    finally:
        dtree.close()


# check the FM301 time coverage attributes on an opened root dataset
@pytest.mark.parametrize("name", ["time_coverage_start", "time_coverage_end"])
def test_time_coverage_fm301_attrs(root_ds, name):
    var = root_ds[name]

    # the variable must be scalar (0-dimensional)
    assert var.ndim == 0

    # the variable must be a double (float64): seconds since the reference time
    assert var.dtype == np.float64, f"{name} must be double, got {var.dtype}"

    # check if the required FM301 attributes are present and correctly typed
    attrs = var.attrs
    for key in ("units", "calendar", "standard_name"):
        assert key in attrs, f"{name} is missing FM301 attribute {key!r}"
        assert isinstance(attrs[key], str), f"{name}:{key} must be a string"

    assert attrs["units"] == "seconds since 1970-01-01T00:00:00Z"
    assert attrs["calendar"] == "standard"
    assert attrs["standard_name"] == "time"
    # Ensure the numeric timestamp is neither NaN nor infinite.
    assert np.isfinite(var.item())
