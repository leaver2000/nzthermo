from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import xarray as xr

import nzthermo as nzt

_DATASET = None


def get_data(**sel: Any):
    global _DATASET
    if _DATASET is None:
        _DATASET = xr.open_dataset(
            "data/hrrr.t00z.wrfprsf00.grib2",
            engine="cfgrib",
            backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
        )

    ds = _DATASET.sel(**sel)
    T = ds["t"].to_numpy()  # (K) (Z, Y, X)
    Z, Y, X = T.shape
    N = Y * X
    T = T.reshape(Z, N).transpose()  # (N, Z)
    P = ds["isobaricInhPa"].to_numpy().astype(np.float32) * 100.0  # (Pa)
    Q = ds["q"].to_numpy()  # (kg/kg) (Z, Y, X)
    Q = Q.reshape(Z, N).transpose()  # (N, Z)
    Td = nzt.dewpoint_from_specific_humidity(P, Q)
    # lat = ds["latitude"].to_numpy()
    # lon = (ds["longitude"].to_numpy() + 180) % 360 - 180
    # timestamp = datetime.datetime.fromisoformat(ds["time"].to_numpy().astype(str).item())
    # extent = [lon.min(), lon.max(), lat.min(), lat.max()]

    return (P, T, Td), (Z, Y, X)


@pytest.fixture
def isobaric():
    return get_data()
