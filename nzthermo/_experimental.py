from __future__ import annotations

from typing import Annotated, Generic, NamedTuple, TypeVar
import numpy as np


from . import functional as F
from ._c import lcl, moist_lapse
from ._typing import Kelvin, N, Pascal, Z, shape
from .core import dry_lapse

float_ = TypeVar("float_", bound=np.float_)
newaxis = np.newaxis


# -------------------------------------------------------------------------------------------------
# Parcel Profile
# -------------------------------------------------------------------------------------------------
def _parcel_profile(
    pressure: np.ndarray,
    temperature: np.ndarray,
    dewpoint: np.ndarray,
    refrence_pressure: np.ndarray | None = None,
    *,
    axis=-1,
):
    Z = pressure.shape[axis]
    P = np.reshape(np.moveaxis(pressure, axis, -1), (-1, Z))  # (N, Z)
    T = temperature.reshape(-1, 1)  # (N, 1)
    Td = dewpoint.reshape(-1, 1)  # (N, 1)
    N = T.shape[0]

    if refrence_pressure is None:
        P0 = P[:, :1]
    else:
        P0 = refrence_pressure.reshape(-1, 1)  # (N, 1)

    if not T.shape == Td.shape == P0.shape == (N, 1):
        raise ValueError("temperature, dewpoint, and refrence pressure must have the same shape")
    # ---------------------------------------------------------------------------------------------
    # - Find the LCL
    # ---------------------------------------------------------------------------------------------
    lcl_p, lcl_t = lcl(P0, T, Td)  # ((N, 1), (N, 1))

    # - lower
    # [[101300. 100000. 97500. 95000. 94171.484 nan ...], ...] (N, Z)
    p_mask = P >= lcl_p
    p_lower = F.mask_insert(p_mask, P, lcl_p, kind="above", copy=False)  # (N, Z)
    t_lower = dry_lapse(p_lower, T, P0)

    # - upper
    # [[... nan 97500. 95000. 94171.484 90000. ...], ...] (N, Z)
    p_upper = F.mask_insert(~p_mask, P, lcl_p, kind="below", copy=False)  # (N, Z)
    t_upper = np.full_like(t_lower, np.nan)

    nans = np.isnan(t_lower) | np.isnan(p_upper)
    nx, zx = np.nonzero(~nans)
    _, i = np.unique(nx, return_index=True)
    nx, zx = nx[i], zx[i]

    t_upper[nx, :] = moist_lapse(
        p_upper[nx, :],
        t_lower[nx, zx],
        p_upper[nx, zx],
    )

    # OKAY now concat every thing
    index = np.nonzero(p_lower.mask)
    p_lower[index] = p_upper[index]
    t_lower[index] = t_upper[index]

    return (p_lower.squeeze(), lcl_p.squeeze()), (t_lower.squeeze(), lcl_t.squeeze())


class ParcelProfile(NamedTuple, Generic[float_]):
    pressure: Annotated[
        Pascal[np.ndarray[shape[N, Z], np.dtype[float_]]],
        "",
    ]
    lcl_pressure: Annotated[
        Pascal[np.ndarray[shape[N], np.dtype[float_]]],
        "",
    ]

    temperature: Annotated[
        Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
        "",
    ]
    parcel_temperature: Annotated[
        Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
        "",
    ]
    lcl_temperature: Annotated[
        Kelvin[np.ndarray[shape[N], np.dtype[float_]]],
        "",
    ]

    dewpoint: Annotated[
        Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
        "",
    ]


def parcel_profile(
    pressure: Pascal[np.ndarray[shape[Z], np.dtype[float_]]],
    temperature: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    dewpoint: Kelvin[np.ndarray[shape[N, Z], np.dtype[float_]]],
    *,
    pressure_2m: Pascal[np.ndarray[shape[N], np.dtype[float_]] | None] = None,
    temperature_2m: Kelvin[np.ndarray[shape[N], np.dtype[float_]] | None] = None,
    dewpoint_2m: Kelvin[np.ndarray[shape[N], np.dtype[float_]] | None] = None,
) -> ParcelProfile[float_]:
    assert pressure.ndim == 1
    P = pressure.reshape(1, -1)  # (1, Z)
    T, Td = temperature, dewpoint
    Z = T.shape[1]
    Z += 1  # add one for the 2m values

    if pressure_2m is not None:
        P2 = pressure_2m.reshape(-1, 1)
    else:
        P2 = P[:, :1]

    if temperature_2m is None:
        T2 = T[:, :1]
    else:
        T2 = temperature_2m.reshape(-1, 1)

    if dewpoint_2m is None:
        Td2 = Td[:, :1]
    else:
        Td2 = dewpoint_2m.reshape(-1, 1)

    (pp_p, lcl_p), (pp_t, lcl_t) = _parcel_profile(P, T2, Td2, P2, axis=1)
    if pp_t.shape == (N, Z - 1):  # concat the 2m values
        pp_t = np.concatenate([T2, pp_t], axis=1)

    interp_t, interp_td = F.interpolate_nz(lcl_p, pressure, T, Td, interp_nan=True)

    t_new = F.insert_along_z(T, interp_t, pressure, lcl_p)
    td_new = F.insert_along_z(Td, interp_td, pressure, lcl_p)

    return ParcelProfile(
        pressure=pp_p,
        lcl_pressure=lcl_p,
        temperature=pp_t,
        parcel_temperature=t_new,
        lcl_temperature=lcl_t,
        dewpoint=td_new,
    )
