"""Regular-grid interpolation primitives for axisymmetric numerical fields."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class AxisymmetricFieldTable(NamedTuple):
    """Field components tabulated on a regular (R,Z) grid.

    Component arrays have shape ``(nR, nZ)``. The table is deliberately
    geometry-neutral with respect to the orbit integrator; future generic
    guiding-center backends can consume the interpolated local fields.
    """

    R: jnp.ndarray
    Z: jnp.ndarray
    B_R: jnp.ndarray
    B_Z: jnp.ndarray
    B_phi: jnp.ndarray
    E_R: jnp.ndarray
    E_Z: jnp.ndarray
    E_phi: jnp.ndarray


class AxisymmetricFieldValues(NamedTuple):
    B_R: jnp.ndarray
    B_Z: jnp.ndarray
    B_phi: jnp.ndarray
    E_R: jnp.ndarray
    E_Z: jnp.ndarray
    E_phi: jnp.ndarray


def bilinear_regular_grid(xgrid, ygrid, values, x, y):
    """Bilinear interpolation on monotone regular/nonuniform 1-D axes."""

    ix = jnp.searchsorted(xgrid, x, side="right") - 1
    iy = jnp.searchsorted(ygrid, y, side="right") - 1
    ix = jnp.clip(ix, 0, xgrid.shape[0] - 2)
    iy = jnp.clip(iy, 0, ygrid.shape[0] - 2)
    x0 = xgrid[ix]
    x1 = xgrid[ix + 1]
    y0 = ygrid[iy]
    y1 = ygrid[iy + 1]
    tx = (x - x0) / jnp.maximum(x1 - x0, 1.0e-30)
    ty = (y - y0) / jnp.maximum(y1 - y0, 1.0e-30)
    f00 = values[ix, iy]
    f10 = values[ix + 1, iy]
    f01 = values[ix, iy + 1]
    f11 = values[ix + 1, iy + 1]
    return (
        (1.0 - tx) * (1.0 - ty) * f00
        + tx * (1.0 - ty) * f10
        + (1.0 - tx) * ty * f01
        + tx * ty * f11
    )


def sample_axisymmetric_table(table: AxisymmetricFieldTable, R, Z):
    def interp(a):
        return bilinear_regular_grid(table.R, table.Z, a, R, Z)

    return AxisymmetricFieldValues(
        interp(table.B_R),
        interp(table.B_Z),
        interp(table.B_phi),
        interp(table.E_R),
        interp(table.E_Z),
        interp(table.E_phi),
    )
