"""Implicit RAMc-like radial electric-field evolution."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from core.constants import PI
from core.precision import real_dtype


class ElectricFieldBoundary(NamedTuple):
    kind: str = "conducting"
    wall_radius: float = 1.0


def _laplacian_tridiagonal(r, epsilon: float, ramc_geometry: bool):
    """Coefficients of d2/dr2 + c(r)/r d/dr on a uniform radial grid."""

    dr = r[1] - r[0]
    n = r.shape[0]
    lower = jnp.zeros(n, dtype=real_dtype())
    diag = jnp.zeros(n, dtype=real_dtype())
    upper = jnp.zeros(n, dtype=real_dtype())

    # Regularity at r=0: radial Laplacian -> 2 d2/dr2.
    diag = diag.at[0].set(-2.0 / (dr * dr))
    upper = upper.at[0].set(2.0 / (dr * dr))

    ri = r[1:-1]
    c = jnp.where(ramc_geometry, 1.0 / (1.0 - epsilon * epsilon * ri * ri), 1.0)
    lo = 1.0 / (dr * dr) - c / (2.0 * ri * dr)
    di = -2.0 / (dr * dr)
    up = 1.0 / (dr * dr) + c / (2.0 * ri * dr)
    lower = lower.at[1:-1].set(lo)
    diag = diag.at[1:-1].set(di)
    upper = upper.at[1:-1].set(up)
    return lower, diag, upper


def _apply_boundary(lower, diag, upper, rhs, r, boundary: ElectricFieldBoundary):
    dr = r[1] - r[0]
    if boundary.kind == "conducting":
        lower = lower.at[-1].set(0.0)
        diag = diag.at[-1].set(1.0)
        upper = upper.at[-1].set(0.0)
        rhs = rhs.at[-1].set(0.0)
    elif boundary.kind == "vacuum_robin":
        a = r[-1]
        b = boundary.wall_radius
        robin = 1.0 / jnp.maximum(a * jnp.log(b / a), 1.0e-30)
        lower = lower.at[-1].set(-1.0 / dr)
        diag = diag.at[-1].set(1.0 / dr + robin)
        upper = upper.at[-1].set(0.0)
        rhs = rhs.at[-1].set(0.0)
    else:
        raise ValueError(f"unknown electric-field boundary: {boundary.kind}")
    return lower, diag, upper, rhs


def bdf2_electric_field_step(
    r,
    e_n,
    e_nm1,
    j_np1,
    j_n,
    j_nm1,
    eta_np1,
    eta_n,
    eta_nm1,
    dt_coupling,
    epsilon: float = 0.0,
    current_geometry_factor=None,
    boundary: ElectricFieldBoundary = ElectricFieldBoundary(),
    ramc_geometry: bool = True,
):
    """BDF2 update of the dimensionless RAMc electric-field equation.

    The solved equation is

      dE/dt = eta/(4*pi) L[E] + E d(ln eta)/dt
              - eta G(r) d(j_kin)/dt,

    where ``L`` is the circular-RAMc radial diffusion operator and ``G`` is
    supplied separately. Setting ``G=1`` reproduces the simplified field
    solver in the supplied RAMc source.
    """

    if current_geometry_factor is None:
        current_geometry_factor = jnp.ones_like(r)

    djdt = (3.0 * j_np1 - 4.0 * j_n + j_nm1) / (2.0 * dt_coupling)
    detadt = (3.0 * eta_np1 - 4.0 * eta_n + eta_nm1) / (2.0 * dt_coupling)
    dlneta = detadt / jnp.maximum(eta_np1, 1.0e-30)

    llo, ldi, lup = _laplacian_tridiagonal(r, epsilon, ramc_geometry)
    D = eta_np1 / (4.0 * PI)
    time_diag = 3.0 / (2.0 * dt_coupling)

    lower = -D * llo
    diag = time_diag - D * ldi - dlneta
    upper = -D * lup
    rhs = (4.0 * e_n - e_nm1) / (2.0 * dt_coupling)
    rhs = rhs - eta_np1 * current_geometry_factor * djdt

    lower, diag, upper, rhs = _apply_boundary(
        lower, diag, upper, rhs, r, boundary
    )
    sol = jax.lax.linalg.tridiagonal_solve(lower, diag, upper, rhs[:, None])[:, 0]
    return sol


def backward_euler_electric_field_step(
    r,
    e_n,
    j_np1,
    j_n,
    eta_np1,
    eta_n,
    dt_coupling,
    epsilon: float = 0.0,
    current_geometry_factor=None,
    boundary: ElectricFieldBoundary = ElectricFieldBoundary(),
    ramc_geometry: bool = True,
):
    """Backward-Euler initializer for the first plasma-coupling step."""

    if current_geometry_factor is None:
        current_geometry_factor = jnp.ones_like(r)
    djdt = (j_np1 - j_n) / dt_coupling
    dlneta = (eta_np1 - eta_n) / (
        dt_coupling * jnp.maximum(eta_np1, 1.0e-30)
    )
    llo, ldi, lup = _laplacian_tridiagonal(r, epsilon, ramc_geometry)
    D = eta_np1 / (4.0 * PI)
    lower = -D * llo
    diag = 1.0 / dt_coupling - D * ldi - dlneta
    upper = -D * lup
    rhs = e_n / dt_coupling - eta_np1 * current_geometry_factor * djdt
    lower, diag, upper, rhs = _apply_boundary(lower, diag, upper, rhs, r, boundary)
    return jax.lax.linalg.tridiagonal_solve(lower, diag, upper, rhs[:, None])[:, 0]
