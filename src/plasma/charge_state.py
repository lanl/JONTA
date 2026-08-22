"""Implicit charge-state evolution using ionization/recombination rates."""

from __future__ import annotations

import jax.numpy as jnp

from core.precision import real_dtype


def charge_state_matrix(ne_m3, ionization_m3_s, recombination_m3_s):
    """Build A for dn/dt = A n along the final charge-state axis."""

    s = ionization_m3_s
    a = recombination_m3_s
    zcount = s.shape[-1]
    shape = s.shape[:-1] + (zcount, zcount)
    A = jnp.zeros(shape, dtype=real_dtype())

    diag = -ne_m3[..., None] * (s + a)
    idx = jnp.arange(zcount)
    A = A.at[..., idx, idx].set(diag)
    # Ionization z -> z+1.
    if zcount > 1:
        z = jnp.arange(zcount - 1)
        A = A.at[..., z + 1, z].set(ne_m3[..., None] * s[..., :-1])
        # Recombination z -> z-1.
        A = A.at[..., z, z + 1].set(ne_m3[..., None] * a[..., 1:])
    return A


def bdf2_charge_state_step(n_n, n_nm1, ne_m3, ionization, recombination, dt_s):
    """BDF2 solve for linear charge-state kinetics at supplied n_e,T_e rates."""

    A = charge_state_matrix(ne_m3, ionization, recombination)
    zcount = n_n.shape[-1]
    eye = jnp.eye(zcount, dtype=real_dtype())
    lhs = (3.0 / (2.0 * dt_s)) * eye - A
    rhs = (4.0 * n_n - n_nm1) / (2.0 * dt_s)
    out = jnp.linalg.solve(lhs, rhs[..., None])[..., 0]
    # Small negative roundoff can occur in stiff solves. Preserve species
    # density exactly after clipping.
    total = jnp.sum(n_n, axis=-1, keepdims=True)
    out = jnp.maximum(out, 0.0)
    out_sum = jnp.sum(out, axis=-1, keepdims=True)
    return jnp.where(out_sum > 0.0, out * total / out_sum, out)


def backward_euler_charge_state_step(n_n, ne_m3, ionization, recombination, dt_s):
    """First-step initializer for BDF2 charge-state evolution."""

    A = charge_state_matrix(ne_m3, ionization, recombination)
    zcount = n_n.shape[-1]
    eye = jnp.eye(zcount, dtype=real_dtype())
    lhs = eye / dt_s - A
    rhs = n_n / dt_s
    out = jnp.linalg.solve(lhs, rhs[..., None])[..., 0]
    total = jnp.sum(n_n, axis=-1, keepdims=True)
    out = jnp.maximum(out, 0.0)
    out_sum = jnp.sum(out, axis=-1, keepdims=True)
    return jnp.where(out_sum > 0.0, out * total / out_sum, out)
