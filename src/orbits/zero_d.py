"""0-D momentum-space deterministic runaway-electron dynamics."""

from __future__ import annotations

import jax.numpy as jnp

from core.math import momentum_from_gamma
from core.state import KinematicState
from fields.uniform import UniformField

from .radiation import synchrotron_low_beta


def zero_d_rhs(
    kin: KinematicState,
    time: float,
    field: UniformField,
    alpha_syn: float = 0.0,
) -> KinematicState:
    """Uniform-field RAMc limit in (gamma, xi).

    The Lorentz-force contributions are

        dgamma/dt = -(p/gamma) xi E_parallel
        dxi/dt    = -(1-xi^2) E_parallel / p

    using the RAMc electron sign convention. The returned ODE derivative
    combines the Lorentz characteristic with the deterministic synchrotron
    right-hand-side operator for efficient particle integration; physically,
    radiation is not part of the Vlasov transport term. Spatial coordinates
    are unchanged.
    """

    del time
    gamma = kin.gamma
    xi = kin.xi
    p = momentum_from_gamma(gamma)
    p_safe = jnp.maximum(p, 1.0e-14)
    v = p / jnp.maximum(gamma, 1.0)

    dgamma_syn, dxi_syn = synchrotron_low_beta(gamma, xi, alpha_syn, field.b)
    dgamma = -v * xi * field.e_parallel + dgamma_syn
    dxi = -(1.0 - xi * xi) * field.e_parallel / p_safe + dxi_syn

    z = jnp.zeros_like(gamma)
    return KinematicState(dgamma, dxi, z, z, z)
