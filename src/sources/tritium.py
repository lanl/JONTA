"""Tritium beta-decay source from Ekmark et al. (JPP 2024)."""

from __future__ import annotations

import jax.numpy as jnp

from core.constants import (
    ALPHA_FS,
    ME_C2_EV,
    PI,
    TRITIUM_BETA_ENDPOINT_EV,
    TRITIUM_HALF_LIFE_S,
)
from core.precision import real_dtype

C_T = 31800.0


def tritium_source_density_p(p, n_tritium_m3):
    """Isotropic source density S_T(p) per normalized momentum-space volume.

    ``p`` is normalized to m_e c. The normalization follows Ekmark et al. so
    that integral d^3p S_T = ln(2) n_T / tau_T.
    """

    p = jnp.asarray(p, dtype=real_dtype())
    gamma = jnp.sqrt(1.0 + p * p)
    beta = p / gamma
    gamma_max = 1.0 + TRITIUM_BETA_ENDPOINT_EV / ME_C2_EV
    denom = 1.0 - jnp.exp(-4.0 * PI * ALPHA_FS / jnp.maximum(beta, 1.0e-30))
    shape = (
        (1.0 / jnp.maximum(p * p, 1.0e-30))
        * p
        * gamma
        * (gamma_max - gamma) ** 2
        / denom
    )
    rate = C_T * jnp.log(2.0) / (4.0 * PI) * n_tritium_m3 / TRITIUM_HALF_LIFE_S
    return jnp.where((gamma <= gamma_max) & (p > 0.0), rate * shape, 0.0)


def tritium_total_rate(n_tritium_m3):
    return jnp.log(2.0) * n_tritium_m3 / TRITIUM_HALF_LIFE_S


def sample_tritium_momentum(key, n_samples: int, grid_size: int = 2048):
    """Sample p from the normalized beta spectrum by tabulated inverse CDF."""

    gamma_max = 1.0 + TRITIUM_BETA_ENDPOINT_EV / ME_C2_EV
    pmax = jnp.sqrt(gamma_max * gamma_max - 1.0)
    p = jnp.linspace(1.0e-10, pmax, grid_size, dtype=real_dtype())
    # Radial probability is proportional to 4*pi*p^2*S_T; n_T cancels.
    radial_pdf = 4.0 * PI * p * p * tritium_source_density_p(p, 1.0)
    dp = p[1:] - p[:-1]
    trapezoids = 0.5 * (radial_pdf[1:] + radial_pdf[:-1]) * dp
    cdf = jnp.concatenate([jnp.zeros((1,), dtype=real_dtype()), jnp.cumsum(trapezoids)])
    cdf = cdf / cdf[-1]
    u = jax_random_uniform(key, (n_samples,))
    return jnp.interp(u, cdf, p)


def jax_random_uniform(key, shape):
    # Tiny wrapper avoids importing jax at module import in documentation tools.
    import jax

    return jax.random.uniform(key, shape=shape, dtype=real_dtype())
