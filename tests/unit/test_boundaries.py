from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from boundaries.basic import momentum_reservoir_boundary
from core.initialization import particles_from_arrays
from core.math import momentum_from_gamma


def test_momentum_reservoir_boundary_is_fixed_shape_and_jittable():
    p = jnp.asarray([0.1, 0.3, 2.0, 20.2], dtype=jnp.float64)
    gamma = jnp.sqrt(1.0 + p * p)
    xi = jnp.asarray([-0.8, -0.2, 0.4, 0.9], dtype=jnp.float64)
    zero = jnp.zeros_like(p)
    weight = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)
    particles = particles_from_arrays(
        gamma,
        xi,
        zero,
        zero,
        zero,
        weight,
        pid=jnp.arange(p.size, dtype=jnp.int64),
    )

    apply_boundary = jax.jit(lambda state: momentum_reservoir_boundary(state, 0.3, 20.0))
    out = apply_boundary(particles)
    p_out = momentum_from_gamma(out.kin.gamma)

    np.testing.assert_allclose(np.asarray(p_out), [0.3, 0.3, 2.0, 19.8], atol=2e-13)
    np.testing.assert_allclose(np.asarray(out.kin.xi), np.asarray(xi))
    np.testing.assert_allclose(np.asarray(out.weight), np.asarray(weight))
    np.testing.assert_array_equal(np.asarray(out.alive), np.asarray(particles.alive))
    np.testing.assert_array_equal(np.asarray(out.pid), np.asarray(particles.pid))
