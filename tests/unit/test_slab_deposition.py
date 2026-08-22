"""Uniform 0-D/slab moment deposition tests."""

import jax.numpy as jnp
import numpy as np
import pytest

from core.initialization import particles_from_arrays
from deposition.slab import deposit_slab_parallel_current, deposit_slab_weight


def test_slab_current_preserves_electron_sign_and_alive_mask():
    particles = particles_from_arrays(
        [2.0, 3.0, 4.0],
        [-0.8, 0.5, -0.9],
        [0.0] * 3,
        [0.0] * 3,
        [0.0] * 3,
        [2.0, 3.0, 5.0],
        alive=[True, False, True],
    )
    current = deposit_slab_parallel_current(particles, 100.0)
    expected = deposit_slab_parallel_current(
        particles._replace(alive=jnp.array([True, False, False])), 100.0
    ) + deposit_slab_parallel_current(
        particles._replace(alive=jnp.array([False, False, True])), 100.0
    )
    np.testing.assert_allclose(current, expected)
    assert float(current) > 0.0
    assert float(deposit_slab_weight(particles)) == 7.0


def test_slab_runaway_threshold_requires_temperature():
    particles = particles_from_arrays([2.0], [-0.8], [0.0], [0.0], [0.0], [1.0])
    with pytest.raises(ValueError, match="te_ev required"):
        deposit_slab_parallel_current(particles, 100.0, a_runaway=1.0)
    assert float(
        deposit_slab_parallel_current(particles, 100.0, a_runaway=1.0, te_ev=1.0e6)
    ) == 0.0
