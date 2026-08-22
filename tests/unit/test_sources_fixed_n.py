import jax
import jax.numpy as jnp

from core.initialization import particles_from_arrays
from core.state import KinematicState
from sources.base import append_source_candidates, inject_and_thin


def test_source_candidates_add_exact_weight():
    p = particles_from_arrays(
        gamma=jnp.array([2.0, 3.0]),
        xi=jnp.array([0.0, 0.0]),
        x=jnp.zeros(2),
        y=jnp.zeros(2),
        phi=jnp.zeros(2),
        weight=jnp.array([4.0, 6.0]),
    )
    source_kin = KinematicState(
        jnp.array([1.1, 1.2, 1.3]),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
        jnp.zeros(3),
    )
    c = append_source_candidates(p, source_kin, 5.0)
    assert c.weight.shape[0] == 5
    assert jnp.allclose(jnp.sum(c.weight), 15.0)


def test_source_injection_returns_fixed_n_and_preserves_total_weight():
    p = particles_from_arrays(
        gamma=jnp.array([2.0, 3.0]),
        xi=jnp.array([0.0, 0.0]),
        x=jnp.zeros(2),
        y=jnp.zeros(2),
        phi=jnp.zeros(2),
        weight=jnp.array([4.0, 6.0]),
    )
    source_kin = KinematicState(
        jnp.array([1.1, 1.2]),
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.zeros(2),
        jnp.zeros(2),
    )
    out = inject_and_thin(p, source_kin, 5.0, jax.random.key(1))
    assert out.weight.shape == p.weight.shape
    assert jnp.allclose(jnp.sum(out.weight), 15.0)
