"""Fixed-capacity, variable-active population invariants."""

import jax
import jax.numpy as jnp

from core.initialization import particles_from_arrays
from population.capacity import capacity_control, compact_particles


def _particles(weights, alive=None, pid=None):
    n = len(weights)
    return particles_from_arrays(
        gamma=jnp.arange(n, dtype=jnp.float64) + 2.0,
        xi=jnp.zeros(n),
        x=jnp.zeros(n),
        y=jnp.zeros(n),
        phi=jnp.zeros(n),
        weight=jnp.asarray(weights, dtype=jnp.float64),
        alive=alive,
        pid=pid,
    )


def test_compaction_reuses_dead_slots_without_resampling():
    particles = _particles(
        [1.0, 0.0, 2.0, 0.0],
        alive=jnp.asarray([True, False, True, False]),
        pid=jnp.asarray([11, 12, 13, 14]),
    )
    out = compact_particles(particles, 4)

    assert jnp.array_equal(out.alive, jnp.asarray([True, True, False, False]))
    assert jnp.array_equal(out.pid[:2], jnp.asarray([11, 13]))
    assert jnp.allclose(out.weight[:2], jnp.asarray([1.0, 2.0]))
    assert jnp.allclose(jnp.sum(out.weight), 3.0)
    assert jnp.all(jnp.isfinite(out.kin.gamma))


def test_capacity_control_thins_only_on_overflow_and_preserves_weight():
    candidates = _particles(
        [1.0, 2.0, 3.0, 4.0],
        pid=jnp.asarray([21, 22, 23, 24]),
    )
    out, diag = capacity_control(candidates, 2, jax.random.key(4), global_step=7)

    assert bool(diag.overflowed)
    assert bool(diag.thinned)
    assert int(diag.output_active) == 2
    assert jnp.allclose(diag.output_weight, diag.candidate_weight)
    assert jnp.unique(out.pid).size == 2


def test_capacity_control_compacts_when_live_candidates_fit():
    candidates = _particles(
        [1.0, 0.0, 2.0, 0.0],
        alive=jnp.asarray([True, False, True, False]),
        pid=jnp.asarray([31, 32, 33, 34]),
    )
    out, diag = capacity_control(candidates, 4, jax.random.key(5), global_step=9)

    assert not bool(diag.overflowed)
    assert not bool(diag.thinned)
    assert jnp.array_equal(out.pid[:2], jnp.asarray([31, 33]))
    assert jnp.allclose(out.weight, jnp.asarray([1.0, 2.0, 0.0, 0.0]))
