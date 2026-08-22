"""Fixed-shape, variable-active marker population machinery.

Particle arrays keep device-local capacity.  Operators may create temporary
candidate arrays larger than that capacity; candidates are compacted without
resampling when they fit, and stratified-thinned only on local overflow.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from core.precision import index_dtype
from core.state import KinematicState, ParticleState
from resampling.multinomial import effective_sample_size, stratified_resample


class PopulationDiagnostics(NamedTuple):
    """Population-control quantities returned by one local update."""

    candidate_active: jnp.ndarray
    output_active: jnp.ndarray
    candidate_weight: jnp.ndarray
    output_weight: jnp.ndarray
    candidate_ess: jnp.ndarray
    overflowed: jnp.ndarray
    thinned: jnp.ndarray


def _safe_state(kin, weight, alive, pid):
    """Make dead slots finite and inert while preserving live values."""

    alive = alive & (weight > 0.0)
    z = jnp.zeros_like(weight)
    kin = KinematicState(
        jnp.where(alive, kin.gamma, jnp.ones_like(kin.gamma)),
        jnp.where(alive, kin.xi, z),
        jnp.where(alive, kin.x, z),
        jnp.where(alive, kin.y, z),
        jnp.where(alive, kin.phi, z),
    )
    weight = jnp.where(alive, jnp.maximum(weight, 0.0), z)
    pid = jnp.where(alive, pid, jnp.asarray(-1, dtype=index_dtype()))
    return ParticleState(kin, weight, alive, pid)


def compact_particles(candidates: ParticleState, capacity: int) -> ParticleState:
    """Compact live candidates into fixed ``capacity`` slots.

    Individual candidate weights and IDs remain unchanged.  This path performs
    no statistical resampling; unused slots become dead markers.
    """

    capacity = int(capacity)
    if capacity < 1:
        raise ValueError("capacity must be positive")
    n = int(candidates.weight.shape[0])
    live = candidates.alive & (candidates.weight > 0.0)
    order = jnp.argsort(~live, stable=True)
    if n >= capacity:
        idx = order[:capacity]
        kin = KinematicState(*(value[idx] for value in candidates.kin))
        out = ParticleState(kin, candidates.weight[idx], live[idx], candidates.pid[idx])
        return _safe_state(out.kin, out.weight, out.alive, out.pid)

    # Generic padding path, useful for source-only updates with few candidates.
    pad = capacity - n
    def padded(value, fill):
        return jnp.concatenate([value, jnp.full((pad,), fill, dtype=value.dtype)])

    kin = KinematicState(
        padded(candidates.kin.gamma, 1.0),
        padded(candidates.kin.xi, 0.0),
        padded(candidates.kin.x, 0.0),
        padded(candidates.kin.y, 0.0),
        padded(candidates.kin.phi, 0.0),
    )
    weight = padded(candidates.weight, 0.0)
    alive = padded(live, False)
    pid = padded(candidates.pid, -1)
    order = jnp.concatenate([order, jnp.arange(n, capacity, dtype=index_dtype())])
    kin = KinematicState(*(value[order] for value in kin))
    return _safe_state(kin, weight[order], alive[order], pid[order])


def capacity_control(
    candidates: ParticleState,
    capacity: int,
    key,
    *,
    global_step: int = 0,
):
    """Apply local capacity policy and return state plus diagnostics.

    If live candidates fit, compact directly.  If not, stratified thinning is
    the only population-control operation.  Total represented weight is exact
    in both cases.
    """

    capacity = int(capacity)
    live = candidates.alive & (candidates.weight > 0.0)
    weights = jnp.where(live, jnp.maximum(candidates.weight, 0.0), 0.0)
    n_live = jnp.sum(live, dtype=index_dtype())
    total = jnp.sum(weights)
    ess = effective_sample_size(weights)
    overflowed = n_live > capacity

    def no_overflow(_):
        return compact_particles(candidates, capacity)

    def overflow(_):
        return stratified_resample(
            candidates,
            key,
            capacity,
            global_step=global_step,
        )

    out = jax_cond(overflowed, overflow, no_overflow, operand=None)
    diagnostics = PopulationDiagnostics(
        n_live,
        jnp.sum(out.alive, dtype=index_dtype()),
        total,
        jnp.sum(out.weight),
        ess,
        overflowed,
        overflowed,
    )
    return out, diagnostics


def jax_cond(pred, true_fun, false_fun, operand):
    """Local import wrapper keeps module import order cycle-free."""

    import jax

    return jax.lax.cond(pred, true_fun, false_fun, operand)
