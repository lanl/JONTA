"""Fixed-N random thinning/resampling utilities."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.precision import index_dtype, real_dtype
from core.rng import derive_particle_ids
from core.state import KinematicState, ParticleState


def effective_sample_size(weight):
    w = jnp.maximum(weight, 0.0)
    total = jnp.sum(w)
    return jnp.where(total > 0.0, total * total / jnp.sum(w * w), 0.0)


def _resampled_state(candidates, idx, total, n_out, global_step):
    kin = KinematicState(
        candidates.kin.gamma[idx],
        candidates.kin.xi[idx],
        candidates.kin.x[idx],
        candidates.kin.y[idx],
        candidates.kin.phi[idx],
    )
    weight = jnp.full((n_out,), total / n_out, dtype=real_dtype())
    alive = jnp.ones((n_out,), dtype=jnp.bool_)
    pid = derive_particle_ids(
        candidates.pid[idx],
        global_step,
        jnp.arange(n_out, dtype=index_dtype()),
    )
    return ParticleState(kin, weight, alive, pid)


def multinomial_resample(
    candidates: ParticleState, key, n_out: int, *, global_step: int = 0
) -> ParticleState:
    """Randomly thin a weighted candidate ensemble to exactly ``n_out`` slots.

    Selection is multinomial with probabilities proportional to marker
    weight. Selected markers receive equal weight ``sum(w)/n_out``. Hence the
    total represented particle number is conserved exactly by resampling,
    while other moments are unbiased Monte Carlo estimators.
    """

    w = jnp.where(candidates.alive, jnp.maximum(candidates.weight, 0.0), 0.0)
    total = jnp.sum(w)

    def _nonempty(_):
        logits = jnp.where(w > 0.0, jnp.log(w), -jnp.inf)
        idx = jax.random.categorical(key, logits, shape=(n_out,))
        return _resampled_state(candidates, idx, total, n_out, global_step)

    def _empty(_):
        z = jnp.zeros((n_out,), dtype=real_dtype())
        kin = KinematicState(jnp.ones_like(z), z, z, z, z)
        return ParticleState(kin, z, jnp.zeros((n_out,), dtype=jnp.bool_), jnp.arange(n_out, dtype=index_dtype()))

    return jax.lax.cond(total > 0.0, _nonempty, _empty, operand=None)


def stratified_resample(
    candidates: ParticleState, key, n_out: int, *, global_step: int = 0
) -> ParticleState:
    """Low-variance random thinning/resampling to exactly ``n_out`` slots.

    The unit interval is divided into ``n_out`` equal strata and one uniform
    variate is drawn in each stratum.  Compared with multinomial resampling,
    this preserves the same weighted candidate distribution while greatly
    reducing Monte-Carlo variance.  Selected markers receive equal weight
    ``sum(w)/n_out``, so represented particle number is exact.
    """

    w = jnp.where(candidates.alive, jnp.maximum(candidates.weight, 0.0), 0.0)
    total = jnp.sum(w)

    def _nonempty(_):
        probs = w / total
        cdf = jnp.cumsum(probs)
        # Ensure roundoff cannot leave the final stratum outside the CDF.
        cdf = cdf.at[-1].set(1.0)
        u = jax.random.uniform(key, (n_out,), dtype=real_dtype())
        positions = (jnp.arange(n_out, dtype=real_dtype()) + u) / n_out
        idx = jnp.searchsorted(cdf, positions, side="right")
        idx = jnp.minimum(idx, w.shape[0] - 1)
        return _resampled_state(candidates, idx, total, n_out, global_step)

    def _empty(_):
        z = jnp.zeros((n_out,), dtype=real_dtype())
        kin = KinematicState(jnp.ones_like(z), z, z, z, z)
        return ParticleState(
            kin,
            z,
            jnp.zeros((n_out,), dtype=jnp.bool_),
            jnp.arange(n_out, dtype=index_dtype()),
        )

    return jax.lax.cond(total > 0.0, _nonempty, _empty, operand=None)
