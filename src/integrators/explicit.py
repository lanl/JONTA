"""Swappable explicit integrators.

The Bogacki--Shampine 5(4) tableau matches PETSc's ``TSRK5BS`` method used
by RAMc.  The fixed-step function exposes both the fifth-order solution and
the embedded fourth-order estimate; the bounded adaptive wrapper is JAX-safe
and can be used inside ``jit``/``vmap``/``scan`` kernels.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


def _axpy(y, a, k):
    return jax.tree.map(lambda yi, ki: yi + a * ki, y, k)


def euler_step(rhs, y, t, dt):
    return _axpy(y, dt, rhs(y, t))


def midpoint_step(rhs, y, t, dt):
    k1 = rhs(y, t)
    ymid = _axpy(y, 0.5 * dt, k1)
    k2 = rhs(ymid, t + 0.5 * dt)
    return _axpy(y, dt, k2)


def rk4_step(rhs, y, t, dt):
    k1 = rhs(y, t)
    k2 = rhs(_axpy(y, 0.5 * dt, k1), t + 0.5 * dt)
    k3 = rhs(_axpy(y, 0.5 * dt, k2), t + 0.5 * dt)
    k4 = rhs(_axpy(y, dt, k3), t + dt)
    return jax.tree.map(
        lambda yi, a, b, c, d: yi + (dt / 6.0) * (a + 2.0 * b + 2.0 * c + d),
        y,
        k1,
        k2,
        k3,
        k4,
    )


class AdaptiveStepResult(NamedTuple):
    """Adaptive interval result and bounded-loop diagnostics."""

    state: object
    accepted_steps: jax.Array
    rejected_steps: jax.Array
    converged: jax.Array
    next_dt: jax.Array


def bogacki_shampine5_step(rhs, y, t, dt):
    """Advance one fixed step with PETSc ``TSRK5BS`` coefficients.

    Returns ``(fifth_order_state, fourth_order_state, error_state)``.  The
    eight-stage tableau has the FSAL property; the final stage is retained in
    the returned error estimate so callers can reuse it if desired.
    """

    k1 = rhs(y, t)
    y2 = _axpy(y, dt * (1.0 / 6.0), k1)
    k2 = rhs(y2, t + dt * (1.0 / 6.0))

    y3 = jax.tree.map(
        lambda yi, a, b: yi + dt * (2.0 / 27.0 * a + 4.0 / 27.0 * b),
        y,
        k1,
        k2,
    )
    k3 = rhs(y3, t + dt * (1.0 / 3.0))

    y4 = jax.tree.map(
        lambda yi, a, b, c: yi
        + dt * (183.0 / 1372.0 * a - 162.0 / 343.0 * b + 1053.0 / 1372.0 * c),
        y,
        k1,
        k2,
        k3,
    )
    k4 = rhs(y4, t + dt * (1.0 / 2.0))

    y5 = jax.tree.map(
        lambda yi, a, b, c, d: yi
        + dt * (68.0 / 297.0 * a - 4.0 / 11.0 * b + 42.0 / 143.0 * c + 1960.0 / 3861.0 * d),
        y,
        k1,
        k2,
        k3,
        k4,
    )
    k5 = rhs(y5, t + dt * (2.0 / 3.0))

    y6 = jax.tree.map(
        lambda yi, a, b, c, d, e: yi
        + dt
        * (
            597.0 / 22528.0 * a
            + 81.0 / 352.0 * b
            + 63099.0 / 585728.0 * c
            + 58653.0 / 366080.0 * d
            + 4617.0 / 20480.0 * e
        ),
        y,
        k1,
        k2,
        k3,
        k4,
        k5,
    )
    k6 = rhs(y6, t + dt)

    y7 = jax.tree.map(
        lambda yi, a, b, c, d, e, f: yi
        + dt
        * (
            174197.0 / 959244.0 * a
            - 30942.0 / 79937.0 * b
            + 8152137.0 / 19744439.0 * c
            + 666106.0 / 1039181.0 * d
            - 29421.0 / 29068.0 * e
            + 482048.0 / 414219.0 * f
        ),
        y,
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
    )
    k7 = rhs(y7, t + dt)

    y8 = jax.tree.map(
        lambda yi, a, _b, c, d, e, f, g, _h: yi
        + dt
        * (
            587.0 / 8064.0 * a
            + 4440339.0 / 15491840.0 * c
            + 24353.0 / 124800.0 * d
            + 387.0 / 44800.0 * e
            + 2152.0 / 5985.0 * f
            + 7267.0 / 94080.0 * g
        ),
        y,
        k1,
        k2,
        k3,
        k4,
        k5,
        k6,
        k7,
        k7,
    )
    k8 = rhs(y8, t + dt)

    fifth = y8
    fourth = jax.tree.map(
        lambda yi, a, c, d, e, f, g, h: yi
        + dt
        * (
            2479.0 / 34992.0 * a
            + 123.0 / 416.0 * c
            + 612941.0 / 3411720.0 * d
            + 43.0 / 1440.0 * e
            + 2272.0 / 6561.0 * f
            + 79937.0 / 1113912.0 * g
            + 3293.0 / 556956.0 * h
        ),
        y,
        k1,
        k3,
        k4,
        k5,
        k6,
        k7,
        k8,
    )
    error = jax.tree.map(lambda high, low: high - low, fifth, fourth)
    return fifth, fourth, error


def _error_norm(error, reference, atol, rtol):
    leaves = jax.tree.leaves(
        jax.tree.map(
            lambda err, ref: jnp.abs(err) / (atol + rtol * jnp.maximum(jnp.abs(ref), 1.0)),
            error,
            reference,
        )
    )
    return jnp.max(jnp.stack([jnp.max(leaf) for leaf in leaves]))


def adaptive_bogacki_shampine5_interval(
    rhs,
    y,
    t,
    interval,
    *,
    atol=1.0e-9,
    rtol=1.0e-9,
    initial_dt=None,
    min_dt=1.0e-14,
    max_steps=10000,
):
    """Advance exactly one output interval with bounded adaptive BS5(4).

    The internal rejection loop is static-bounded for JAX compilation.  A
    failed interval is reported through ``converged=False`` rather than being
    silently replaced by another integrator.
    """

    if isinstance(interval, (int, float)) and interval <= 0.0:
        raise ValueError("interval must be positive")
    if isinstance(atol, (int, float)) and (atol <= 0.0 or rtol < 0.0):
        raise ValueError("atol must be positive and rtol non-negative")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    interval_array = jnp.asarray(interval)
    start_dt = (
        interval_array
        if initial_dt is None
        else jnp.minimum(jnp.asarray(initial_dt), interval_array)
    )
    start_dt = jnp.maximum(start_dt, jnp.asarray(min_dt, dtype=interval_array.dtype))
    target = t + interval

    def condition(carry):
        current, _state, dt_now, accepted, rejected, converged = carry
        return (current < target) & (accepted + rejected < max_steps) & converged

    def body(carry):
        current, state, dt_now, accepted, rejected, converged = carry
        trial_dt = jnp.minimum(dt_now, target - current)
        fifth, _fourth, error = bogacki_shampine5_step(rhs, state, current, trial_dt)
        norm = _error_norm(error, fifth, atol, rtol)
        finite = jnp.all(jnp.isfinite(jnp.asarray(norm)))
        accept = finite & (norm <= 1.0)
        safe_norm = jnp.maximum(norm, 1.0e-16)
        factor = jnp.clip(0.9 * safe_norm ** (-0.2), 0.1, 5.0)
        next_dt = jnp.maximum(min_dt, trial_dt * factor)
        next_state = jax.tree.map(lambda old, new: jnp.where(accept, new, old), state, fifth)
        next_current = jnp.where(accept, current + trial_dt, current)
        next_accepted = accepted + accept.astype(jnp.int32)
        next_rejected = rejected + (~accept).astype(jnp.int32)
        next_converged = converged & (accept | (trial_dt > min_dt))
        return next_current, next_state, next_dt, next_accepted, next_rejected, next_converged

    final_t, final_state, next_dt, accepted, rejected, converged = jax.lax.while_loop(
        condition,
        body,
        (jnp.asarray(t), y, jnp.asarray(start_dt), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32), jnp.asarray(True)),
    )
    converged = converged & (final_t >= target)
    return AdaptiveStepResult(final_state, accepted, rejected, converged, next_dt)
