"""Geometry-neutral particle/field Picard iteration.

Field representation, moment deposition, and plasma solve stay outside this
module. Backends inject those operations through small callbacks. This keeps
slab, circular, and future 2-D configurations on one nonlinear data path.
"""

from __future__ import annotations

from typing import Any, Callable, NamedTuple

import jax.numpy as jnp

from core.state import ParticleState

from .macrostep import ParticleMacrostepResult, advance_particle_macrostep


class ParticleFieldCouplingResult(NamedTuple):
    """Result shared by all geometry backends."""

    particles: ParticleState
    field_state: Any
    history: Any
    iterations: int
    residual: float
    max_large_angle_fraction: float
    converged: bool


def relative_l2_residual(new, old) -> float:
    """Relative L2 change for array/scalar field guesses."""

    denominator = max(float(jnp.linalg.norm(new)), 1.0e-30)
    return float(jnp.linalg.norm(new - old)) / denominator


def _relax(old, new, relaxation: float):
    return (1.0 - relaxation) * old + relaxation * new


def picard_particle_field_step(
    particles_n: ParticleState,
    field_state_n,
    background_np1,
    history,
    particle_block,
    particle_dt: float,
    base_key,
    global_step0: int,
    dt_coupling: float,
    *,
    initial_guess,
    make_particle_field: Callable,
    solve_field: Callable,
    finalize_field: Callable,
    update_history: Callable,
    moment_depositor: Callable | None = None,
    time_n: float = 0.0,
    max_iterations: int = 4,
    tolerance: float = 1.0e-6,
    relaxation: float = 1.0,
    residual_fn: Callable = relative_l2_residual,
    require_partitioned: bool = False,
) -> ParticleFieldCouplingResult:
    """Advance one nonlinear particle/plasma macrostep.

    Backend callbacks own field representation and plasma equations:

    ``make_particle_field(field_state_n, guess)``
        Build field state consumed by particle orbit kernel.
    ``solve_field(field_state_n, history, moments, background, dt)``
        Return new field guess from deposited particle moments.
    ``finalize_field(field_state_n, guess, moments, background, dt)``
        Apply backend updates such as safety factor or algebraic closure.
    ``update_history(field_state_n, history, field, moments, background)``
        Construct explicit next-level history.

    Particle pushes always restart from ``particles_n``. Final particles and
    moments are re-evaluated under accepted final field guess.
    """

    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if dt_coupling <= 0.0 or particle_dt <= 0.0:
        raise ValueError("coupling and particle timesteps must be positive")
    if not 0.0 < relaxation <= 1.0:
        raise ValueError("relaxation must lie in (0, 1]")
    if moment_depositor is None:
        raise ValueError("moment_depositor is required for geometry-neutral coupling")
    if require_partitioned and not getattr(particle_block, "particle_output_partitioned", False):
        raise ValueError(
            "particle_block must preserve partitions for local moment reduction"
        )

    guess = initial_guess
    residual = float("inf")
    max_q = 0.0

    for k in range(max_iterations):
        particle_field = make_particle_field(field_state_n, guess)
        macro: ParticleMacrostepResult = advance_particle_macrostep(
            particle_block,
            particles_n,
            particle_field,
            background_np1,
            time_n,
            particle_dt,
            base_key,
            global_step0,
            moment_depositor=moment_depositor,
        )
        max_q = max(max_q, float(macro.max_large_angle_fraction))
        solved = solve_field(
            field_state_n,
            history,
            macro.moments,
            background_np1,
            dt_coupling,
        )
        next_guess = _relax(guess, solved, relaxation)
        residual = residual_fn(next_guess, guess)
        guess = next_guess
        if residual < tolerance:
            break

    particle_field = make_particle_field(field_state_n, guess)
    final_macro = advance_particle_macrostep(
        particle_block,
        particles_n,
        particle_field,
        background_np1,
        time_n,
        particle_dt,
        base_key,
        global_step0,
        moment_depositor=moment_depositor,
    )
    max_q = max(max_q, float(final_macro.max_large_angle_fraction))
    final_field = finalize_field(
        field_state_n,
        guess,
        final_macro.moments,
        background_np1,
        dt_coupling,
    )
    new_history = update_history(
        field_state_n,
        history,
        final_field,
        final_macro.moments,
        background_np1,
    )
    return ParticleFieldCouplingResult(
        final_macro.particles,
        final_field,
        new_history,
        k + 1,
        residual,
        max_q,
        residual < tolerance,
    )
