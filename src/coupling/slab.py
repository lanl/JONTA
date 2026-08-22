"""Homogeneous 0-D/slab particle/Ohm coupling adapter."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from core.state import ParticleState
from deposition.slab import deposit_slab_parallel_current, deposit_slab_weight
from fields.uniform import UniformField

from .driver import ParticleFieldCouplingResult, picard_particle_field_step
from .macrostep import ParticleMoments


class SlabFieldHistory(NamedTuple):
    """History retained for diagnostics and future dynamic slab closures."""

    e_parallel_n: jnp.ndarray
    jkin_n: jnp.ndarray


def picard_slab_step(
    particles_n: ParticleState,
    field_n: UniformField,
    background_np1,
    particle_block,
    particle_dt: float,
    base_key,
    global_step0: int,
    a_minor_cm: float,
    dt_coupling: float,
    *,
    eta: float,
    j_total: float,
    current_scale: float = 1.0,
    a_runaway: float = 0.0,
    te_ev=None,
    max_iterations: int = 4,
    tolerance: float = 1.0e-6,
    relaxation: float = 1.0,
    time_n: float = 0.0,
    require_partitioned: bool = False,
) -> ParticleFieldCouplingResult:
    """Solve scalar algebraic Ohm feedback for homogeneous slab fields.

    The slab closure is ``E_parallel = eta * (j_total - j_kin)``. It uses
    same particle/Picard machinery as circular RAMc while avoiding radial
    grids, safety-factor profiles, and circular geometry factors.
    """

    history = SlabFieldHistory(field_n.e_parallel, jnp.asarray(0.0))

    def make_particle_field(field_state, guess):
        return UniformField(guess, field_state.b)

    def deposit(pool):
        return ParticleMoments(
            deposit_slab_parallel_current(
                pool,
                a_minor_cm,
                current_scale=current_scale,
                a_runaway=a_runaway,
                te_ev=te_ev,
            ),
            deposit_slab_weight(pool),
        )

    def solve_field(field_state, field_history, moments, background, dt):
        del field_state, field_history, background, dt
        return eta * (j_total - moments.parallel_current)

    def finalize_field(field_state, guess, moments, background, dt):
        del moments, background, dt
        return UniformField(guess, field_state.b)

    def update_history(field_state, field_history, field, moments, background):
        del field_state, field_history, background
        return SlabFieldHistory(field.e_parallel, moments.parallel_current)

    return picard_particle_field_step(
        particles_n,
        field_n,
        background_np1,
        history,
        particle_block,
        particle_dt,
        base_key,
        global_step0,
        dt_coupling,
        initial_guess=field_n.e_parallel,
        make_particle_field=make_particle_field,
        solve_field=solve_field,
        finalize_field=finalize_field,
        update_history=update_history,
        moment_depositor=deposit,
        time_n=time_n,
        max_iterations=max_iterations,
        tolerance=tolerance,
        relaxation=relaxation,
        require_partitioned=require_partitioned,
    )
