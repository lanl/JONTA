"""Reference 1-D particle/Ohm coupling for the circular RAMc geometry."""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from core.config import OrbitNormalization
from core.state import BackgroundProfiles, CircularFieldProfiles, FieldHistory, ParticleState
from deposition.radial import deposit_ramc_parallel_current
from .electric_field import ElectricFieldBoundary, bdf2_electric_field_step
from .safety_factor import update_circular_q


class CouplingResult(NamedTuple):
    particles: ParticleState
    field_profiles: CircularFieldProfiles
    history: FieldHistory
    iterations: int
    residual: float
    max_large_angle_fraction: float


def picard_ramc1d_step(
    particles_n: ParticleState,
    field_profiles_n: CircularFieldProfiles,
    background_np1: BackgroundProfiles,
    history: FieldHistory,
    particle_block,
    particle_dt: float,
    n_particle_steps: int,
    base_key,
    global_step0: int,
    norm: OrbitNormalization,
    a_minor_cm: float,
    dt_coupling: float,
    time_n: float = 0.0,
    a_runaway: float = 0.0,
    max_iterations: int = 4,
    tolerance: float = 1.0e-6,
    relaxation: float = 1.0,
    boundary: ElectricFieldBoundary = ElectricFieldBoundary(),
    evolve_q: bool = False,
    j_bootstrap=None,
):
    """Picard coupling around a BDF2 radial electric-field solve.

    Each Picard iteration re-advances the particles from the same n-level
    ensemble using the current E1 guess, deposits j_kin^(n+1), and solves the
    BDF2 Ohm/field equation. This is intentionally simple and robust; more
    sophisticated JFNK/Newton coupling can replace it behind the same
    interface later.
    """

    e_guess = history.e1_n
    particles_guess = particles_n
    max_q = 0.0
    residual = float("inf")

    for k in range(max_iterations):
        profiles_guess = CircularFieldProfiles(
            field_profiles_n.r, e_guess, field_profiles_n.q
        )
        particles_guess, max_q_arr = particle_block(
            particles_n,
            profiles_guess,
            background_np1,
            time_n,
            particle_dt,
            base_key,
            global_step0,
        )
        max_q = max(max_q, float(max_q_arr))
        j_np1 = deposit_ramc_parallel_current(
            particles_guess,
            field_profiles_n.r,
            field_profiles_n.q,
            background_np1,
            norm.epsilon,
            a_minor_cm,
            a_runaway=a_runaway,
        )
        e_solved = bdf2_electric_field_step(
            field_profiles_n.r,
            history.e1_n,
            history.e1_nm1,
            j_np1,
            history.jkin_n,
            history.jkin_nm1,
            background_np1.eta_bar,
            history.eta_n,
            history.eta_nm1,
            dt_coupling,
            epsilon=norm.epsilon,
            boundary=boundary,
            ramc_geometry=True,
        )
        e_next = (1.0 - relaxation) * e_guess + relaxation * e_solved
        denom = max(float(jnp.linalg.norm(e_next)), 1.0e-30)
        residual = float(jnp.linalg.norm(e_next - e_guess)) / denom
        e_guess = e_next
        if residual < tolerance:
            break

    j_final = deposit_ramc_parallel_current(
        particles_guess,
        field_profiles_n.r,
        field_profiles_n.q,
        background_np1,
        norm.epsilon,
        a_minor_cm,
        a_runaway=a_runaway,
    )
    q_final = field_profiles_n.q
    if evolve_q:
        q_final = update_circular_q(
            field_profiles_n.r,
            e_guess,
            background_np1.eta_bar,
            j_final,
            norm.epsilon,
            norm.a_omega_ce_over_c,
            j_bootstrap=j_bootstrap,
        )
    final_profiles = CircularFieldProfiles(
        field_profiles_n.r, e_guess, q_final
    )
    new_history = FieldHistory(
        e1_n=e_guess,
        e1_nm1=history.e1_n,
        jkin_n=j_final,
        jkin_nm1=history.jkin_n,
        eta_n=background_np1.eta_bar,
        eta_nm1=history.eta_n,
    )
    return CouplingResult(
        particles_guess,
        final_profiles,
        new_history,
        k + 1,
        residual,
        max_q,
    )
