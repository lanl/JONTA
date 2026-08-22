"""Reference 1-D particle/Ohm coupling for the circular RAMc geometry."""

from __future__ import annotations

from typing import NamedTuple

from core.config import OrbitNormalization
from core.state import BackgroundProfiles, CircularFieldProfiles, FieldHistory, ParticleState
from deposition.radial import deposit_ramc_parallel_current, deposit_weight_density

from .driver import picard_particle_field_step
from .electric_field import ElectricFieldBoundary, bdf2_electric_field_step
from .macrostep import ParticleMoments
from .safety_factor import update_circular_q


class CouplingResult(NamedTuple):
    particles: ParticleState
    field_profiles: CircularFieldProfiles
    history: FieldHistory
    iterations: int
    residual: float
    max_large_angle_fraction: float
    converged: bool


def deposit_circular_particle_moments(
    particles: ParticleState,
    grid,
    q_profile,
    background: BackgroundProfiles,
    norm: OrbitNormalization,
    a_minor_cm: float,
    a_runaway: float = 0.0,
) -> ParticleMoments:
    """Deposit RAMc circular radial moments for one flat particle pool."""

    return ParticleMoments(
        deposit_ramc_parallel_current(
            particles,
            grid,
            q_profile,
            background,
            norm.epsilon,
            a_minor_cm,
            a_runaway=a_runaway,
        ),
        deposit_weight_density(particles, grid),
    )


def picard_ramc1d_step(
    particles_n: ParticleState,
    field_profiles_n: CircularFieldProfiles,
    background_np1: BackgroundProfiles,
    history: FieldHistory,
    particle_block,
    particle_dt: float,
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
    require_partitioned: bool = False,
):
    """Picard coupling around a BDF2 radial electric-field solve.

    Each Picard iteration re-advances the particles from the same n-level
    ensemble using the current E1 guess, deposits j_kin^(n+1), and solves the
    BDF2 Ohm/field equation. This is intentionally simple and robust; more
    sophisticated JFNK/Newton coupling can replace it behind the same
    interface later.
    """

    def make_particle_field(field_n, e_guess):
        return CircularFieldProfiles(field_n.r, e_guess, field_n.q)

    def deposit_moments(pool):
        return deposit_circular_particle_moments(
            pool,
            field_profiles_n.r,
            field_profiles_n.q,
            background_np1,
            norm,
            a_minor_cm,
            a_runaway=a_runaway,
        )

    def solve_field(field_n, field_history, moments, background, dt):
        return bdf2_electric_field_step(
            field_n.r,
            field_history.e1_n,
            field_history.e1_nm1,
            moments.parallel_current,
            field_history.jkin_n,
            field_history.jkin_nm1,
            background.eta_bar,
            field_history.eta_n,
            field_history.eta_nm1,
            dt,
            epsilon=norm.epsilon,
            boundary=boundary,
            ramc_geometry=True,
        )

    def finalize_field(field_n, e_guess, moments, background, dt):
        del dt
        q_final = field_n.q
        if evolve_q:
            q_final = update_circular_q(
                field_n.r,
                e_guess,
                background.eta_bar,
                moments.parallel_current,
                norm.epsilon,
                norm.a_omega_ce_over_c,
                j_bootstrap=j_bootstrap,
            )
        return CircularFieldProfiles(field_n.r, e_guess, q_final)

    def update_history(field_n, field_history, field_final, moments, background):
        del field_n
        return FieldHistory(
            e1_n=field_final.e1,
            e1_nm1=field_history.e1_n,
            jkin_n=moments.parallel_current,
            jkin_nm1=field_history.jkin_n,
            eta_n=background.eta_bar,
            eta_nm1=field_history.eta_n,
        )

    generic_result = picard_particle_field_step(
        particles_n,
        field_profiles_n,
        background_np1,
        history,
        particle_block,
        particle_dt,
        base_key,
        global_step0,
        dt_coupling,
        initial_guess=history.e1_n,
        make_particle_field=make_particle_field,
        solve_field=solve_field,
        finalize_field=finalize_field,
        update_history=update_history,
        moment_depositor=deposit_moments,
        time_n=time_n,
        max_iterations=max_iterations,
        tolerance=tolerance,
        relaxation=relaxation,
        require_partitioned=require_partitioned,
    )
    return CouplingResult(
        generic_result.particles,
        generic_result.field_state,
        generic_result.history,
        generic_result.iterations,
        generic_result.residual,
        generic_result.max_large_angle_fraction,
        generic_result.converged,
    )
