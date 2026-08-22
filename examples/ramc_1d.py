"""One circular-RAMc 1-D particle/Ohm coupling step."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from boundaries import radial_absorbing_wall
from collisions.small_angle import small_angle_step
from core.config import OrbitNormalization, SmallAngleConfig
from core.initialization import uniform_markers
from core.state import BackgroundProfiles, CircularFieldProfiles, FieldHistory
from coupling import ElectricFieldBoundary, picard_ramc1d_step
from deposition import deposit_ramc_parallel_current
from integrators import rk4_step
from orbits import ramc_circular_rhs
from plasma import ramc_spitzer_eta_bar
from simulation import build_particle_block


N = 50_000
NR = 96
DT_PARTICLE = 2.0e-7
N_PARTICLE_STEPS = 20
DT_COUPLING = DT_PARTICLE * N_PARTICLE_STEPS

key = jax.random.key(3)
r = jnp.linspace(0.0, 1.0, NR)
q = 2.1 + 2.0 * r * r
e1 = jnp.full_like(r, 2.5).at[-1].set(0.0)

norm = OrbitNormalization(
    epsilon=1.0 / 3.0,
    c_tau_over_a=5.0e5,
    a_omega_ce_over_c=6.22e3,
    alpha_syn=0.1,
)
field = CircularFieldProfiles(r, e1, q)

ne = jnp.full_like(r, 1.0e14)
te = jnp.full_like(r, 1.0e3)
ti = te
zeff = jnp.ones_like(r)
eta = ramc_spitzer_eta_bar(te, zeff, 1.0e14, 200.0, 15.0, ne)
background = BackgroundProfiles(r, ne, te, ti, zeff, eta)

particles = uniform_markers(
    key,
    N,
    gamma_range=(1.2, 2.0),
    xi_range=(-1.0, 1.0),
    r_range=(0.05, 0.75),
    total_weight=1.0e16,
)

j0 = deposit_ramc_parallel_current(
    particles, r, q, background, norm.epsilon, 200.0
)
history = FieldHistory(e1, e1, j0, j0, eta, eta)

sa_cfg = SmallAngleConfig(1.0e14, 15.0)
orbit = lambda kin, t, f: ramc_circular_rhs(kin, t, f, norm)
small = lambda p, bg, dt, rng, step: small_angle_step(p, bg, dt, rng, step, sa_cfg)
boundary = lambda p: radial_absorbing_wall(p, 1.0)
particle_block = build_particle_block(
    orbit,
    rk4_step,
    n_steps=N_PARTICLE_STEPS,
    small_angle_operator=small,
    boundary_operator=boundary,
)

result = picard_ramc1d_step(
    particles,
    field,
    background,
    history,
    particle_block,
    DT_PARTICLE,
    N_PARTICLE_STEPS,
    key,
    0,
    norm,
    a_minor_cm=200.0,
    dt_coupling=DT_COUPLING,
    max_iterations=3,
    tolerance=1.0e-5,
    boundary=ElectricFieldBoundary("conducting", 1.0),
)

print("Picard iterations:", result.iterations)
print("relative coupling residual:", result.residual)
print("E1(0):", float(result.field_profiles.e1[0]))
