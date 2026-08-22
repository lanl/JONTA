"""Minimal 0-D runaway-electron example."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from collisions.moller import apply_moller_gain_loss
from collisions.small_angle import small_angle_step
from core.config import MollerConfig, SmallAngleConfig
from core.initialization import uniform_markers
from core.state import BackgroundProfiles
from diagnostics import total_weight, weighted_mean_gamma
from fields.uniform import UniformField
from integrators import rk4_step
from orbits.zero_d import zero_d_rhs
from simulation import build_particle_block


N = 20_000
DT = 2.0e-4
NSTEPS = 100

key = jax.random.key(7)
particles = uniform_markers(
    key,
    N,
    gamma_range=(1.2, 3.0),
    xi_range=(-1.0, -0.2),
    r_range=(0.0, 0.0),
    total_weight=1.0,
)

r = jnp.array([0.0, 1.0])
background = BackgroundProfiles(
    r,
    jnp.full_like(r, 1.0e14),
    jnp.full_like(r, 100.0),
    jnp.full_like(r, 100.0),
    jnp.ones_like(r),
    jnp.ones_like(r),
)
field = UniformField(e_parallel=4.0, b=1.0)

sa_cfg = SmallAngleConfig(ne0_cm3=1.0e14, coulog0=15.0)
la_cfg = MollerConfig(
    ne0_cm3=1.0e14,
    coulog0=15.0,
    gamma_min=1.0 + 50.0e3 / 510998.95,
)

orbit = lambda kin, t, f: zero_d_rhs(kin, t, f, alpha_syn=0.02)
small = lambda p, bg, dt, rng, step: small_angle_step(p, bg, dt, rng, step, sa_cfg)
large = lambda p, bg, dt, rng, step: apply_moller_gain_loss(p, bg, dt, rng, step, la_cfg)

block = build_particle_block(
    orbit,
    rk4_step,
    n_steps=NSTEPS,
    small_angle_operator=small,
    large_angle_operator=large,
    large_angle_every=20,
)

particles, max_q = block(particles, field, background, 0.0, DT, key, 0)
print("total represented electrons:", float(total_weight(particles)))
print("weighted mean gamma:", float(weighted_mean_gamma(particles)))
print("maximum raw large-angle collision fraction:", float(max_q))
