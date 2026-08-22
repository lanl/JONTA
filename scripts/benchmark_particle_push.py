"""Small single-device throughput benchmark for the deterministic particle push."""

from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp

from core.initialization import uniform_markers
from core.state import BackgroundProfiles
from fields.uniform import UniformField
from integrators import rk4_step
from orbits import zero_d_rhs
from simulation import build_particle_block


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=1_000_000)
    parser.add_argument("--steps", type=int, default=100)
    args = parser.parse_args()

    key = jax.random.key(0)
    p = uniform_markers(key, args.particles, gamma_range=(2, 5), xi_range=(-1, 1), r_range=(0, 0))
    r = jnp.array([0.0, 1.0])
    bg = BackgroundProfiles(r, jnp.full(2, 1e14), jnp.full(2, 1e3), jnp.full(2, 1e3), jnp.ones(2), jnp.ones(2))
    orbit = lambda kin, t, field: zero_d_rhs(kin, t, field, alpha_syn=0.02)
    block = build_particle_block(orbit, rk4_step, n_steps=args.steps)

    # Compile.
    out, _ = block(p, UniformField(2.5), bg, 0.0, 1e-4, key, 0)
    jax.block_until_ready(out.kin.gamma)

    t0 = time.perf_counter()
    out, _ = block(p, UniformField(2.5), bg, 0.0, 1e-4, key, 0)
    jax.block_until_ready(out.kin.gamma)
    elapsed = time.perf_counter() - t0
    particle_steps = args.particles * args.steps
    print(f"backend: {jax.default_backend()}")
    print(f"elapsed: {elapsed:.6f} s")
    print(f"particle-steps/s: {particle_steps / elapsed:.6e}")


if __name__ == "__main__":
    main()
