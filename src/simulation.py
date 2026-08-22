"""High-level particle-step orchestration.

The simulation driver intentionally contains little physics. It composes an
orbit RHS, integrator, collision operators, and boundary model selected when
the kernel is constructed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from core.config import ExecutionConfig
from core.precision import PrecisionValue, configure_precision, index_dtype, real_dtype
from core.state import ParticleState
from parallel import merge_particle_partitions, partition_particles, resolve_execution


def advance_deterministic(particles: ParticleState, rhs, integrator, t, dt):
    kin_new = integrator(rhs, particles.kin, t, dt)
    return ParticleState(kin_new, particles.weight, particles.alive, particles.pid)


def build_particle_block(
    orbit_rhs,
    integrator,
    n_steps: int,
    small_angle_operator=None,
    large_angle_operator=None,
    large_angle_every: int = 0,
    boundary_operator=None,
    strang_small_angle: bool = True,
    precision: PrecisionValue = None,
    execution: ExecutionConfig | None = None,
    preserve_partitioning: bool = False,
):
    """Build and JIT a fixed-length GPU-resident particle block.

    ``orbit_rhs`` must have signature ``rhs(kin, t, field_state)``.
    Small-angle and large-angle operators receive the background state as an
    explicit argument. ``large_angle_every`` is measured in particle steps.
    ``execution`` selects the serial reference path or particle-sharded
    execution on the requested CPU/GPU device set. Population control is
    local to each particle partition; compact field state remains replicated.
    ``preserve_partitioning`` keeps the leading device axis in the parallel
    return value for coupling-time local moment deposition. The default
    retains the flat state API used by diagnostics and legacy callers.
    """

    if large_angle_operator is not None and large_angle_every <= 0:
        raise ValueError("large_angle_every must be positive when a large-angle operator is supplied")
    configure_precision(precision)
    execution_plan = resolve_execution(execution)
    def block(particles, field_state, background, t0, dt, base_key, global_step0):
        def body(carry, i):
            p, max_q = carry
            step = global_step0 + i
            t = t0 + i * dt

            if small_angle_operator is not None and strang_small_angle:
                p = small_angle_operator(p, background, 0.5 * dt, base_key, 2 * step)

            def rhs(kin, tt):
                return orbit_rhs(kin, tt, field_state)

            p = advance_deterministic(p, rhs, integrator, t, dt)

            if small_angle_operator is not None:
                sa_dt = 0.5 * dt if strang_small_angle else dt
                sa_step = 2 * step + 1 if strang_small_angle else step
                p = small_angle_operator(p, background, sa_dt, base_key, sa_step)

            if boundary_operator is not None:
                p = boundary_operator(p)

            if large_angle_operator is not None:
                do_la = ((step + 1) % large_angle_every) == 0

                def apply_la(pp):
                    return large_angle_operator(
                        pp,
                        background,
                        dt * large_angle_every,
                        base_key,
                        step,
                    )

                def no_la(pp):
                    return pp, jnp.asarray(0.0, dtype=real_dtype())

                p, q = jax.lax.cond(do_la, apply_la, no_la, p)
                max_q = jnp.maximum(max_q, q)

            return (p, max_q), None

        (out, max_q), _ = jax.lax.scan(
            body,
            (particles, jnp.asarray(0.0, dtype=real_dtype())),
            jnp.arange(n_steps, dtype=index_dtype()),
        )
        return out, max_q

    compiled_block = jax.jit(block)
    if execution_plan.config.mode == "serial":
        compiled_block.particle_output_partitioned = False
        return compiled_block

    mapped_block = jax.pmap(
        compiled_block,
        in_axes=(0, None, None, None, None, None, None),
        devices=execution_plan.devices,
    )

    def parallel_block(particles, field_state, background, t0, dt, base_key, global_step0):
        partitioned = partition_particles(particles, execution_plan.n_devices)
        local_particles, local_max_q = mapped_block(
            partitioned,
            field_state,
            background,
            t0,
            dt,
            base_key,
            global_step0,
        )
        if preserve_partitioning:
            return local_particles, local_max_q
        return merge_particle_partitions(local_particles), jnp.max(local_max_q)

    parallel_block.particle_output_partitioned = preserve_partitioning
    return parallel_block
