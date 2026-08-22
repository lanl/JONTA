"""Particle-first JAX sharding helpers."""

from __future__ import annotations

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from core.state import KinematicState, ParticleState


def particle_mesh(devices=None):
    devices = jax.devices() if devices is None else devices
    return Mesh(np.asarray(devices), ("particle",))


def shard_particles(particles: ParticleState, mesh: Mesh):
    """Shard every particle array on its leading axis across ``mesh``."""

    sharding = NamedSharding(mesh, P("particle"))
    kin = KinematicState(*[jax.device_put(x, sharding) for x in particles.kin])
    return ParticleState(
        kin,
        jax.device_put(particles.weight, sharding),
        jax.device_put(particles.alive, sharding),
        jax.device_put(particles.pid, sharding),
    )


def partition_particles(particles: ParticleState, n_devices: int):
    """Reshape a particle ensemble into equal per-device batches.

    This explicit batch form is useful with ``jax.pmap`` and CPU device
    emulation.  It does not perform communication or population control; the
    caller remains responsible for global reductions/resampling.
    """

    n_devices = int(n_devices)
    if n_devices < 1:
        raise ValueError("n_devices must be positive")
    n_markers = particles.weight.shape[0]
    if n_markers % n_devices:
        raise ValueError(
            f"marker count {n_markers} is not divisible by n_devices={n_devices}"
        )

    def partition(value):
        return value.reshape((n_devices, n_markers // n_devices) + value.shape[1:])

    kin = KinematicState(*[partition(value) for value in particles.kin])
    return ParticleState(
        kin,
        partition(particles.weight),
        partition(particles.alive),
        partition(particles.pid),
    )


def merge_particle_partitions(particles: ParticleState):
    """Flatten the leading device and local-marker axes of a partition."""

    if particles.weight.ndim < 2:
        raise ValueError("expected a partitioned particle ensemble")

    def merge(value):
        return value.reshape((-1,) + value.shape[2:])

    kin = KinematicState(*[merge(value) for value in particles.kin])
    return ParticleState(
        kin,
        merge(particles.weight),
        merge(particles.alive),
        merge(particles.pid),
    )


def replicated(value, mesh: Mesh):
    return jax.device_put(value, NamedSharding(mesh, P()))
