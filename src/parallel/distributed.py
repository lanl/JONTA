"""Process-level JAX coordination helpers."""

from __future__ import annotations

import os

import jax
import numpy as np
from jax.experimental import multihost_utils


def initialize_from_environment():
    """Initialize JAX distributed runtime from explicit or Slurm settings."""

    coordinator = os.environ.get("JAX_COORDINATOR_ADDRESS")
    if not coordinator:
        jax.distributed.initialize(cluster_detection_method="slurm")
        return

    process_count = int(
        os.environ.get("JAX_NUM_PROCESSES", os.environ.get("SLURM_NTASKS", "1"))
    )
    process_id = int(
        os.environ.get("JAX_PROCESS_ID", os.environ.get("SLURM_PROCID", "0"))
    )
    local_ids = os.environ.get("JAX_LOCAL_DEVICE_IDS")
    local_device_ids = (
        [int(value) for value in local_ids.split(",") if value]
        if local_ids
        else None
    )
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=process_count,
        process_id=process_id,
        local_device_ids=local_device_ids,
    )


def global_barrier(distributed: bool, name: str):
    """Synchronize processes when distributed execution is enabled."""

    if distributed:
        multihost_utils.sync_global_devices(name)


def global_values(value, distributed: bool):
    """Gather one scalar from every process, or return local scalar."""

    local = np.asarray([value])
    if not distributed:
        return local
    return np.asarray(multihost_utils.process_allgather(local)).reshape(-1)
