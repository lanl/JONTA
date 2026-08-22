"""Backend/device selection for serial and particle-sharded execution."""

from __future__ import annotations

from dataclasses import dataclass

import jax

from core.config import ExecutionConfig


@dataclass(frozen=True)
class ExecutionPlan:
    """Resolved devices for one compiled particle execution policy."""

    config: ExecutionConfig
    devices: tuple

    @property
    def n_devices(self) -> int:
        return len(self.devices)

    @property
    def is_multi_device(self) -> bool:
        return self.n_devices > 1


def _devices_for_platform(platform: str) -> tuple:
    if platform == "auto":
        return tuple(jax.devices())
    if platform == "cpu":
        return tuple(jax.devices("cpu"))

    # JAX commonly exposes CUDA/ROCm accelerators through ``gpu``. The
    # fallbacks keep the selection helper usable with alternative plugins
    # without introducing backend-specific physics code.
    errors = []
    for backend in ("gpu", "cuda", "rocm", "metal"):
        try:
            devices = tuple(jax.devices(backend))
        except RuntimeError as exc:
            errors.append(str(exc))
            continue
        if devices:
            return devices
    raise RuntimeError("no GPU devices available; tried gpu/cuda/rocm/metal") from (
        RuntimeError(errors[-1]) if errors else None
    )


def resolve_execution(config: ExecutionConfig | None = None) -> ExecutionPlan:
    """Resolve a validated execution configuration to concrete local devices."""

    config = ExecutionConfig() if config is None else config
    available = _devices_for_platform(config.platform)
    if not available:
        raise RuntimeError(f"no devices available for platform={config.platform!r}")

    requested = 1 if config.mode == "serial" else config.n_devices
    if requested is None:
        requested = len(available)
    if requested > len(available):
        raise RuntimeError(
            f"requested {requested} {config.platform} devices, "
            f"but only {len(available)} are available"
        )
    return ExecutionPlan(config, tuple(available[:requested]))
