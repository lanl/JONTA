"""Small immutable configuration containers used inside JAX kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NamedTuple

ExecutionMode = Literal["serial", "parallel"]
ExecutionPlatform = Literal["auto", "cpu", "gpu"]


@dataclass(frozen=True)
class ExecutionConfig:
    """Host-side execution policy for compiled particle blocks.

    ``serial`` selects one JAX device and is the reference/debug path.
    ``parallel`` shards the leading particle axis across ``n_devices`` using
    the backend selected by ``platform``.  ``n_devices=None`` means all
    devices available on that backend.
    """

    mode: ExecutionMode = "serial"
    platform: ExecutionPlatform = "auto"
    n_devices: int | None = None

    def __post_init__(self):
        if self.mode not in ("serial", "parallel"):
            raise ValueError("execution mode must be 'serial' or 'parallel'")
        if self.platform not in ("auto", "cpu", "gpu"):
            raise ValueError("execution platform must be 'auto', 'cpu', or 'gpu'")
        if self.n_devices is not None and self.n_devices < 1:
            raise ValueError("n_devices must be positive when specified")
        if self.mode == "serial" and self.n_devices not in (None, 1):
            raise ValueError("serial execution accepts at most one device")


class OrbitNormalization(NamedTuple):
    """RAMc normalization parameters for circular guiding-center dynamics.

    epsilon
        Inverse aspect ratio ``a / R0``.
    c_tau_over_a
        ``c * tau_c0 / a``.
    a_omega_ce_over_c
        ``a * omega_ce0 / c``.
    alpha_syn
        Synchrotron strength ``tau_c0 / tau_s``.
    """

    epsilon: float
    c_tau_over_a: float
    a_omega_ce_over_c: float
    alpha_syn: float = 0.0


class SmallAngleConfig(NamedTuple):
    ne0_cm3: float
    coulog0: float
    pitch_scattering: bool = True
    friction: bool = True
    energy_scattering: bool = True
    n_sa: int = 100
    p_min: float = 1.0e-3
    partial_screening: bool = False
    impurity_fraction: float = 0.0
    impurity_nuclear_charge: float = 1.0
    impurity_charge_state: float = 1.0
    impurity_radius_abohr: float = 1.0
    impurity_mean_excitation_ev: float = 1.0
    screening_k: float = 5.0
    # Relativistic Coulomb-logarithm options used by the mixed
    # Fokker-Planck--Boltzmann large-angle formulation.  The default keeps
    # the legacy thermal-log behavior.
    relativistic_coulog: bool = False
    large_angle_reduced_coulog: bool = False
    large_angle_source_coulog: bool = False
    large_angle_gamma_min: float = 1.02
    gamma_floor: float = 1.0


class MollerConfig(NamedTuple):
    ne0_cm3: float
    coulog0: float
    gamma_min: float
    max_collision_fraction: float = 0.25
    bisection_steps: int = 36
    target_electron_factor: float = 1.0


class CadenceConfig(NamedTuple):
    """Independent simulation cadences expressed in particle steps."""

    dt_particle: float
    large_angle_every: int
    coupling_every: int
    diagnostics_every: int = 1


class ThermalSourceConfig(NamedTuple):
    """Optional fixed-capacity Maxwellian reservoir."""

    enabled: bool = False
    p_min: float = 1.0e-3
