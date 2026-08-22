"""Strict host-side YAML configuration for simulations and benchmarks.

Configuration is parsed before JAX arrays/kernels are built. Raw mappings never
cross into physics modules; callers receive immutable typed objects and can
convert selected sections to legacy JAX-facing config containers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Literal, Mapping

import yaml

from .config import (
    CadenceConfig,
    ExecutionConfig,
    MollerConfig,
    OrbitNormalization,
    SmallAngleConfig,
)
from .precision import configure_precision

GeometryKind = Literal["slab", "circular"]
PrecisionKind = Literal["float32", "float64"]


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a mapping")
    return dict(value)


def _strict_mapping(value: Any, allowed: set[str], path: str) -> dict[str, Any]:
    data = _mapping(value, path)
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise ValueError(f"{path} contains unknown keys: {', '.join(unknown)}")
    return data


def _pair(value: Any, path: str, *, lower: float | None = None, upper: float | None = None):
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{path} must contain exactly two values")
    pair = (float(value[0]), float(value[1]))
    if pair[0] > pair[1]:
        raise ValueError(f"{path} lower bound exceeds upper bound")
    if lower is not None and pair[0] < lower:
        raise ValueError(f"{path} lower bound must be >= {lower}")
    if upper is not None and pair[1] > upper:
        raise ValueError(f"{path} upper bound must be <= {upper}")
    return pair


def _positive(value: Any, path: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise ValueError(f"{path} must be positive")
    return result


def _nonnegative(value: Any, path: str) -> float:
    result = float(value)
    if result < 0.0:
        raise ValueError(f"{path} must be nonnegative")
    return result


@dataclass(frozen=True)
class ModelConfig:
    geometry: GeometryKind = "slab"
    precision: PrecisionKind = "float64"
    field: str = "uniform"

    def __post_init__(self):
        if self.geometry not in ("slab", "circular"):
            raise ValueError("model.geometry must be 'slab' or 'circular'")
        if self.precision not in ("float32", "float64"):
            raise ValueError("model.precision must be 'float32' or 'float64'")


@dataclass(frozen=True)
class ExecutionSettings:
    mode: Literal["serial", "parallel"] = "serial"
    platform: Literal["auto", "cpu", "gpu"] = "auto"
    n_devices: int | None = None

    def __post_init__(self):
        if self.mode not in ("serial", "parallel"):
            raise ValueError("execution.mode must be 'serial' or 'parallel'")
        if self.platform not in ("auto", "cpu", "gpu"):
            raise ValueError("execution.platform must be 'auto', 'cpu', or 'gpu'")
        if self.n_devices is not None and self.n_devices < 1:
            raise ValueError("execution.n_devices must be positive")
        if self.mode == "serial" and self.n_devices not in (None, 1):
            raise ValueError("serial execution accepts at most one device")

    def to_runtime(self) -> ExecutionConfig:
        return ExecutionConfig(self.mode, self.platform, self.n_devices)


@dataclass(frozen=True)
class GeometryConfig:
    n_radial: int = 96
    minor_radius_cm: float = 100.0
    epsilon: float = 0.0
    q0: float = 2.1
    q2: float = 0.0
    c_tau_over_a: float = 1.0
    a_omega_ce_over_c: float = 1.0
    alpha_syn: float = 0.0

    def __post_init__(self):
        if self.n_radial < 2:
            raise ValueError("geometry.n_radial must be >= 2")
        _positive(self.minor_radius_cm, "geometry.minor_radius_cm")
        _nonnegative(self.epsilon, "geometry.epsilon")
        _positive(self.c_tau_over_a, "geometry.c_tau_over_a")
        _positive(self.a_omega_ce_over_c, "geometry.a_omega_ce_over_c")
        _nonnegative(self.alpha_syn, "geometry.alpha_syn")


@dataclass(frozen=True)
class FieldConfig:
    e_over_ec: float = 0.0
    e_scan: tuple[float, ...] = ()

    def __post_init__(self):
        try:
            scan = tuple(float(value) for value in self.e_scan)
        except (TypeError, ValueError) as exc:
            raise ValueError("field.e_scan must contain numeric values") from exc
        object.__setattr__(self, "e_scan", scan)
        if not self.e_scan and self.e_over_ec == 0.0:
            raise ValueError("field.e_over_ec or field.e_scan must be supplied")


@dataclass(frozen=True)
class BackgroundConfig:
    ne_cm3: float = 1.0e14
    te_ev: float = 1.0e3
    ti_ev: float = 1.0e3
    zeff: float = 1.0
    coulomb_log: float = 15.0

    def __post_init__(self):
        _positive(self.ne_cm3, "background.ne_cm3")
        _positive(self.te_ev, "background.te_ev")
        _positive(self.ti_ev, "background.ti_ev")
        if self.zeff <= 0.0:
            raise ValueError("background.zeff must be positive")
        _positive(self.coulomb_log, "background.coulomb_log")


@dataclass(frozen=True)
class ParticleConfig:
    markers: int = 4096
    total_weight: float = 1.0
    seed: int = 0
    gamma_range: tuple[float, float] = (1.01, 2.0)
    xi_range: tuple[float, float] = (-1.0, 1.0)
    radius_range: tuple[float, float] = (0.0, 0.9)
    pid_policy: str = "stable"

    def __post_init__(self):
        if self.markers < 1:
            raise ValueError("particles.markers must be positive")
        _positive(self.total_weight, "particles.total_weight")
        if self.gamma_range[0] < 1.0:
            raise ValueError("particles.gamma_range must start at gamma >= 1")
        _pair(self.gamma_range, "particles.gamma_range", lower=1.0)
        _pair(self.xi_range, "particles.xi_range", lower=-1.0, upper=1.0)
        _pair(self.radius_range, "particles.radius_range", lower=0.0, upper=1.0)
        if self.pid_policy != "stable":
            raise ValueError("particles.pid_policy must be 'stable'")


@dataclass(frozen=True)
class TimestepConfig:
    dt_particle: float = 1.0e-3
    large_angle_every: int = 0
    coupling_every: int = 1
    diagnostics_every: int = 1

    def __post_init__(self):
        _positive(self.dt_particle, "timesteps.dt_particle")
        if self.large_angle_every < 0:
            raise ValueError("timesteps.large_angle_every must be >= 0")
        if self.coupling_every < 1 or self.diagnostics_every < 1:
            raise ValueError("timesteps cadence values must be positive")

    def to_runtime(self) -> CadenceConfig:
        return CadenceConfig(
            self.dt_particle,
            self.large_angle_every,
            self.coupling_every,
            self.diagnostics_every,
        )


@dataclass(frozen=True)
class SmallAngleSettings:
    pitch_scattering: bool = True
    friction: bool = True
    energy_scattering: bool = True
    max_nu_dt: float = 0.5
    partial_screening: bool = False
    relativistic_coulog: bool = False
    large_angle_reduced_coulog: bool = False
    large_angle_source_coulog: bool = False
    large_angle_gamma_min: float = 1.02
    impurity_fraction: float = 0.0
    impurity_nuclear_charge: float = 1.0
    impurity_charge_state: float = 1.0
    impurity_radius_abohr: float = 1.0
    impurity_mean_excitation_ev: float = 1.0
    screening_k: float = 5.0

    def __post_init__(self):
        if self.max_nu_dt <= 0.0:
            raise ValueError("collisions.small_angle.max_nu_dt must be positive")
        if self.large_angle_gamma_min < 1.0:
            raise ValueError("collisions.small_angle.large_angle_gamma_min must be >= 1")
        if self.impurity_fraction < 0.0:
            raise ValueError("collisions.small_angle.impurity_fraction must be nonnegative")
        if self.impurity_nuclear_charge <= 0.0 or self.impurity_charge_state <= 0.0:
            raise ValueError("collisions.small_angle impurity charges must be positive")
        if self.impurity_radius_abohr <= 0.0 or self.impurity_mean_excitation_ev <= 0.0:
            raise ValueError("collisions.small_angle impurity scales must be positive")
        if self.screening_k <= 0.0:
            raise ValueError("collisions.small_angle.screening_k must be positive")

    def to_runtime(self, background: BackgroundConfig) -> SmallAngleConfig:
        return SmallAngleConfig(
            background.ne_cm3,
            background.coulomb_log,
            self.pitch_scattering,
            self.friction,
            self.energy_scattering,
            self.max_nu_dt,
            self.partial_screening,
            self.impurity_fraction,
            self.impurity_nuclear_charge,
            self.impurity_charge_state,
            self.impurity_radius_abohr,
            self.impurity_mean_excitation_ev,
            self.screening_k,
            relativistic_coulog=self.relativistic_coulog,
            large_angle_reduced_coulog=self.large_angle_reduced_coulog,
            large_angle_source_coulog=self.large_angle_source_coulog,
            large_angle_gamma_min=self.large_angle_gamma_min,
        )


@dataclass(frozen=True)
class LargeAngleSettings:
    enabled: bool = False
    gamma_min: float = 1.02
    max_collision_fraction: float = 0.25
    bisection_steps: int = 36
    conservative: bool = True
    target_electron_factor: float = 1.0

    def __post_init__(self):
        if self.gamma_min < 1.0:
            raise ValueError("collisions.large_angle.gamma_min must be >= 1")
        if not 0.0 < self.max_collision_fraction <= 1.0:
            raise ValueError("collisions.large_angle.max_collision_fraction must lie in (0, 1]")
        if self.bisection_steps < 1:
            raise ValueError("collisions.large_angle.bisection_steps must be positive")
        if self.target_electron_factor <= 0.0:
            raise ValueError("collisions.large_angle.target_electron_factor must be positive")

    def to_runtime(self, background: BackgroundConfig) -> MollerConfig:
        return MollerConfig(
            background.ne_cm3,
            background.coulomb_log,
            self.gamma_min,
            self.max_collision_fraction,
            self.bisection_steps,
            self.target_electron_factor,
        )


@dataclass(frozen=True)
class CollisionConfig:
    small_angle: SmallAngleSettings = field(default_factory=SmallAngleSettings)
    large_angle: LargeAngleSettings = field(default_factory=LargeAngleSettings)


@dataclass(frozen=True)
class PopulationConfig:
    capacity: int | None = None
    thinning: str = "stratified"
    overflow_only: bool = True
    thinning_frequency: int = 1

    def __post_init__(self):
        if self.capacity is not None and self.capacity < 1:
            raise ValueError("population.capacity must be positive")
        if self.thinning != "stratified":
            raise ValueError("population.thinning must be 'stratified'")
        if self.thinning_frequency < 1:
            raise ValueError("population.thinning_frequency must be positive")


@dataclass(frozen=True)
class DiagnosticsConfig:
    fit_start_fraction: float = 0.35
    bins: int = 320
    sample_every: int = 1

    def __post_init__(self):
        if not 0.0 <= self.fit_start_fraction < 1.0:
            raise ValueError("diagnostics.fit_start_fraction must lie in [0, 1)")
        if self.bins < 1 or self.sample_every < 1:
            raise ValueError("diagnostics bins/sample_every must be positive")


@dataclass(frozen=True)
class ReferenceConfig:
    data_file: str | None = None
    comparison: str = "none"
    growth_rate_sem: float | None = None
    r2_min: float | None = None

    def __post_init__(self):
        if self.growth_rate_sem is not None and self.growth_rate_sem < 0.0:
            raise ValueError("reference.growth_rate_sem must be nonnegative")
        if self.r2_min is not None and not 0.0 <= self.r2_min <= 1.0:
            raise ValueError("reference.r2_min must lie in [0, 1]")


@dataclass(frozen=True)
class OutputConfig:
    directory: str = "benchmark_results"
    formats: tuple[str, ...] = ("json", "csv", "png")
    save_histories: bool = True
    save_final_state: bool = False

    def __post_init__(self):
        allowed = {"json", "csv", "png"}
        if not set(self.formats) <= allowed:
            raise ValueError("output.formats contains unsupported format")


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    reference: str | None = None
    description: str = ""
    replicas: int = 1

    def __post_init__(self):
        if not self.name:
            raise ValueError("benchmark.name must not be empty")
        if self.replicas < 1:
            raise ValueError("benchmark.replicas must be positive")


@dataclass(frozen=True)
class JontaConfig:
    model: ModelConfig
    execution: ExecutionSettings
    geometry: GeometryConfig
    field: FieldConfig
    background: BackgroundConfig
    particles: ParticleConfig
    timesteps: TimestepConfig
    collisions: CollisionConfig
    population: PopulationConfig
    diagnostics: DiagnosticsConfig
    benchmark: BenchmarkConfig
    reference: ReferenceConfig
    output: OutputConfig
    source_path: str | None = None

    @property
    def orbit_normalization(self) -> OrbitNormalization:
        return OrbitNormalization(
            self.geometry.epsilon,
            self.geometry.c_tau_over_a,
            self.geometry.a_omega_ce_over_c,
            self.geometry.alpha_syn,
        )

    @property
    def execution_config(self) -> ExecutionConfig:
        return self.execution.to_runtime()

    @property
    def cadence_config(self) -> CadenceConfig:
        return self.timesteps.to_runtime()

    @property
    def small_angle_config(self) -> SmallAngleConfig:
        return self.collisions.small_angle.to_runtime(self.background)

    @property
    def moller_config(self) -> MollerConfig:
        return self.collisions.large_angle.to_runtime(self.background)

    def apply_precision(self):
        """Apply process-static precision before JAX array/JIT creation."""

        return configure_precision(self.model.precision)

    def resolved_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("source_path", None)
        return data


_TOP_KEYS = {
    "model",
    "execution",
    "geometry",
    "field",
    "background",
    "particles",
    "timesteps",
    "collisions",
    "population",
    "diagnostics",
    "benchmark",
    "reference",
    "output",
}


def _construct(cls, value: Any, path: str):
    data = _strict_mapping(value, {item.name for item in fields(cls)}, path)
    return cls(**data)


def config_from_mapping(mapping: Mapping[str, Any], *, source_path: str | Path | None = None) -> JontaConfig:
    """Validate mapping and construct immutable runtime configuration."""

    data = _strict_mapping(mapping, _TOP_KEYS, "config")
    model = _construct(ModelConfig, data.get("model", {}), "model")
    execution = _construct(ExecutionSettings, data.get("execution", {}), "execution")
    geometry = _construct(GeometryConfig, data.get("geometry", {}), "geometry")
    field_config = _construct(FieldConfig, data.get("field", {}), "field")
    background = _construct(BackgroundConfig, data.get("background", {}), "background")
    particle_data = _strict_mapping(data.get("particles", {}), {item.name for item in fields(ParticleConfig)}, "particles")
    if "gamma_range" in particle_data:
        particle_data["gamma_range"] = _pair(particle_data["gamma_range"], "particles.gamma_range", lower=1.0)
    if "xi_range" in particle_data:
        particle_data["xi_range"] = _pair(particle_data["xi_range"], "particles.xi_range", lower=-1.0, upper=1.0)
    if "radius_range" in particle_data:
        particle_data["radius_range"] = _pair(particle_data["radius_range"], "particles.radius_range", lower=0.0, upper=1.0)
    particles = ParticleConfig(**particle_data)
    timesteps = _construct(TimestepConfig, data.get("timesteps", {}), "timesteps")
    collision_data = _strict_mapping(data.get("collisions", {}), {"small_angle", "large_angle"}, "collisions")
    collisions = CollisionConfig(
        _construct(SmallAngleSettings, collision_data.get("small_angle", {}), "collisions.small_angle"),
        _construct(LargeAngleSettings, collision_data.get("large_angle", {}), "collisions.large_angle"),
    )
    population = _construct(PopulationConfig, data.get("population", {}), "population")
    diagnostics = _construct(DiagnosticsConfig, data.get("diagnostics", {}), "diagnostics")
    benchmark_data = _strict_mapping(data.get("benchmark", {}), {item.name for item in fields(BenchmarkConfig)}, "benchmark")
    if "name" not in benchmark_data:
        raise ValueError("benchmark.name is required")
    benchmark = BenchmarkConfig(**benchmark_data)
    reference = _construct(ReferenceConfig, data.get("reference", {}), "reference")
    output_data = _strict_mapping(data.get("output", {}), {item.name for item in fields(OutputConfig)}, "output")
    if "formats" in output_data:
        output_data["formats"] = tuple(output_data["formats"])
    output = OutputConfig(**output_data)
    if model.geometry == "circular" and geometry.epsilon <= 0.0:
        raise ValueError("geometry.epsilon must be positive for circular geometry")
    if population.capacity is not None and population.capacity != particles.markers:
        raise ValueError("population.capacity must equal particles.markers")
    if source_path is not None:
        source = Path(source_path).resolve()
        if not Path(output.directory).is_absolute():
            output = OutputConfig(
                str((source.parent / output.directory).resolve()),
                output.formats,
                output.save_histories,
                output.save_final_state,
            )
        reference_path = reference.data_file
        if reference_path is not None and not Path(reference_path).is_absolute():
            reference = ReferenceConfig(
                str((source.parent / reference_path).resolve()),
                reference.comparison,
                reference.growth_rate_sem,
                reference.r2_min,
            )
        source_path = str(source)
    return JontaConfig(
        model,
        execution,
        geometry,
        field_config,
        background,
        particles,
        timesteps,
        collisions,
        population,
        diagnostics,
        benchmark,
        reference,
        output,
        None if source_path is None else str(source_path),
    )


def load_config(path: str | Path) -> JontaConfig:
    """Load, validate, and resolve YAML configuration from ``path``."""

    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return config_from_mapping(raw, source_path=config_path)


def dump_resolved_config(config: JontaConfig, path: str | Path):
    """Write deterministic resolved YAML beside benchmark outputs."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.resolved_dict(), handle, sort_keys=True)
