from pathlib import Path

import pytest

from core.configuration import config_from_mapping, dump_resolved_config, load_config
from core.precision import configure_precision, current_precision


def _minimal_config():
    return {
        "model": {"geometry": "slab", "precision": "float64"},
        "field": {"e_over_ec": 3.2321},
        "particles": {"markers": 128, "gamma_range": [10.0, 20.0], "xi_range": [-1.0, -0.8]},
        "population": {"capacity": 128},
        "benchmark": {"name": "mcdevitt-b3", "replicas": 4},
    }


def test_yaml_config_constructs_typed_runtime_sections():
    config = config_from_mapping(_minimal_config())

    assert config.model.geometry == "slab"
    assert config.particles.gamma_range == (10.0, 20.0)
    assert config.execution_config.mode == "serial"
    assert config.cadence_config.dt_particle == pytest.approx(1.0e-3)
    assert config.small_angle_config.ne0_cm3 == pytest.approx(1.0e14)
    assert config.moller_config.gamma_min == pytest.approx(1.02)
    assert config.thermal_source_config.enabled is False


def test_config_accepts_optional_thermal_source_settings():
    config = config_from_mapping(
        {
            **_minimal_config(),
            "sources": {"thermal": {"enabled": True, "p_min": 2.0e-3}},
        }
    )
    assert config.thermal_source_config.enabled is True
    assert config.thermal_source_config.p_min == pytest.approx(2.0e-3)


def test_config_accepts_optional_momentum_boundaries():
    config = config_from_mapping(
        {
            **_minimal_config(),
            "boundaries": {
                "momentum": {
                    "high_absorbing": True,
                    "p_max": 8.0,
                }
            },
        }
    )
    assert config.boundaries.momentum.high_absorbing is True
    assert config.boundaries.momentum.p_max == pytest.approx(8.0)


def test_config_rejects_unknown_and_missing_required_keys():
    unknown = _minimal_config()
    unknown["not_a_section"] = {}
    with pytest.raises(ValueError, match="unknown keys"):
        config_from_mapping(unknown)

    missing_name = _minimal_config()
    del missing_name["benchmark"]["name"]
    with pytest.raises(ValueError, match="benchmark.name is required"):
        config_from_mapping(missing_name)


def test_config_rejects_inconsistent_active_angle_partition():
    invalid = _minimal_config()
    invalid["collisions"] = {
        "small_angle": {
            "large_angle_reduced_coulog": True,
            "large_angle_gamma_min": 1.02,
        },
        "large_angle": {"gamma_min": 1.10},
    }
    with pytest.raises(ValueError, match="must match"):
        config_from_mapping(invalid)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("timesteps", {"dt_particle": 0.0}),
        ("geometry", {"minor_radius_cm": -1.0}),
        ("particles", {"gamma_range": [0.5, 2.0]}),
        ("execution", {"mode": "serial", "n_devices": 2}),
    ],
)
def test_config_rejects_physically_invalid_values(path, value):
    invalid = _minimal_config()
    invalid[path] = value
    with pytest.raises(ValueError):
        config_from_mapping(invalid)


def test_load_and_dump_resolve_paths_deterministically(tmp_path: Path):
    source = tmp_path / "case.yaml"
    source.write_text(
        "\n".join(
            [
                "field:",
                "  e_over_ec: 3.2",
                "benchmark:",
                "  name: path-case",
                "reference:",
                "  data_file: reference.csv",
                "output:",
                "  directory: results",
            ]
        ),
        encoding="utf-8",
    )
    config = load_config(source)
    assert config.source_path == str(source.resolve())
    assert config.reference.data_file == str((tmp_path / "reference.csv").resolve())
    assert config.output.directory == str((tmp_path / "results").resolve())

    resolved = tmp_path / "resolved.yaml"
    dump_resolved_config(config, resolved)
    reloaded = load_config(resolved)
    assert reloaded.resolved_dict() == config.resolved_dict()


def test_config_applies_process_precision():
    previous = current_precision()
    try:
        config = config_from_mapping({**_minimal_config(), "model": {"precision": "float32"}})
        assert config.apply_precision().bits == 32
    finally:
        configure_precision(previous)


@pytest.mark.parametrize(
    "path",
    [
        Path("benchmarks/slab/avalanche_decay/config.yaml"),
        Path("benchmarks/one_d/radial_avalanche/config.yaml"),
    ],
)
def test_checked_in_benchmark_templates_are_loadable(path):
    config = load_config(path)
    assert config.benchmark.name
    assert config.output.directory
