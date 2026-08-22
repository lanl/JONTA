# Configuration

JONTA uses a strict, host-side YAML configuration for simulations and physical
benchmarks. The loader validates the complete parameter tree before JAX arrays
or compiled kernels are created. Physics kernels receive typed runtime
containers, never raw YAML dictionaries.

## Layout

The top-level sections are:

| Section | Contents |
| --- | --- |
| `model` | geometry family and process precision |
| `execution` | serial/parallel mode, backend, device count |
| `geometry` | grid and orbit normalization |
| `field` | fixed field and optional scan values |
| `background` | density, temperatures, charge, Coulomb logarithm |
| `particles` | fixed marker capacity, seed, initialization ranges |
| `timesteps` | particle, large-angle, coupling, diagnostic cadences |
| `collisions` | small-angle and Moller operator controls |
| `population` | fixed-capacity thinning policy |
| `diagnostics` | history sampling and exponential-fit controls |
| `benchmark` | name, provenance description, replicas |
| `reference` | comparison data and acceptance targets |
| `output` | result directory, formats, saved artifacts |

Unknown keys are errors. Ranges, capacities, cadences, geometry requirements,
and reference thresholds are checked at load time. Relative output/reference
paths are resolved relative to the YAML file, so a benchmark is reproducible
from any working directory.

## Use

```python
from core.configuration import load_config

config = load_config("benchmarks/slab/avalanche_decay/config.yaml")
config.apply_precision()  # before creating arrays or JIT kernels
runtime_execution = config.execution_config
runtime_cadence = config.cadence_config
```

To archive the exact resolved inputs alongside results:

```python
from core.configuration import dump_resolved_config, load_config

config = load_config("benchmarks/slab/avalanche_decay/config.yaml")
dump_resolved_config(config, "benchmark_results/slab_b3/config.resolved.yaml")
```

The same validation is available without importing Python code manually:

```bash
PYTHONPATH=src python scripts/resolve_config.py \
  benchmarks/slab/avalanche_decay/config.yaml \
  --output benchmark_results/slab_b3/config.resolved.yaml
```

Configurations are stored with their consuming benchmark under
[`../benchmarks/slab/`](../benchmarks/slab/) and [`../benchmarks/one_d/`](../benchmarks/one_d/).
Benchmark drivers are being
migrated to consume this schema; until that migration is complete, a driver's
CLI defaults remain documented in its module and the YAML template is the
canonical parameter record for new runs.

Precision is process-static. Select `float64` for validation and supported CPU
or accelerator backends. `float32` is available for experiments and backends
without reliable FP64 support.
