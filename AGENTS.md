# JONTA Agent Guide

This file is the repository-wide contract for human contributors and LLM coding agents. Read it before modifying `src/`, physics equations, numerical methods, or benchmark cases.

## 1. Project intent

JONTA is a JAX-based, GPU-accelerated kinetic Monte Carlo code for runaway electrons. The priorities are, in order:

1. physical correctness;
2. numerical convergence and robustness;
3. GPU throughput and multi-GPU scalability;
4. statistical efficiency;
5. maintainability and future extensibility;
6. automatic differentiation when it comes naturally, but not at the expense of the items above.

The primary production targets are NVIDIA RTX PRO 6000 Blackwell and NVIDIA B200 GPUs. The core physics path uses FP64.

## 2. Architectural invariants

Do not violate these without an explicit design decision documented in `docs/architecture.md`.

- Particle arrays have fixed size during production evolution.
- Marker population changes are represented by weights and statistically sound thinning/resampling, not dynamic array growth.
- Particle dynamics are independent of a specific field representation.
- Geometry-specific equations belong in geometry-specific orbit/field modules.
- Field interpolation does not belong inside an integrator.
- Integrators do not own physics equations.
- Synchrotron radiation is part of deterministic orbit evolution.
- Small-angle and large-angle collision operators are independently replaceable.
- Large-angle avalanche physics is a linearized conservative Møller gain-loss model by default, not a source-only branching model.
- Particle, large-angle, plasma-coupling, and diagnostics cadences are independent.
- Plasma/Ohm coupling is a separate multiphysics layer; BDF2 is the reference implicit scheme.
- Serial and particle-sharded execution are explicit `ExecutionConfig` modes; serial is the trusted reference.
- Parallel particle evolution must not be enabled with large-angle population control until a distributed global resampler is implemented.
- High-level orchestration contains as little physics as possible.

## 3. Directory responsibilities

- `src/core/`: state, constants, configuration, initialization, RNG, small numerical helpers.
- `src/fields/`: field/profile representations and interpolation only.
- `src/orbits/`: deterministic characteristics, including radiation.
- `src/integrators/`: generic time integration of deterministic characteristics.
- `src/collisions/`: small- and large-angle collision physics.
- `src/sources/`: tritium, Compton, and external kinetic sources.
- `src/resampling/`: fixed-N thinning/resampling algorithms.
- `src/deposition/`: particle-to-grid moments and binning.
- `src/plasma/`: background plasma, atomic data, charge states, resistivity, energy equations.
- `src/coupling/`: implicit Ohm/electric-field and other multiphysics coupling algorithms.
- `src/boundaries/`: loss and boundary models.
- `src/diagnostics/`: reduced observables; avoid full-particle output by default.
- `src/parallel/`: JAX device/sharding helpers only.
- `src/simulation.py`: composition/orchestration of the above modules.

Lower-level modules must not import `simulation.py` or examples.

The concise LLM entry workflow, task-routing table, validation ladder, benchmark
policy, and handoff checklist are in `docs/agent_workflow.md`.

## 4. JAX and accelerator rules

- Enable and preserve FP64 (`jax_enable_x64=True`).
- Keep hot kernels JAX-native. Do not call NumPy, SciPy, filesystem APIs, network APIs, or host callbacks from compiled particle kernels.
- Prefer arrays with static shapes. Do not resize particle arrays inside a run.
- Prefer batched array algebra and `jax.lax` control flow over Python loops in hot paths.
- Python loops are acceptable for low-dimensional outer nonlinear coupling iterations when each expensive inner operation is compiled/device-resident.
- Do not add persistent per-particle arrays for quantities that are cheap to recompute unless profiling demonstrates a benefit.
- Avoid device-host synchronization in timestep loops and diagnostics.
- Particle sharding is the default multi-GPU decomposition. Compact axisymmetric field/plasma profiles should normally be replicated.
- Any performance optimization that materially complicates the equations must be benchmarked and documented.

## 5. RNG and statistical rules

- Random streams must not depend on device-local execution order.
- Tie particle stochastic streams to stable particle slot IDs, global step, and operator stream where practical.
- Resampling must preserve total represented weight exactly and other moments without bias in expectation.
- Do not resample more frequently than required by statistical quality or a fixed-N gain/loss/source operation.
- Any change to resampling must include tests for total weight and statistical bias/convergence.

## 6. Physics-change rules

When changing physics:

1. identify the governing equation and citation;
2. update `docs/physics.md`;
3. keep the old model available when it is a useful benchmark unless explicitly deprecated;
4. add or update a validation test;
5. distinguish a physical model change from a numerical realization change.

Never silently "fix" legacy RAMc physics while porting it. If a legacy expression appears inconsistent with the documentation, document the discrepancy and choose intentionally.

## 7. Numerical-change rules

- Integrator choice must be supported by convergence tests, not preference.
- A future symplectic/structure-preserving integrator must satisfy the same integrator contract as RK methods.
- Large-angle cadence must resolve the collision fraction; clipping in the reference Møller implementation is a safety guard, not a substitute for convergence.
- BDF2 startup uses a first-order implicit step or explicitly initialized history.
- Nonlinear particle/plasma coupling convergence must be checked with a residual, not a fixed iteration count alone.

## 8. Testing expectations

Use the smallest appropriate level:

- **unit tests**: local formulas, conservation identities, array contracts;
- **regression tests**: known JONTA/RAMc cases;
- **convergence tests**: timestep, particle number, spatial grid, collision cadence;
- **physics validation**: RAMc and published runaway-electron benchmarks;
- **performance tests**: particle-steps/s, RHS evaluations/s, memory/particle, scaling.

Run at minimum:

```bash
PYTHONPATH=src pytest -q
```

before delivering a code change. GPU-specific changes should also be tested on at least one CUDA device when available.

## 9. Documentation expectations

Keep these synchronized with code:

- `docs/physics.md`: physical equations and approximations;
- `docs/numerics.md`: discretization and stochastic algorithms;
- `docs/conventions.md`: normalization, signs, units, state layout;
- `docs/architecture.md`: module/data flow and dependency rules;
- `docs/validation.md`: benchmark definitions and acceptance strategy;
- `docs/code_map.md`: equation-to-implementation map;
- `docs/installation.md`: supported environment setup;
- `docs/benchmarks.md`: reproducible benchmark commands and artifact policy.
- `docs/agent_workflow.md`: LLM task routing, execution modes, validation, and handoff.

## 10. Agent workflow

Before editing:

1. read the relevant docs and neighboring modules;
2. search for existing implementations before creating duplicates;
3. identify which tests define expected behavior.

While editing:

- make focused changes;
- use explicit names and short files;
- preserve modular interfaces;
- put equations in docstrings/comments near non-obvious kernels;
- avoid unrelated refactors.

After editing:

- run tests;
- run formatting/linting if available;
- summarize changed files, tests, and any unresolved physics assumptions.

## 11. Anti-patterns

Do not introduce:

- dynamic marker creation/deletion that changes array shape;
- hidden global mutable state;
- geometry conditionals scattered through generic modules;
- host-side per-particle loops;
- runtime Python object dispatch inside hot kernels;
- silent FP32 downcasts;
- duplicated copies of the same physics formula in unrelated modules;
- full particle dumps as the default diagnostic path;
- direct OPEN-ADAS/network access from simulation kernels.
