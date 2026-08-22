# JONTA Architecture

## 1. Purpose

JONTA is structured as a modular, accelerator-native kinetic Monte Carlo code. This document defines software boundaries, persistent state, data flow, and the intended JAX/multi-GPU execution model. Physical equations are in `physics.md`; discretization choices are in `numerics.md`.

## 2. Architectural goals

The architecture is designed around the following constraints:

- fixed-size weighted marker ensembles;
- FP64 physics by default; process-static FP32 mode exists for portability and
  reduced-precision experiments;
- regular, batched GPU execution;
- independent particle, large-angle, plasma-coupling, and diagnostic cadences;
- geometry- and field-representation agnostic core design;
- axisymmetric fields as the primary initial production use case;
- swappable integrators and physics operators;
- low-dimensional plasma coupling separated from high-dimensional particle evolution;
- particle sharding for multi-GPU execution;
- explicit module boundaries that are easy for humans and LLM agents to inspect.

The implementation should preserve these properties even as additional geometries, field representations, collision models, sources, and plasma closures are added.

## 3. Repository layout

```text
jonta/
├── README.md
├── AGENTS.md
├── pyproject.toml
├── docs/
│   ├── architecture.md
│   ├── code_map.md
│   ├── conventions.md
│   ├── numerics.md
│   ├── physics.md
│   └── validation.md
├── src/
│   ├── boundaries/
│   ├── collisions/
│   ├── core/
│   ├── coupling/
│   ├── deposition/
│   ├── diagnostics/
│   ├── fields/
│   ├── integrators/
│   ├── orbits/
│   ├── parallel/
│   ├── plasma/
│   ├── resampling/
│   ├── sources/
│   └── simulation.py
├── tests/
│   ├── unit/
│   ├── regression/
│   └── convergence/
├── examples/
└── scripts/
```

The project deliberately does not place the code under a second `src/jonta/` package layer. Modules are organized directly beneath `src/`.

## 4. State model

### 4.1 Particle state

The primary particle state is a fixed-size structure of arrays:

```text
ParticleState
├── kin.gamma[N]
├── kin.xi[N]
├── kin.x[N]
├── kin.y[N]
├── kin.phi[N]
├── weight[N]
├── alive[N]
└── pid[N]
```

The number of array slots `N` remains fixed during a run. Physical population growth/loss is represented through weights. `alive` is a mask used for boundary losses and zero-weight slots; it does not resize arrays.

The initial phase-space choice follows RAMc: Lorentz factor `gamma`, pitch `xi=p_parallel/p`, and configuration-space coordinates. Other particle coordinate representations may be introduced behind a compatible orbit layer if needed.

### 4.2 Field state

Field state is backend-specific. The initial circular RAMc backend stores radial arrays:

```text
CircularFieldProfiles
├── r[Nr]
├── e1[Nr]
└── q[Nr]
```

Other backends may contain analytic parameters, 2-D `(R,Z)` tables, spectral coefficients, or precomputed derivatives. The particle integrator must not depend on how the field data are stored.

### 4.3 Background plasma state

The initial radial background representation is:

```text
BackgroundProfiles
├── r[Nr]
├── ne_cm3[Nr]
├── te_ev[Nr]
├── ti_ev[Nr]
├── zeff[Nr]
└── eta_bar[Nr]
```

Charge-state populations and species-specific atomic data are separate state objects because their dimensions depend on the selected plasma composition.

### 4.4 BDF2 history

Two-step implicit evolution requires explicit history. Electric-field coupling stores `E1`, kinetic current, and resistivity at the two previous levels. Charge-state and thermal-energy solvers likewise retain their previous state. History is explicit state, not hidden module memory.

## 5. Module contracts

### 5.1 `fields/`

Owns field/profile representation and interpolation. It does not advance particles.

Current implementations:

- uniform 0-D field;
- circular RAMc radial profiles;
- regular-grid axisymmetric interpolation primitives.

Future implementations may include EFIT, MHD, spectral, and composite field backends.

### 5.2 `orbits/`

Owns deterministic characteristics. Radiation reaction belongs here because it is part of deterministic phase-space flow.

Current implementations:

- `zero_d_rhs`: uniform-field momentum-space model;
- `ramc_circular_rhs`: analytic circular RAMc guiding-center equations;
- synchrotron radiation model.

Orbit modules consume field state but do not know which integrator is used.

### 5.3 `integrators/`

Owns only deterministic time integration. Integrators accept an RHS callable and a state pytree.

Current implementations:

- Euler (debugging/reference);
- explicit midpoint/RK2;
- RK4.

Future structure-preserving or symplectic guiding-center integrators should implement the same construction-time contract.

### 5.4 `collisions/`

Small- and large-angle operators are independent modules.

- Small-angle default: Maxwellian-background test-particle friction, energy diffusion, and pitch scattering.
- Large-angle default: conservative weighted Møller gain-loss realization followed by fixed-N thinning.

Collision operators do not perform field interpolation or plasma evolution.

### 5.5 `sources/`

Contains kinetic sources independent of collision operators:

- tritium beta decay;
- Compton scattering;
- future external source models.

Source sampling and fixed-N injection are separated: physical source spectra are not tied to a particular resampling method.

### 5.6 `resampling/`

Owns statistical population control. The current production branching path
uses stratified random thinning/resampling with exact total-weight preservation;
ordinary multinomial resampling is retained as a reference implementation.
Alternative statistically valid algorithms can be added without changing
collision/source physics.

### 5.7 `deposition/`

Owns particle-to-grid moments. The initial implementation contains generic radial linear binning and a RAMc-compatible normalized parallel-current deposition.

### 5.8 `plasma/`

Owns background-plasma physics and atomic data:

- normalized Spitzer resistivity;
- electron/ion thermal-energy bookkeeping;
- charge-state rate matrices;
- preprocessed OPEN-ADAS-compatible coefficient tables.

Network access and raw atomic-data acquisition are explicitly outside simulation kernels.

### 5.9 `coupling/`

Owns low-dimensional implicit multiphysics updates:

- generic backward Euler/BDF2 linear solves;
- radial BDF2 electric-field solve;
- reduced algebraic Ohm law;
- circular safety-factor update;
- Picard particle/field coupling driver.

The coupling layer may re-run compiled particle blocks during a nonlinear iteration, but particle physics remains in its own modules.

### 5.10 `boundaries/`

Owns material/computational loss rules. Boundary models mask/zero marker weights rather than resize arrays.

### 5.11 `diagnostics/`

Owns reduced moments and validation quantities. Full particle dumps should be opt-in rather than the default output mechanism.

### 5.12 `parallel/`

Owns JAX sharding helpers. Physics modules should not contain explicit device-count logic.

### 5.13 `simulation.py`

Composes the selected orbit RHS, integrator, collision operators, and boundaries into compiled particle blocks. It should remain thin.

## 6. Timestep/data-flow hierarchy

Three independent physics cadences are central:

- particle timestep `dt_p`;
- large-angle interval `dt_LA`;
- plasma coupling interval `dt_c`.

Diagnostics have a fourth cadence.

A typical coupling interval is:

```text
state at t_n
   |
   | hold / predict plasma fields over coupling interval
   v
+--------------------------------------------------+
| compiled particle block                         |
|                                                  |
| repeated dt_p:                                  |
|   small-angle half step (optional Strang)       |
|   deterministic orbit + radiation               |
|   small-angle half step                         |
|   boundary mask                                 |
|                                                  |
| at configured cadence:                          |
|   conservative large-angle gain/loss            |
|   random thinning back to fixed N               |
+--------------------------------------------------+
   |
   v
local moment deposition
   |
   v
multi-GPU reduction of binned moments
   |
   v
BDF2 plasma / Ohm solve
   |
   +--> E1, Te, Ti, charge states, eta, optionally q
   |
   v
nonlinear convergence check / next Picard iterate
```

The large-angle interval does not define the plasma coupling interval and vice versa.

## 7. JAX execution model

### 7.1 Compilation boundary

The expensive particle evolution should run in compiled blocks containing many fixed particle timesteps. `jax.lax.scan` is used for regular timestep repetition.

Low-dimensional outer Picard iterations may remain Python-level initially because each iteration launches compiled particle/coupling kernels. This keeps the implementation transparent and avoids compiling an unnecessarily large nonlinear graph.

### 7.2 Fixed shapes

The persistent particle arrays keep shape `(N,)`. Møller gain/loss temporarily forms a fixed `(3N,)` candidate ensemble, then resamples back to `(N,)`. Because all shapes are known at trace time, this remains compatible with XLA compilation.

### 7.3 Memory layout

NamedTuple fields produce a structure-of-arrays representation at the JAX array level. Derived orbit quantities are temporary unless profiling demonstrates a benefit to storing them.

## 8. Multi-GPU architecture

Execution is selected explicitly with `core.config.ExecutionConfig`:

```python
ExecutionConfig(mode="serial", platform="cpu")
ExecutionConfig(mode="parallel", platform="cpu", n_devices=4)
ExecutionConfig(mode="parallel", platform="gpu")
```

Serial execution is the one-device reference path. Parallel execution uses
the same compiled particle block and shards only the leading particle axis;
the selected backend is resolved by `parallel.resolve_execution`. CPU and
CUDA therefore share the physics kernels. A parallel run with one available
device is valid but degenerate; actual scaling requires multiple devices.

The default decomposition is particle sharding:

```text
GPU 0          GPU 1          ...          GPU G-1
N/G markers    N/G markers                 N/G markers
    |              |                            |
 orbit/collisions  orbit/collisions            orbit/collisions
    |              |                            |
 local bins        local bins                  local bins
     \             |                           /
                  reduction
                      |
              global radial moments
                      |
              replicated plasma solve
```

Compact 1-D or 2-D field and plasma data should normally be replicated on each device. This avoids remote field lookup and keeps the dominant particle kernel communication-free.

For the 1-D RAMc problem, the amount of globally reduced data scales with `Nr`, not particle count, so weak scaling should be favorable.

## 9. Extension rules

Adding a new model should require a local implementation and tests rather than edits across the codebase.

Examples:

- new integrator: add `step(rhs, state, t, dt)` under `integrators/`;
- new field backend: add field state/sampler under `fields/` and, if needed, a geometry-specific RHS under `orbits/`;
- new small-angle operator: add a function under `collisions/` with the particle/background contract;
- new resampler: implement fixed-N particle input/output under `resampling/`;
- new plasma closure: add it under `plasma/` and consume it from the coupling layer.

## 10. Current implementation scope

The repository currently contains executable reference implementations for:

- 0-D uniform-field deterministic dynamics;
- circular 1-D RAMc guiding-center dynamics;
- fixed-step RK integration;
- Maxwellian-background small-angle collisions;
- conservative weighted Møller large-angle gain/loss;
- fixed-N stratified random thinning;
- radial current deposition;
- normalized Spitzer resistivity;
- BDF2 electric-field evolution;
- charge-state BDF2 infrastructure for preprocessed OPEN-ADAS rates;
- electron/ion thermal-energy bookkeeping;
- particle-first JAX sharding helpers.

These are a development baseline, not yet a claim of full physics validation against every RAMc benchmark.
