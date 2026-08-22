# JONTA Equation-to-Code Map

## Purpose

This document is a navigation aid for maintainers and LLM agents. It maps the main equations/models to their implementation and tests. Keep it synchronized when moving or replacing physics kernels.

| Model / equation | Implementation | Primary tests / validation |
|---|---|---|
| Kinetic state `(gamma, xi, x, y, phi)` | `src/core/state.py` | unit tests across modules |
| RAMc normalization | `src/core/config.py`, `docs/conventions.md` | benchmark configuration review |
| Uniform 0-D Lorentz + synchrotron characteristics | `src/orbits/zero_d.py`, `src/orbits/radiation.py` | `tests/unit/test_zero_d.py` |
| Circular RAMc guiding-center characteristics | `src/orbits/ramc_circular.py` | `tests/unit/test_circular_orbit.py`; RAMc Secs. III.A, III.H, IV.A |
| Circular analytic field/profile sampling | `src/fields/circular.py`, `src/fields/profiles.py` | orbit tests |
| Axisymmetric numerical field interpolation primitive | `src/fields/interpolated.py` | `tests/unit/test_fields.py`; production interpolation convergence remains backend-specific |
| Euler / midpoint / RK4 | `src/integrators/explicit.py` | `tests/unit/test_integrators.py`; orbit convergence suite |
| Maxwellian test-particle small-angle collisions, including partially screened pitch scattering | `src/collisions/small_angle.py`, `src/collisions/coulomb.py` | `tests/unit/test_small_angle.py`; Maxwellian relaxation; McDevitt 2019 Figs. 3 and 6 |
| Møller differential/integrated cross section | `src/collisions/moller.py` | `tests/unit/test_moller.py`; RAMc Sec. II.C / Fig. 15 |
| Conservative Møller gain-loss realization | `src/collisions/moller.py::gain_loss_candidates` | energy/weight unit tests; avalanche growth validation |
| Fixed-capacity compaction and overflow thinning | `src/population/capacity.py`, `src/resampling/multinomial.py` | `tests/unit/test_population.py`; `tests/unit/test_resampling.py`; `benchmarks/test_population_statistics.py` |
| Tritium source (Ekmark 2024) | `src/sources/tritium.py` | `tests/unit/test_sources.py`; normalization integral TODO |
| Compton source (Ekmark 2024) | `src/sources/compton.py` | published Ekmark comparison TODO |
| Fixed-capacity source injection helper | `src/sources/base.py` | `tests/unit/test_sources_capacity.py`; capacity-statistics framework shared with validation |
| Absorbing radial/energy boundaries | `src/boundaries/basic.py` | orbit/loss regression TODO |
| Generic radial CIC binning | `src/deposition/radial.py` | `tests/unit/test_deposition.py`; RAMc current-profile validation TODO |
| RAMc normalized current deposition | `src/deposition/radial.py::deposit_ramc_parallel_current` | RAMc 1-D coupling tests TODO |
| Uniform 0-D/slab scalar moments | `src/deposition/slab.py` | `tests/unit/test_slab_deposition.py`; slab coupling regression |
| RAMc normalized Spitzer resistivity | `src/plasma/resistivity.py` | profile reference values TODO |
| Atomic table interpolation | `src/plasma/atomic.py` | OPEN-ADAS table tests TODO |
| Charge-state rate equation | `src/plasma/charge_state.py` | `tests/unit/test_charge_state.py` |
| Electron/ion energy bookkeeping | `src/plasma/energy.py` | `tests/unit/test_plasma_energy.py` |
| Algebraic Ohm law | `src/coupling/ohm.py` | `tests/unit/test_ohm_safety.py` |
| RAMc radial electric-field equation + BDF2 | `src/coupling/electric_field.py` | `tests/unit/test_coupling.py`; manufactured/BDF2 convergence TODO |
| Geometry-neutral particle/field Picard driver | `src/coupling/driver.py` | `tests/unit/test_coupling_driver.py` |
| Homogeneous slab algebraic Ohm adapter | `src/coupling/slab.py` | `tests/unit/test_slab_coupling.py` |
| RAMc circular safety-factor update | `src/coupling/safety_factor.py` | `tests/unit/test_ohm_safety.py`; RAMc profile regression TODO |
| Picard particle/field nonlinear coupling | `src/coupling/ramc1d.py` | coupled timestep convergence TODO |
| Geometry-neutral particle macrostep/reduction | `src/coupling/macrostep.py` | `tests/integration/test_coupling_macrostep.py` |
| Reduced moments | `src/diagnostics/moments.py` | operator tests |
| Avalanche growth / independent-replica statistics | `src/diagnostics/avalanche.py` | `benchmarks/test_large_angle_avalanche.py`; `benchmarks/test_population_statistics.py` |
| Guo-2017 runaway-vortex analytic diagnostics | `src/diagnostics/runaway_vortex.py` | `benchmarks/test_runaway_vortex.py` |
| Serial/parallel backend and device selection | `src/core/config.py::ExecutionConfig`, `src/parallel/execution.py` | `tests/integration/test_simulation.py`; CPU multi-device equivalence |
| Particle sharding | `src/parallel/sharding.py` | deterministic CPU multi-device test in `tests/integration/test_parallel_orbit.py`; local population test in `tests/integration/test_parallel_population.py` |
| Compiled particle block / operator composition | `src/simulation.py` | `tests/integration/test_simulation.py` |

## RAMc source mapping

The supplied legacy source is used as a reference, not copied structurally.

- `RAMcEx/src/push.cpp::RHSFunctionDKEanalytic` -> `src/orbits/ramc_circular.py`
- `RAMcEx/src/SmallAngleCollisions.cpp::DoSmallAngleCollision` -> `src/collisions/small_angle.py`
- `RAMcEx/src/LargeAngleCollisions.cpp::DoLargeAngleCollision` -> Møller cross-section/kinematic pieces in `src/collisions/moller.py`; JONTA uses conservative gain/loss plus local fixed-capacity control.
- `RAMcEx/src/EF-solvers.cpp` -> `src/coupling/electric_field.py`
- `RAMcEx/src/EF-solvers.cpp::DepositCurrent` -> `src/deposition/radial.py`
- `RAMcEx/src/EvaluateProfiles.cpp::Geteta` -> `src/plasma/resistivity.py` for the Spitzer branch.
- `RAMcEx/src/EvaluateProfiles.cpp::ComputeSafetyFactor` -> `src/coupling/safety_factor.py`.

## Intentional departures from RAMc

1. fixed particle count rather than dynamic birth/splitting;
2. fixed particle timestep rather than per-particle adaptive PETSc integration;
3. conservative Møller gain-loss treatment rather than source-only secondary creation;
4. JAX-native batched kernels rather than MPI master/slave work scheduling;
5. implicit BDF2 plasma coupling rather than the legacy adaptive field integrator;
6. separate electron/ion energy and charge-state framework with OPEN-ADAS data;
7. geometry/field backend modularity rather than circular geometry embedded in the overall code design.

### Avalanche initialization helpers

`src/core/initialization.py` also contains the optional analytic avalanche
warm-start utilities: the Rosenbluth--Putvinski growth coefficient and energy
scale, the radiation-limited O-point estimate, the momentum-dependent pitch
channel width, and `rosenbluth_legendre_markers()`.  These routines construct
only the initial marker cloud; they are never used as a replacement kinetic
solver.
