# JONTA Numerical Methods

## 1. Purpose

This document defines the numerical realization of the physics in `physics.md`: marker representation, timestep hierarchy, deterministic integration, stochastic collisions, conservative large-angle gain/loss, fixed-capacity population control, deposition, and implicit plasma coupling.

## 2. Precision

The core physics path defaults to IEEE FP64. JONTA selects one process-static
real dtype through `core.precision`; particle coordinates, weights, profiles,
collision coefficients, deposited moments, and implicit plasma solves follow
that selection. FP32 is experimental and requires independent convergence
evidence; FP64 remains reference precision.

The target Blackwell GPUs support FP64; mixed precision is not an initial design requirement.

## 3. Fixed-capacity weighted Monte Carlo ensemble

The kinetic distribution is represented by markers

\[
\{z_i,w_i\}_{i=1}^{N},
\]

with fixed per-device capacity `C`. For an observable \(A\),

\[
\int f A\,d\Gamma \approx \sum_i w_i A(z_i).
\]

The physical particle count represented by the ensemble is

\[
W=\sum_i w_i,
\]

and need not be constant because sources and avalanche can transfer particles from the background into the kinetic population.

Array size remains constant. Losses set marker weight to zero and `alive=False`.
Gain/loss and source operations create temporary candidate ensembles, compact live
candidates into free slots, and thin only when local live candidates exceed `C`.

### 3.1 Effective sample size

A useful weight-quality diagnostic is

\[
N_{\rm eff}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}.
\]

Resampling should not be invoked merely because a large-angle event occurred.
It is required on local capacity overflow, or by an explicitly configured ESS
policy after independent convergence evidence.

## 4. Independent timestep hierarchy

JONTA does not use a single global meaning of "macro timestep." The relevant cadences are:

- particle timestep \(\Delta t_p\);
- large-angle interval \(\Delta t_{LA}\);
- plasma-coupling interval \(\Delta t_c\);
- diagnostic/output interval.

Typically

\[
\Delta t_{LA}=N_{LA}\Delta t_p,
\qquad
\Delta t_c=N_c\Delta t_p,
\]

with integer ratios for efficient compiled loops, but there is no physical requirement that \(\Delta t_{LA}=\Delta t_c\).

The particle timestep is chosen from orbit and small-angle convergence. The large-angle interval is chosen from collision-fraction convergence. The coupling interval is chosen from nonlinear plasma/Ohm robustness.

## 5. Particle microstep

The reference particle step uses operator splitting between deterministic evolution and the small-angle stochastic operator. The default construction uses Strang-like splitting:

\[
\mathcal P_{\Delta t_p}
=
C_{SA}(\Delta t_p/2)
\,G(\Delta t_p)\,
C_{SA}(\Delta t_p/2),
\]

where \(G\) advances the deterministic guiding-center plus radiation characteristics.

This mirrors the useful separation in RAMc while keeping every marker on the same fixed particle timestep.

The split ordering is an algorithmic choice. Convergence tests, not historical compatibility alone, determine whether this remains the production default.

## 6. Deterministic integration

The deterministic equations are advanced by a swappable fixed-step integrator. The current reference implementations are:

- Euler, for debugging only;
- explicit midpoint/RK2;
- RK4.

The integrator contract is conceptually

```python
state_next = integrator(rhs, state, t, dt)
```

and contains no field or collision-specific logic.

### 6.1 Integrator selection

Integrator choice will be based on error per field/RHS evaluation. Relevant convergence measures include:

- magnetic-moment conservation;
- toroidal canonical-momentum conservation in axisymmetry;
- orbit topology and finite-orbit-width behavior;
- Ware-pinch velocity;
- transport and runaway-growth benchmarks.

A future symplectic/structure-preserving guiding-center integrator is explicitly anticipated. Because guiding-center dynamics are noncanonical, such a method must be appropriate to the guiding-center Poisson/Hamiltonian structure rather than a generic canonical leapfrog inserted without derivation.

## 7. Small-angle Monte Carlo operator

The initial operator follows the RAMc Maxwellian-background test-particle model. Pitch and energy random increments are Gaussian. The collision coefficients are evaluated at the beginning of the collision substep.

Pitch is reflected at \(|\xi|=1\), implementing the Neumann boundary used by RAMc. Energy is reflected at \(\gamma=1\).

Unlike legacy RAMc, the initial JONTA implementation does not use per-particle adaptive collision subcycling. A fixed \(\Delta t_p\) is selected so that the relevant collisional statistics converge. The reference implementation also limits \(\nu_D\Delta t\) with the same kind of safety cap used by RAMc; reaching that cap systematically indicates that \(\Delta t_p\) is too large for the intended physical regime.

## 8. Conservative large-angle Møller realization

### 8.1 Collision fraction

For a marker with state \(z_i\), the large-angle operator computes the fraction

\[
q_i \simeq \nu_{LA}(z_i)\Delta t_{LA}
\]

of its represented weight participating in a Møller collision during the large-angle interval. In the RAMc normalization the reference rate is based on the integrated Møller cross section,

\[
q_i =
\frac{n_e/n_{e0}}{4\pi\ln\Lambda_0}
\,v_i\,\sigma_M(\gamma_i;\gamma_{\min})\,
\Delta t_{LA},
\]

with an optional target-electron factor for bound-electron models.

`MollerConfig.max_collision_fraction` is a safety guard. If the raw fraction approaches or exceeds it, the correct remedy is normally a smaller large-angle interval or subcycling, not reliance on clipping.

### 8.2 Weighted gain-loss candidates

For incoming marker weight \(w_i\), one sampled collision pair generates three weighted candidates:

\[
\begin{array}{lll}
\text{uncollided} &: z_i, & w_i(1-q_i),\\
\text{outgoing primary} &: z_{3,i}, & w_iq_i,\\
\text{outgoing target/secondary} &: z_{4,i}, & w_iq_i.
\end{array}
\]

The represented kinetic population therefore changes from \(w_i\) to \(w_i(1+q_i)\), as expected when background electrons are promoted into the energetic population.

For the cold-target collision,

\[
(\gamma_3-1)+(\gamma_4-1)=\gamma_i-1,
\]

so the gain-loss candidate ensemble conserves kinetic energy exactly before resampling.

### 8.3 Sampling outgoing energy

RAMc provides an analytic expression for the integrated Møller cross section above a secondary-energy cutoff. JONTA uses that expression as a cumulative tail integral and inverts it with fixed-iteration bisection. This avoids a variable-length acceptance/rejection loop inside a GPU kernel.

The number of bisection iterations is fixed at compilation/runtime configuration and therefore produces regular accelerator work.

### 8.4 Sampling outgoing pitch

After the secondary energy is sampled, the cold-target Møller scattering angle is

\[
\cos\theta_s=
\sqrt{
\frac{(\gamma_0+1)(\gamma_s-1)}
{(\gamma_0-1)(\gamma_s+1)}
}.
\]

A uniform azimuth about the incoming momentum gives

\[
\xi_s
=
\xi_0\cos\theta_s
+
\sqrt{1-\xi_0^2}\sin\theta_s\cos\chi,
\]

which is the direct sampling representation of the RAMc \(\Pi\) pitch distribution. The outgoing primary pitch is reconstructed from momentum conservation.

### 8.5 Fixed-capacity population control

The `3C` candidate work array is first compacted into `C` fixed slots without
resampling when its live candidate count fits. If live candidates exceed `C`,
the reference algorithm is **stratified random resampling** using normalized
candidate weights

\[
P_j=\frac{w_j}{\sum_kw_k}.
\]

The cumulative distribution is divided into `C` equal-probability strata and
one uniform random variate is drawn independently inside each stratum. Selected
markers receive equal weight

\[
w'=\frac{\sum_kw_k}{C}.
\]

This preserves total represented weight exactly and preserves other moments
without bias in expectation.  Compared with ordinary multinomial resampling it
substantially reduces branching noise while retaining genuinely random local
overflow thinning. It still introduces sampling variance; therefore capacity
and large-angle cadence must be convergence parameters.

Other statistically valid resamplers may be added behind the same local
capacity contract; global multinomial resampling remains available as a
reference implementation.

### 8.6 Statistical contract for branching calculations

Each device-local fixed-capacity representation is one **correlated particle
ensemble**, not independent samples after resampling. The reference
statistical protocol is therefore:

1. overflow resampling returns equal-weight markers, with the represented
   population carried by
   \(W=\sum_i w_i\);
2. uncertainties on avalanche growth rates, thresholds, spectra, and related
   branching observables are estimated from **independent PRNG replica
   ensembles**;
3. marker-count convergence is checked by verifying ordinary Monte-Carlo
   scaling, approximately \(\sigma\propto N^{-1/2}\), when that scaling is
   expected;
4. the large-angle macrostep is independently refined.  The raw maximum
   one-step collision fraction must remain below the clipping limit, and a
   resolved calculation should demonstrate invariance as \(\Delta t_{LA}\) is
   decreased.  Once this is established, use the **largest resolved**
   \(\Delta t_{LA}\) in production so random resampling is not performed more
   often than necessary;
5. CPU validation uses the same JIT-compiled kernels and estimators as GPU
   production runs.  GPU runs increase marker count, replica count, and/or
   duration rather than changing the algorithm.

A cheap two-type linear branching test with a known dominant eigenvalue is
included in `benchmarks/test_population_statistics.py` to verify this
contract independently of runaway-electron physics.

## 9. Tritium, Compton, and external sources

Physical source models return phase-space source densities and/or source samplers independently of capacity management.

A generic fixed-capacity source update is:

1. sample a fixed number of source candidates from the physical source spectrum;
2. assign total expected source weight over the source interval;
3. concatenate existing and source candidates;
4. compact into capacity, thinning only on overflow.

This keeps source physics independent of the population-control algorithm.

## 10. Boundary losses

Material/computational losses do not delete array entries. A lost marker is represented by

\[
w_i\rightarrow0,
\qquad
\texttt{alive}_i\rightarrow\texttt{False}.
\]

Subsequent resampling can repopulate fixed computational slots.

## 11. Radial deposition

For the circular 1-D model, particle moments are deposited onto a radial grid with linear cloud-in-cell weights. Generic binned moments use JAX array reductions. The RAMc current depositor retains the legacy geometric normalization and center regularity condition.

With large particle counts, statistical accuracy is expected to dominate over high-order particle shapes; more elaborate deposition can be added if convergence studies justify it.

### 11.1 Scalar slab / 0-D deposition

Uniform-field and slab models do not require radial bins, circular safety
factor, or flux-surface Jacobians. Their depositor returns scalar current and
represented weight. Any volume or normalization factor is passed explicitly
as `current_scale`; coupling only sees the resulting `ParticleMoments`.

This makes 0-D coupling a geometry adapter, not a second particle algorithm.

## 12. Electric-field coupling

### 12.1 BDF2

The reference radial field equation is

\[
\frac{\partial E_1}{\partial t}
=
\frac{\eta}{4\pi}L_rE_1
+E_1\frac{\partial\ln\eta}{\partial t}
-\eta G(r)\frac{\partial j_{\rm kin}}{\partial t},
\]

where `L_r` is the selected circular radial diffusion operator and `G(r)` represents optional geometry factors.

Constant-step BDF2 uses

\[
\left.\frac{\partial y}{\partial t}\right|_{n+1}
\approx
\frac{3y^{n+1}-4y^n+y^{n-1}}{2\Delta t_c}.
\]

The diffusion term and multiplicative resistivity term are evaluated at the new plasma level, producing a tridiagonal implicit field solve. JONTA uses JAX's tridiagonal linear solver for this low-dimensional system.

### 12.2 Startup

The first coupling interval should use backward Euler unless a physically consistent \(n-1\) state is already available. Simply duplicating the initial state is acceptable for small test problems but is not the preferred production startup when second-order temporal accuracy matters.

### 12.3 Boundary conditions

The reference solver supports:

- conducting wall: \(E_1(a)=0\);
- vacuum-region Robin approximation:

\[
\left.\frac{\partial E_1}{\partial r}\right|_a
=-\frac{E_1(a)}{a\ln(b/a)}.
\]

Additional resistive-wall treatments may be added later.

## 13. Nonlinear particle/plasma coupling

The nonlinear loop is

\[
E^{(k)}
\rightarrow f^{n+1,(k)}
\rightarrow j_{\rm kin}^{n+1,(k)}
\rightarrow E^{n+1,(k+1)}.
\]

The initial reference method is geometry-neutral Picard iteration with optional
under-relaxation. A backend supplies field construction, moment deposition, and
plasma solve callbacks. Every iteration re-advances particles from the same
`n`-level ensemble using current field guess, deposits moments, and solves
backend field equation. Circular RAMc uses radial BDF2 callback; slab uses
scalar field callback.

Convergence is measured from the relative field residual, not merely the number of iterations.

After the final field iterate is accepted, the particle block is run once more
under that field. The returned particle state and deposited kinetic current
therefore correspond to the returned electric-field profile, even when the
Picard loop stops at its iteration limit.

More sophisticated Newton/JFNK coupling can replace Picard if needed. A fully implicit nonlinear solve over every particle coordinate is not the intended architecture.

## 14. Safety-factor update

For the circular RAMc backend, the active legacy formula is implemented as a radial integral of Ohmic plus kinetic/bootstrap current followed by

\[
q(r)
=
\frac{r^2\epsilon}
{(4\pi/\bar\omega_{ce})\int_0^r dr'\,r'[E_1/\bar\eta+j_{\rm kin}+j_{bs}] }.
\]

The axis value is regularized with \(q(0)=q(r_1)\). Evolving `q` is optional because many validation problems prescribe it.

## 15. Electron/ion energy and charge-state evolution

The plasma energy equations are advanced at the coupling cadence. Given an estimate of new-level power density, BDF2 for an energy density \(U\) is

\[
U^{n+1}
=
\frac{4U^n-U^{n-1}+2\Delta t_c P^{n+1}}{3}.
\]

The new-level power may depend on \(T_e,T_i\), charge states, resistivity, radiation, and runaway moments, so it belongs inside the same outer multiphysics iteration when those feedbacks are enabled.

For charge states,

\[
\frac{d\mathbf n_Z}{dt}=A(T_e,n_e)\mathbf n_Z,
\]

and JONTA solves the linear new-level rate matrix implicitly with backward Euler/BDF2. Species density is renormalized after small positivity corrections.

OPEN-ADAS data are preprocessed into tables before the simulation and interpolated inside JAX kernels.

## 16. RNG strategy

The reference stochastic kernels derive random streams from:

- base seed/key;
- global particle-step index;
- operator stream identifier;
- stable particle slot ID.

This avoids dependence on device-local execution order and is intended to remain reproducible when the particle array is re-sharded. RNG throughput should be benchmarked on production GPUs; a lower-cost counter mapping may replace the reference per-particle key construction if it preserves the same invariants.

## 17. JAX compilation strategy

The expensive inner evolution is built as a fixed-length `jax.lax.scan` and JIT compiled. Model choices such as orbit backend and integrator are made when constructing the block, avoiding runtime Python polymorphism inside the GPU kernel.

The recommended execution pattern is:

- compile tens/hundreds of particle timesteps per block;
- keep particle/field arrays device resident;
- perform reduced diagnostics/deposition on device;
- synchronize to the host only at coupling/output points where required.

## 18. Multi-GPU numerics

Particles are sharded on their leading dimension. Compact field and plasma profiles are replicated. Local particle moments are reduced across the mesh before the plasma solve.

Coupling adapters can set `require_partitioned=True`; this rejects particle
blocks that flatten device outputs before deposition. Use
`build_particle_block(..., preserve_partitioning=True)` when enforcing local
device moment reduction.

The first portable multi-device correctness slice is deterministic guiding-center
evolution. It can be exercised on CPU device emulation before running on CUDA:

```bash
XLA_FLAGS=--xla_force_host_platform_device_count=2 \
  JAX_PLATFORMS=cpu pytest -q tests/integration/test_parallel_orbit.py
```

The test compares a one-device reference with a two- or four-device particle
partition and requires the orbit state to agree to floating-point tolerance.
This validates the particle decomposition without conflating it with global
population control.

Production particle blocks expose the same distinction through
`ExecutionConfig(mode="serial"|"parallel", platform="auto"|"cpu"|"gpu")`.
The serial block is the trusted reference. The parallel block uses `pmap` over
the leading marker axis and replicates compact field/background state.
Large-angle and source population control are local to each particle partition;
no distributed global resampler is required by the intended particle/plasma
coupling. Device-local capacity and overflow frequency must be included in
statistical convergence tests.

## 19. Current numerical limitations / optimization targets

The current code is a reference implementation. Important optimization/validation targets are:

- replace/benchmark per-particle RNG key folding if it becomes expensive;
- optimize Møller energy inversion if fixed bisection dominates large-angle cost;
- determine the lowest-cost converged deterministic integrator;
- establish acceptable large-angle and coupling cadences;
- validate the conservative avalanche model against RAMc and published rates;
- connect the sharded moment-reduction helper to the production multi-device
  plasma driver;
- integrate preprocessed OPEN-ADAS tables into the full nonlinear plasma loop.
