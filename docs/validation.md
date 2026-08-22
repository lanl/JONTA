# JONTA Validation Plan

## 1. Purpose

JONTA is a physics code. Passing unit tests is necessary but not sufficient. This document defines the hierarchy used to establish formula correctness, numerical convergence, statistical correctness, compatibility with RAMc, and GPU performance.

The supplied `RAMc-doc.pdf` is the primary legacy reference for the initial guiding-center, collision, deposition, and 1-D field models. Published benchmarks cited there provide independent targets.

## 2. Validation philosophy

JONTA is not required to reproduce the exact stochastic trajectory or adaptive-timestep history of RAMc. It must reproduce converged physical observables within statistical/numerical uncertainty.

Validation should separate four error sources:

1. deterministic discretization error;
2. Monte Carlo sampling error;
3. spatial/profile discretization error;
4. model differences intentionally introduced in JONTA.

A physical model change must not be disguised as a numerical discrepancy.

## 3. Level 0: local/unit identities

These tests should run quickly on CPU and every pull request.

### Orbit algebra

- `E=0`, radiation off: deterministic magnetic field does no work, `dgamma/dt=0`.
- Uniform 0-D field: sign of acceleration follows RAMc convention.
- Circular field samples remain finite near the magnetic axis.
- Synchrotron terms vanish for `|xi|=1`.

### Small-angle operator

- `|xi|<=1` after reflection.
- `gamma>=1` after low-energy reflection.
- all-disabled collision configuration is identity.
- Chandrasekhar function is finite as `x->0`.

### Møller operator

For every sampled cold-target event:

\[
(\gamma_3-1)+(\gamma_4-1)=\gamma_0-1.
\]

Parallel momentum reconstructed from the two outgoing particles must equal incoming parallel momentum to roundoff.

Before capacity control, weighted gain-loss candidates must conserve kinetic energy and increase kinetic particle weight by exactly the transferred background-electron weight.

### Resampling

- total represented weight is preserved exactly;
- equal weights give `N_eff` equal to active-marker count;
- repeated resampling of a known distribution is unbiased within Monte Carlo confidence intervals.

### Plasma solvers

- zero field/current source remains zero under BDF2;
- charge-state matrix conserves total species density;
- no-rate charge-state system is stationary;
- thermal-energy step reproduces constant-power analytic solutions.

## 4. Level 1: deterministic orbit convergence

### 4.1 Guiding-center invariant conservation

RAMc Sec. III.A / Fig. 9 verifies magnetic moment and toroidal canonical momentum in an axisymmetric system with collisions and synchrotron radiation disabled. JONTA treats them as one guiding-center invariant conservation test and measures both during the same orbit integration.

Reference parameter family:

- `c*tau_c/a ~ 2.3e6`;
- `a/R0=1/3`;
- `E1/Ec=10`;
- `Te=3.1 keV`;
- `q(r)=2.1+2 r^2`;
- `B0=3 T`;
- `a=100 cm`.

JONTA should repeat the test using a timestep sequence

\[
\Delta t,\;\Delta t/2,\;\Delta t/4,\ldots
\]

for each candidate integrator. Error should converge at the expected order until interpolation/roundoff error dominates.

The absolute RAMc adaptive-solver tolerances are not JONTA's acceptance criterion; invariant convergence with fixed timestep is.

The production CLI makes the requested scan explicit. Its default electric-
field grid is `E1/Ec = 0, 1, 10, 10^2, ..., 10^8` (zero is a non-log anchor),
and its timestep grid is
`5.12e-7, 1.28e-7, 3.2e-8, 8e-9, 2e-9, 5e-10, 1.25e-10, 1e-10` in units of
`tau_c`. The positive-field grid spans decades; the timestep grid uses
four-fold refinement followed by a final floating-point-floor probe. Both
midpoint/RK2 and RK4 are run at every grid point, with `final_time=1e-5 tau_c`
and eight particles by default. The output directory contains the CSV, three
plots, and a JSON manifest of the exact requested grid and runtime backend.

This full grid is intentionally expensive: the final timestep can imply
`1e8` orbit steps for one integration. A reduced CPU command must override
both `--electric-fields` and `--dts`; its plots are execution previews, not
acceptance evidence for the full scan. Multi-device CPU correctness is covered
by the separate deterministic sharding test; the invariant CLI is currently a
single-device benchmark.

### 4.2 Trapped/passing orbits

RAMc Sec. IV.A / Fig. 23 provides qualitative orbit topology for `gamma~10` and `gamma~98`. JONTA should reproduce:

- increasing banana width with trapped-particle energy;
- outward displacement/asymmetry of high-energy passing orbits;
- correct confinement boundary behavior.

### 4.3 Ware pinch

RAMc Sec. III.H / Fig. 21 gives the trapped-electron flux drift

\[
\frac{\Delta\psi_0}{\Delta t}=-R_0E_1c.
\]

The normalized numerical drift should converge to the analytic value. This is a sensitive sign and electric-field coupling test.

### 4.4 Passing-electron drift

RAMc Fig. 22 shows a much weaker finite-aspect-ratio outward drift for a freely accelerated passing electron. The mean drift should have the correct sign and scale and decrease appropriately in limits where finite-aspect-ratio effects vanish.

## 5. Level 2: small-angle collision validation

### 5.1 Maxwellian relaxation

RAMc Sec. III.B / Fig. 10 initializes particles away from equilibrium and demonstrates relaxation to a Maxwellian when electric field, radiation, and large-angle collisions are disabled.

Validation requirements:

- pitch-averaged distribution approaches the Maxwellian reference;
- arbitrary initial energy/pitch distributions converge to the same equilibrium;
- pitch becomes isotropic, with `\langle\xi\rangle -> 0` and `\langle\xi^2\rangle -> 1/3`;
- mean histogram/distribution error scales approximately

\[
\langle\Delta f\rangle\propto N^{-1/2};
\]

- results converge with particle timestep independently of marker-number convergence.

The JONTA temperature/initial-condition scan uses `Te=100 eV, 1 keV, 10 keV` and three deliberately non-equilibrium initial populations (broad uniform energy, hot anisotropic beam, and cold/hot bimodal).  The fixed collision timestep and relaxation interval are scaled with `(vTe/c)^3` so each temperature resolves the same thermal collision timescale.  RAMc Fig. 10 provides the legacy `Te=10 keV` and `100 eV` marker-number scaling target, with particle counts spanning roughly `2e2` to `5e4`.

### 5.2 Neoclassical transport

RAMc Sec. III.C / Fig. 11 benchmarks the banana-regime diffusivity against

\[
\frac{\tau_cD_t}{a^4B_0^2}
=0.689\sqrt{2\epsilon}\,q^2\frac{R_0^2}{a^2}
\frac{E_K}{m_ec^2}
\left(\frac{c}{a\omega_{ce}}\right)^2
(\tau_c\nu_D).
\]

Reference parameters include

- `ne=5e12 cm^-3`;
- `gamma=1.02`;
- `a/R0=1/3`;
- `E1=0`;
- synchrotron off;
- `Te=1 keV`;
- `q0=2.1`, `q2=2`;
- `B0=5.3 T`, `a=200 cm`;
- scans in `Zeff=1,3,5`.

Agreement should be tested only where `nu_* << 1`.

### 5.3 Runaway vortex / bump-on-tail

Guo, McDevitt & Tang (2017) provide a full 0D kinetic benchmark with electric
acceleration, test-particle small-angle collisions, and synchrotron radiation,
while excluding large-angle collisions. This is the first JONTA validation that
exercises all of those terms simultaneously.

The canonical case uses `vTe/c=0.1`, `Z=1`, and `alpha=0.2`. The published
Figure-9 numerical targets are a monotone pitch-integrated tail at `E/Ec=2`, a
bump at `p_b=6.0551` for `E/Ec=2.25`, and a bump at `p_b=8.59096667` for
`E/Ec=2.5`. Guo also reports disappearance of the O-X vortex near
`E/Ec ~= 1.85`.

Analytical reference relations include

\[
p_O=\sqrt{2}\,\frac{(E+\alpha)(E-1)}{(1+Z)\alpha},
\qquad
p_X=\sqrt{\frac{1+(1+Z)/(2\sqrt{2})}{E}},
\]

with the empirical pitch-integrated bump estimate

\[
p_b \simeq p_O/1.55,
\]

and the runaway-tail spread estimate `(p_O-p_X)/1.8`. Guo Eq. (16) supplies
the large-p acceleration-channel width used as an additional coefficient check.

The distribution-level acceptance test is generated entirely by the production
marker Monte-Carlo algorithm. A broad fixed-capacity ensemble is used only to shorten
burn-in; the steady distribution is time averaged after burn-in. The kinetic
hot loop uses Strang splitting between the production small-angle operator and
RK4 electric/synchrotron dynamics and is compiled as one JAX kernel. The test
uses a lower momentum reservoir at `p_min=3 vTe/c` and a reflecting high-energy
zero-flux boundary, consistent with resolving only the nonthermal kinetic
window.

Acceptance requires the no-bump/bump transition of Figure 9 and direct
agreement of the recovered bump locations with the published numerical values.
A separate diagnostic reconstructs the phase-space probability current from the
Monte-Carlo distribution so the local runaway circulation can be inspected.
The only comparison references are the published numerical data and Guo's
analytical formulas.

## 6. Level 3: Dreicer production

RAMc Sec. III.D / Figs. 12-14 compares Monte Carlo Dreicer generation against Kulsrud et al., Kruskal-Bernstein/Connor-Hastie asymptotics, and earlier Monte Carlo work.

Initial benchmark family:

- `Te=50 eV`;
- near-axis initialization;
- `ne~5e13 cm^-3`;
- `E/ED ~ 0.04-0.1`;
- `Zeff=1,2,3,10` cases.

Acceptance must account for the finite thermal reservoir and the ambiguity of fitting production rates at very large `E/ED`. JONTA should report the fit interval and statistical confidence rather than a single rate without uncertainty.

## 7. Level 4: avalanche validation

This is especially important because JONTA intentionally replaces RAMc's source-only branching implementation with conservative gain-loss physics.

### 7.1 Source-limit recovery

In the relativistic, weak-background-depletion limit, the new operator should reproduce the conventional Møller/RAMc secondary source growth rate within statistical and cutoff errors.  The implemented slab benchmark uses Appendix Fig. B3(a) of McDevitt, Guo & Tang (PPCF 61, 054008, 2019): `alpha=0.5`, `Zeff=2`, and a constant `ln Lambda=20`.  The published Monte-Carlo markers are digitized in `benchmarks/slab/avalanche_decay/reference/mcdevitt_2019_figB3_B13.csv` and are compared directly with JONTA particle results.

The corresponding threshold benchmark uses the zero crossing of the fitted exponential population growth and compares with McDevitt Eq. (B15),

\[
\frac{E_{av}}{E_c}=1+1.0906\left[\alpha^{0.6}(Z_{eff}+1)\right]^{0.6801}.
\]

The paper reports excellent agreement for `1/alpha > 10`, with the fit tending to slightly overestimate the Monte-Carlo threshold for smaller `1/alpha`.  JONTA therefore records the sign as well as the magnitude of the threshold residual.

RAMc Sec. III.E / Fig. 15 gives a fully ionized benchmark with approximately:

- `a/R0=1/3`;
- `alpha=0.1`;
- `vTe/c=0.1`;
- `q0=2.1`, `q2=2`;
- ITER-like `a=200 cm`, `B=5.3 T`;
- near-axis particles to suppress trapping effects.

### 7.2 Conservative FP/Boltzmann partition

The production conservative model follows the mixed Fokker--Planck--Boltzmann construction of McDevitt Sec. 5.  The split applies to the **electron-electron** operator only.  Electron-ion pitch scattering remains wholly in the small-angle operator and must retain the full relativistic electron-ion Coulomb logarithm.

For the conservative model, the electron-electron Fokker--Planck remainder uses Eq. (32),

\[
\ln\Lambda_{\min}^{LA}
=\ln\Lambda_0
+\ln\left[2\frac{c}{v_{Te}}\sqrt{\gamma_{\min}^{LA}-1}\right],
\]

which accounts for both the relativistic minimum impact parameter and removal of collisions assigned to the Møller operator.  The conventional source-only comparison uses Eq. (33),

\[
\ln\Lambda_{ee}^{RE}
=\ln\Lambda_0
+\ln\left[\frac{c}{v_{Te}}\sqrt{2(\gamma-1)}\right].
\]

McDevitt Fig. 13 provides the decisive cutoff-invariance test.  For `Z=1`, `alpha=0.1`, `vTe/c=0.1`, `ln Lambda_0=15`, and `E/Ec=2.05, 2.25, 2.5, 3`, the growth rate should remain approximately flat over a broad interval of `gamma_min^LA-1` below the X-point and then fall once the cutoff approaches/exceeds the runaway separatrix energy.  JONTA stores digitized Fig.-13 values for a pointwise comparison, in addition to testing the plateau itself.

McDevitt Fig. 14 then compares the high-fidelity conservative model against the source-only model at `alpha=0.1`, `gamma_min^LA=1.02`, `Zeff=1`, `ln Lambda_0=15`, and `vTe/c=0.1`.  Their avalanche growth rates and thresholds are reported to be remarkably similar; the conservative result is slightly lower in this formulation because primary down-scattering is substantially offset by the reduced Fokker--Planck collision strength.  JONTA treats this near-equivalence as an end-to-end consistency check, not as justification for replacing the conservative production operator.

### 7.3 Partially ionized comparison

RAMc Fig. 16 compares against Hesslow et al./CODE for a weakly ionized impurity case. That benchmark should be enabled when the corresponding screened small-/large-angle collision model is implemented.

McDevitt Appendix Fig. B4 provides a second partially ionized avalanche-threshold target.  It remains deferred until the complete screened drag and large-angle avalanche coefficients are present.  The partially screened pitch-angle operator by itself, although already validated in the spatial-transport benchmark, is not sufficient for this test.

### 7.4 Conservative-event tests

Independent of growth-rate agreement:

- the analytic Møller tail cross section must agree with direct quadrature of the differential kernel;
- sampled cold-target pairs must conserve kinetic energy and close the full momentum triangle to roundoff;
- pre-resampling weighted gain-loss energy conservation must be roundoff-level;
- gain/loss weight accounting must be exact;
- growth rate must converge as `dt_LA` decreases;
- results must converge as the large-angle energy cutoff and small-angle/large-angle partition are refined consistently;
- results must converge with marker count as `N^-1/2` where ordinary Monte Carlo statistics apply.

JONTA uses fixed marker count and random thinning/resampling after the gain-loss step.  Stratified random resampling is preferred to global multinomial resampling because it preserves the same unbiased fixed-`N` interpretation while greatly reducing branching noise.  The population weight, rather than the literal array length, carries avalanche amplification.

### 7.4.1 Fixed-N statistical acceptance

The statistical implementation is accepted only if all of the following are
satisfied:

- direct stratified-resampling tests recover known weighted moments without a
  statistically significant bias;
- a manufactured two-type linear branching process recovers its exact
  dominant-eigenvalue growth rate;
- the replica-to-replica standard deviation decreases approximately as
  \(N^{-1/2}\) under marker-count refinement;
- a representative McDevitt Appendix Fig. B3(a) growth-rate point converges to
  the digitized published value under the same marker-count scan;
- the same Fig. B3 point is invariant, within independent-replica uncertainty,
  when the large-angle macrostep is refined.  The largest converged interval
  is preferred thereafter to avoid unnecessary resampling noise;
- reported stochastic error bars use independent-replica SEMs.  Individual
  markers inside one resampled ensemble are not counted as independent samples.

These CPU-sized tests are intended to validate the statistical algorithm.  The
larger GPU benchmark uses exactly the same production kernel and estimator with
more markers and replicas, so increased GPU scale should primarily reduce the
reported uncertainty rather than change the numerical method.

### 7.5 Toroidal trapping / collisionality

RAMc Sec. III.F / Fig. 17 shows strong collisionality dependence of avalanche growth away from the magnetic axis. The circular 1-D backend should reproduce the asymptotic behavior versus `c*tau_c/a` and radius.

## 8. Level 5: radial deposition and Ohm coupling

### 8.1 Deposition conservation/statistics

For manufactured radial marker distributions:

- deposited total weight/current should converge under grid refinement;
- cloud-in-cell weights must sum to one away from boundary special cases;
- axis regularity `j(0)=j(r1)` must be enforced for the RAMc depositor.

### 8.2 Electric-field diffusion

Use manufactured solutions for

\[
\partial_tE=D L_rE
\]

with known boundary conditions. Verify second-order temporal convergence of BDF2 after startup and expected spatial convergence of the radial finite differences.

### 8.3 Current coupling

With prescribed `j_kin(t,r)` and `eta(t,r)`, compare the BDF2 solver against a high-resolution reference solution of the RAMc field equation.

### 8.4 Nonlinear particle/field loop

For coupled cases:

- Picard residual must decrease below the requested tolerance;
- solution must become independent of relaxation factor as tolerance tightens;
- coupling solution must converge as `dt_c` decreases;
- changing particle timestep at fixed coupling timestep must separately converge.

## 9. Level 6: safety-factor evolution

The circular safety-factor update should be checked against direct quadrature of RAMc Eq. (95)/the active `ComputeSafetyFactor` implementation. Manufactured current profiles with analytic integrals should be included.

## 10. Level 7: electron/ion energy and charge-state coupling

### Charge states

For constant rates, compare BDF2 charge-state evolution with a matrix-exponential reference. Verify total species density conservation and convergence under timestep refinement.

### OPEN-ADAS tables

For every imported table:

- compare interpolated coefficients at tabulated points exactly/within file precision;
- test monotonic/log interpolation behavior where expected;
- record the OPEN-ADAS dataset/version used by benchmark inputs.

### Energy equations

Use constant-power and linear electron-ion exchange problems with analytic solutions before enabling full radiation/ionization feedback.

## 11. Level 8: sources

### Tritium

Numerically integrate `S_T` over momentum and verify

\[
\int d^3p\,S_T
=(\ln2)n_T/\tau_T.
\]

Sampled beta spectra should match the analytic distribution by histogram/KS-type statistical tests.

### Compton

Compare kinetic source integrals against the fluid source calculations in Ekmark et al. for published photon spectra before using the source in production disruption calculations.

## 12. GPU/performance validation

Performance tests are part of acceptance because the project exists to exploit accelerators.

Record:

- deterministic RHS evaluations/s;
- full particle-steps/s;
- collision particle-steps/s;
- Møller operator throughput;
- deposition throughput;
- bytes of persistent particle state per marker;
- compile time separately from execution time;
- peak device memory;
- single-GPU and multi-GPU weak scaling.

Primary hardware:

- NVIDIA RTX PRO 6000 Blackwell;
- NVIDIA B200.

A benchmark result must identify GPU model, JAX/jaxlib version, CUDA stack, particle count, FP64 mode, integrator, and field/collision configuration.

## 13. Multi-GPU correctness

Changing device count must not change the expected physical/statistical solution. Tests should compare:

- total marker weight;
- deposited moments;
- ensemble distributions;
- growth/loss rates;
- random-stream reproducibility when the configured RNG scheme claims partition invariance.

Bitwise-identical floating-point reductions are not required because reduction ordering may differ across device meshes.

## 14. Current automated tests

The repository currently includes fast tests for:

- explicit integrators;
- 0-D force sign and stationary limits;
- circular orbit energy behavior and approximate magnetic-moment consistency;
- analytic/axisymmetric interpolation primitives;
- small-angle physical boundaries and the fixed-capacity momentum-reservoir boundary;
- Møller cross section, two-body event conservation, and gain-loss accounting;
- capacity compaction, overflow thinning, and source-injection weight preservation;
- radial binning weight conservation;
- charge-state density conservation;
- electron/ion energy sign conventions and constant-power BDF2 evolution;
- algebraic Ohm-law consistency and circular safety-factor manufactured behavior;
- BDF2 electric-field stationarity;
- tritium/Compton source sanity checks;
- JIT-compiled particle-block execution;
- Guo-2017 runaway-vortex coefficient/statistical checks and the particle-only bump-on-tail acceptance test.

These tests establish implementation sanity only. The RAMc/published validation levels above remain required before claiming production equivalence.

### Axisymmetric collisional spatial transport

The spatial-transport convergence/benchmark driver follows McDevitt, Guo &
Tang, *Plasma Physics and Controlled Fusion* **61**, 024004 (2019), Figs. 3
and 6 and Appendix A.  It disables electric acceleration, synchrotron
radiation, collisional drag/energy diffusion, and large-angle collisions,
retaining only circular guiding-center motion and pitch-angle scattering.
Radial convection and diffusion are extracted from first and second
displacement moments binned by each marker's initial radius.

The published Figure-3 marker values and visible error-bar limits are stored as
a checked-in digitization with provenance metadata.  Validation plots must
overlay JONTA and the digitized paper data on the same axes rather than compare
only qualitative trends.  JONTA must also report its own Monte Carlo uncertainty:
markers are partitioned into independent sub-ensembles, the central transport
coefficient is fit from the pooled population, and the sub-ensemble spread is
reported as a standard error.  A residual plot and combined-error z score are
used to expose statistically significant discrepancies.

Before comparing any fitted diffusivity with either Figure 3 or Figure 6, the underlying Appendix-A
transport diagnostic must visibly enter a diffusive regime.  For every
energy/radius point JONTA records and plots

\[
\sigma_r^2(t)=\langle\Delta r^2\rangle-\langle\Delta r\rangle^2,
\qquad
D=\frac{1}{2}\frac{d\sigma_r^2}{dt},
\]

including Monte Carlo uncertainty, the post-transient linear fit, and its
`R^2`.  The complete variance history is retained in the benchmark output.
Short-time finite-orbit-width oscillations must not be mistaken for diffusion;
the full-paper benchmark is considered credible only when the fitted interval
is demonstrably linear across the energy/radius scan.

McDevitt et al. Eq. (1) is also evaluated independently: using their
small-inverse-aspect-ratio trapped-fraction factor and normalizing by Eq. (3),
the fully ionized nonrelativistic estimate reduces to `D_non-rel/D0 = c/v`;
this is the dashed 10-keV analytic reference in Fig. 3.

Figure 6 is the corresponding partially screened benchmark.  Its reference
data are digitized separately, and the collision model uses singly ionized
argon at `n_Ar+/n_D=0.1` with the RAMc/Hesslow screening coefficients.  The
benchmark additionally evaluates the banana-regime Eqs. (7)-(8) scaling via
`C_B=p^2 tau_c nu_D^ps`, including the distinction between the total-electron
collision time `tau_c` and the main-ion normalization `tau_c^a` used in `D0`.
This second branch is important because partial screening enhances the
deflection rate strongly at relativistic energies and therefore gives a much
better resolved diffusion signal than the fully ionized MeV cases.

The full published parameter sets are exposed as a GPU-oriented `--paper`
mode.  Select exactly one figure per process so a long validation campaign is
checkpointed figure by figure. Short CPU runs are useful only for kernel
verification/statistical-development work and are not considered successful
paper-benchmark reproductions.

Figure 2 is stricter: it has no reduced/default execution mode. Its observable
is radial localization of a seeded runaway ring, so a short trajectory that
collapses or disperses the ring is a failed physics result, not a useful smoke
test. The driver therefore requires explicit `--paper` and writes no PNG unless
the initial/final radial-localization acceptance gate passes. Failed runs write
only the numerical diagnostics JSON.


## 15. McDevitt-2019 avalanche figure-reproduction workflow

The large-angle/toroidal validation target is a literal reproduction of
McDevitt, Guo & Tang, PPCF **61**, 054008 (2019), Figures 2, 3, 4, 5, 6, 7, 8,
10, 11, 13, 14, B2, B3 and B4.  Run **one figure per process**:

```bash
PYTHONPATH=src python -m benchmarks.test_mcdevitt_2019_avalanche_figures \
  --figure B3 --output-dir mcdevitt_B3 --paper
```

The command-line interface deliberately has no `all` option.  Statistical JONTA
observables carry Monte-Carlo error bars when applicable.  Growth-rate bars are
replicate SEMs; threshold bars propagate the bracketing growth-rate SEMs through
the zero-crossing interpolation; `psi_10` bars propagate the fitted avalanche-rate
uncertainty; weighted particle spectra use the effective weighted-count variance.
Paper curves/markers are given error bars only when the source figure supplies
them or a digitized uncertainty is available.

### Avalanche eigenmode warm-start validation

JONTA can optionally initialize a fixed-`N` avalanche ensemble with an analytic
approximation to the dominant runaway eigenmode.  This is an initialization
optimization only: all reported growth rates continue to come from the
production particle Monte Carlo operator.

The energy marginal uses the Rosenbluth--Putvinski scale

`T_R/(m_e c^2) ~= (E/Ec-1)/(gamma_av tau_c) = 1/(gamma0 tau_c)`

in the high-field limit.  When synchrotron losses are important, the warm-start
scale is limited by the analytic runaway-vortex O-point kinetic energy.  Pitch
is represented as `sum_l a_l(p) P_l(xi)`, with the coefficients obtained by
projecting a narrow field-aligned profile whose width varies with momentum.
This captures the important correlation that high-energy runaways are much
more strongly aligned than low-energy secondaries.

A short-run statistical test uses the digitized McDevitt Appendix Fig. B3(a)
point at `E/Ec=3.2321`.  With identical `N=512`, eight independent replicas,
and only `6 tau_c` of evolution, the legacy uniform energetic seed gives
`gamma_av tau_c = 0.03484 +/- 0.00123` (replica SEM), whereas the analytic
Rosenbluth--Legendre seed gives `0.02636 +/- 0.00112`.  The published value is
`0.027317`.  Thus the warm start places the short calculation within one SEM of
the published eigenvalue while the legacy seed retains a substantial transient
bias.  This does not remove the need for long near-threshold calculations: the
high-field approximation is deliberately not used as a hard threshold shortcut.
