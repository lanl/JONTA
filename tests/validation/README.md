# Validation tests

Timestep, marker-number, grid-resolution, collision-cadence, coupling-cadence,
and statistical validation tests belong here. Coding tests live under
`tests/unit/` and `tests/integration/`; paper/analytical benchmark drivers live
under `benchmarks/`. See `docs/validation.md` for the acceptance hierarchy.

## Guiding-center invariant conservation

`test_guiding_center_invariants.py` is the primary deterministic-orbit convergence benchmark. It disables collisions, radiation, sources, and plasma feedback and checks both axisymmetric guiding-center invariants in the same particle integration:

- toroidal canonical momentum `P_phi`, conserved because the equilibrium is axisymmetric;
- magnetic moment `mu`, the guiding-center adiabatic invariant.

The command-line driver has a documented full validation grid, not a smoke-test
grid:

- `E1/Ec = 0, 1, 10, 10^2, ..., 10^8`. Zero is a non-logarithmic anchor;
  positive fields cover eight decades and the largest fields are deliberate
  numerical stress tests, not representative disruption conditions.
- `dt/tau_c = 5.12e-7, 1.28e-7, 3.2e-8, 8e-9, 2e-9, 5e-10,
  1.25e-10, 1e-10`. This is a four-fold refinement sequence through
  `1.25e-10`, followed by a final `1e-10` floating-point-floor probe.

The default run uses `final_time=1e-5 tau_c`, eight particles, and both
midpoint/RK2 and RK4. It therefore contains 160 integrations; the smallest
requested step can require 100 million orbit steps per integration. Treat it
as a scheduled validation run on the target backend, not a pull-request smoke
test. The CSV records the actual adjusted timestep and a `finite` flag, while
the companion JSON records the exact requested grid, backend, devices, and
profile.

The time integration is inside a JIT-compiled `jax.lax.while_loop`, so the same
test runs on CPU or GPU without Python timestep loops. This does not by itself
make the CLI multi-device: the CPU `pmap` test validates deterministic particle
partition/merge separately.

Both invariant errors are accumulated during the same orbit push, avoiding duplicate integration work. The generated RK2/RK4 figures contain separate panels for `P_phi` and `mu` together with the expected `dt^2` and `dt^4` reference slopes.

```bash
PYTHONPATH=src python tests/validation/test_guiding_center_invariants.py \
  --output-dir convergence_results
```

That command is the full default scan. For a bounded CPU preview, override
*both* axes explicitly and label the output as a preview; for example:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python tests/validation/test_guiding_center_invariants.py \
  --electric-fields 0 10 10000 \
  --dts 2.56e-7 6.4e-8 1.6e-8 \
  --final-time 1e-5 --n-particles 8 \
  --output-dir benchmark_results/adiabatic_invariants_cpu_preview
```

The preview is useful for checking execution and plotting only; it is not
evidence that the full electric-field/timestep acceptance grid has converged.

Use `--ramc-q-profile` to replace the default constant-`q` integration-isolation case with the RAMc-like profile `q(r)=2.1+2r^2`; at sufficiently small timestep this also exposes the independent field/profile-interpolation error floor.

## Small-angle Maxwellian relaxation

`test_small_angle_maxwellian_relaxation.py` validates the complete default
small-angle test-particle collision operator (pitch scattering + friction +
energy diffusion).  With electric field, radiation, large-angle collisions,
and sources disabled, arbitrary marker distributions must relax to the fixed
background Maxwellian.

The default scan uses `Te = 100 eV, 1 keV, 10 keV` and three deliberately
different initial conditions: a broad uniform-energy population, a narrow
anisotropic hot beam, and a cold/hot bimodal counter-directed population.  The
collision step and final time are scaled with `(vTe/c)^3`, allowing the same
fixed-step resolution of the thermal collision time at every temperature.

The hot collision loop is one JIT-compiled `jax.lax.fori_loop`; there are no
Python particle or timestep loops, and the kernel is directly GPU-ready.  The
validation checks the energy CDF against the isotropic Maxwellian and verifies
`<xi> -> 0` and `<xi^2> -> 1/3`.

```bash
PYTHONPATH=src python tests/validation/test_small_angle_maxwellian_relaxation.py \
  --output-dir convergence_results
```

## Axisymmetric spatial transport

`test_spatial_transport.py` implements the McDevitt, Guo & Tang (PPCF 61,
024004, 2019) radial-transport estimator and now treats the fully ionized and
partially screened cases as one benchmark family.  With electric field,
synchrotron, friction, energy diffusion, and large-angle collisions disabled,
monoenergetic isotropic ensembles evolve under the circular guiding-center
equations plus pitch-angle scattering.  Markers are binned by initial radius
and the code fits

- `V = d<Delta r>/dt`, and
- `D = 0.5 d(<Delta r^2>-<Delta r>^2)/dt`.

The published Figure 3 markers and visible vertical error-bar limits are
digitized into `benchmarks/reference_data/mcdevitt_2019_ppcf_fig3.csv`. Figure 3 uses the
fully ionized low-Z collision model.  The red dashed nonrelativistic estimate
is computed independently from McDevitt et al. Eq. (1); after normalization
by their Eq. (3), the pure fully ionized case reduces to `D_non-rel/D0 = c/v`.

Figure 6 is digitized independently into
`benchmarks/reference_data/mcdevitt_2019_ppcf_fig6.csv`. It uses deuterium plus singly
ionized argon with `n_Ar+ = n_D/10` and the partially screened pitch-angle
coefficients.  JONTA uses the same RAMc/Hesslow screening fit (`Z0=18`,
`ZI=1`, `aI=0.329`, `k=5`) and computes the energy-dependent electron-ion and
electron-electron Coulomb logarithms.  The asymptotic scaling from Eqs. (7)-(8)
is also evaluated from `C_B = p^2 tau_c nu_D^ps`.  The normalization includes
the `tau_c^a/tau_c` correction because Eq. (3) defines `D0` from main-ion free
electrons whereas the kinetic equations use the total free-electron collision
time.

The driver now partitions each physical energy/radius ensemble into independent
marker sub-ensembles.  The central diffusivity estimate is formed from the
pooled ensemble, while the sub-ensemble spread supplies a Monte Carlo standard
error.  Validation plots therefore show **both** the digitized paper error bars
and JONTA sampling error bars, plus a pointwise residual plot.  Figures are
written at 360 dpi.

The diffusion coefficient is accepted only from the post-transient linear
regime of the radial variance
`<Delta r^2>-<Delta r>^2`.  The driver writes the complete variance history,
its Monte Carlo SEM, the exact least-squares line used for the diffusion fit,
and the corresponding `R^2`.  It produces a mid-radius overview plus one
all-radii variance-linearity figure per energy, so every reported Figure-3
point can be inspected for a genuine diffusive interval rather than inferred
from a short-time orbit-width transient.  A separate fit-quality plot shows
`R^2(r)` for all energies.

The default command-line mode remains a reduced CPU demonstration and labels
itself as such.  By default it runs **both** collision models.  Use
`--case fully-ionized` or `--case partial-screening` to select one branch.
`--paper` uses the published parameters (`c*tau_c/a=5e6`, 20 collision times,
10 keV, 100 keV, 500 keV, and 1.5 MeV) together with the digitized radii and is
intended for GPU execution.  The default paper configuration uses eight
independent sub-ensembles per point.  Each case writes raw JONTA output and a
point-by-point comparison CSV containing JONTA SEM, paper error-bar limits,
relative error, and a combined-error z score.  A short CPU run is **not**
accepted as a passed reproduction merely because its trend is qualitatively
similar.

## Runaway vortex / bump-on-tail

`benchmarks/test_runaway_vortex.py` reproduces the 0D-2V kinetic benchmark of Guo,
McDevitt & Tang (PPCF **59**, 044003, 2017). Large-angle collisions and
external sources are off; the retained physics are electric-field acceleration,
collisional drag, energy diffusion, pitch-angle scattering, and synchrotron
radiation. The canonical parameters are `vTe/c=0.1`, `Z=1`, and
`alpha=tau_c/tau_s=0.2`.

The reference set is restricted to the paper's numerical data and analytical
relations. Stored numerical targets include the Figure-9 bump locations
`p=6.0551` at `E/Ec=2.25`, `p=8.59096667` at `E/Ec=2.5`, and the reported
O-X disappearance near `1.85 Ec`. Analytical comparisons use Guo Eqs. (16)
and (22)--(25).

The test now has two layers. Fast checks compare the production deterministic
and stochastic coefficients directly with Guo Eqs. (2)--(5), including a
JIT-compiled large-marker one-step test of Monte-Carlo means and variances. A
distribution-level validation evolves a fixed-size equal-weight marker ensemble
through the full 0D kinetic system with a single JIT-compiled JAX loop. It
time-averages the particle distribution after burn-in and verifies: (1) the
pitch-integrated tail remains monotone at `E/Ec=2`; (2) bumps form at
`E/Ec=2.25` and `2.5`; and (3) their locations agree with the published Figure-9
values within the statistical resolution of the CPU-size test.

For the momentum window, the particle benchmark uses `p_min=3 vTe/c=0.3`,
matching Guo's lower kinetic boundary, and a reflecting zero-flux boundary at
`p_max=20`. Markers reaching the unresolved low-energy side are returned to
`p_min`, preserving fixed array size and avoiding a thermal-scale timestep.
The high-statistics `--paper` mode increases marker count, duration, momentum
resolution, and halves the timestep for execution on the target GPUs.

The command-line driver also reconstructs the phase-space probability current
from the particle-sampled distribution for visualization of the runaway vortex;
this diagnostic never enters the particle evolution.

## Large-angle collisions / avalanche

`benchmarks/test_large_angle_avalanche.py` validates the production Møller operator and
the conservative mixed Fokker--Planck--Boltzmann avalanche model against
McDevitt, Guo & Tang (PPCF **61**, 054008, 2019).  All distribution-level
results are produced by the JONTA particle Monte Carlo solver; there is no
auxiliary continuum kinetic solver in this benchmark.

The event-level tests first verify the Møller kernel itself: the analytic tail
cross section is compared with numerical integration of the differential cross
section, sampled outgoing pairs satisfy cold-target energy and momentum
kinematics, and the weighted conservative gain-loss candidate set preserves
kinetic energy to roundoff before thinning.  The production fixed-`N` operator
then uses stratified random thinning/resampling.  This keeps the design choice
of fixed marker count while reducing the variance of repeated avalanche
branching relative to ordinary multinomial thinning.

The paper-level benchmark family contains four complementary checks:

- **Appendix Fig. B3(a):** slab avalanche growth rate for `alpha=0.5`,
  `Zeff=2`, and `ln Lambda=20`, compared directly with digitized Monte Carlo
  markers from the paper.
- **Appendix Fig. B3(b):** the zero-growth avalanche threshold is compared
  with Eq. (B15),
  `Eav/Ec = 1 + 1.0906 [alpha^0.6 (Zeff+1)]^0.6801`.
- **Fig. 13:** the conservative growth rate is scanned versus
  `gamma_min^LA-1` at fixed electric field.  A broad plateau below the X-point
  verifies that the physical answer is insensitive to the arbitrary
  small-/large-angle partition when the two pieces are matched consistently.
- **Fig. 14:** the conservative gain-loss formulation using Eq. (32) for the
  electron-electron Fokker--Planck remainder is compared with the conventional
  source-only model using Eq. (33).  The two growth rates should remain close,
  with the conservative model tending slightly lower.

Only the **electron-electron** collision operator is split between the
Fokker--Planck and Møller pieces.  Electron-ion pitch scattering retains its
full relativistic electron-ion Coulomb logarithm.  This distinction is covered
by a unit test because applying the reduced large-angle logarithm to the ion
term gives a noticeably incorrect Fig.-14 comparison.

The hot loop is fixed-shape JAX: small-angle collisions, deterministic RK4
electric/synchrotron evolution, the large-angle macro-step, absorbing low-energy
boundary, fixed-`N` thinning, and population-history accumulation all execute
inside JIT-compiled control flow.  The large-angle cadence remains a separate
macro-step so it can be refined independently of the orbit/small-angle step.

```bash
PYTHONPATH=src python -m benchmarks.test_large_angle_avalanche \
  --case b3 --output-dir large_angle_results
```

Use `--case threshold`, `--case cutoff`, or `--case compare` to run the other
families independently.  `--paper` increases marker count, duration, number of
random replicates, and timestep resolution for GPU execution.  Running one
family at a time is also useful on CPU because long sequential JAX benchmark
jobs can retain substantial compiled/runtime state.

The broader McDevitt-2019 figure-reproduction driver is
`benchmarks/test_mcdevitt_2019_avalanche_figures.py`. It intentionally accepts **exactly
one** `--figure` argument per process; the previous `--figure all` path has been
removed so long GPU/CPU validation jobs are checkpointed figure by figure.
Use, for example,

```bash
PYTHONPATH=src python -m benchmarks.test_mcdevitt_2019_avalanche_figures \
  --figure 3 --output-dir mcdevitt_fig3
```

where `--paper` selects the high-statistics GPU configuration.  Monte-Carlo
standard errors are plotted wherever the observable is statistical: avalanche
growth rates, interpolated thresholds, inferred `psi_10`, and particle spectra.
Threshold error bars are obtained by propagating the two bracketing growth-rate
SEMs through the linear zero-crossing interpolation.  Figure 2 has no reduced
default: its claim is that a runaway ring remains radially localized, so a
short CPU run is not a valid smoke test.  It can only be requested explicitly
with `--paper`, and the driver writes no PNG unless the initial/final radial
localization acceptance gate passes.  Figures 5/6 are configuration/phase-flow
visualizations, so scalar error bars are not applicable.
Appendix Fig. B4 is included in this figure-by-figure driver using the production
partial-screening drag/pitch operator and bound-electron Moller target factor.


## Fixed-N statistical validation

`test_fixed_n_statistics.py` validates the fixed-marker branching design before
expensive GPU figure reproductions.  It uses independent replica ensembles, a
manufactured linear branching problem with a known growth eigenvalue, marker
scans for the expected approximately `N^-1/2` statistical scaling, and a direct
McDevitt Fig. B3(a) marker-count/macrostep convergence check.  Stochastic error
bars are replica SEMs rather than errors inferred by treating post-resampling
markers as independent.

### Analytic avalanche warm start

`test_avalanche_warm_start.py` tests an optional initialization acceleration for
well-above-marginal avalanche calculations.  It does **not** solve an auxiliary
continuum problem and does not modify the kinetic operator.  The initial marker
cloud is constructed from

- the Rosenbluth--Putvinski high-field growth coefficient, which gives the
  exponentially decaying energy scale used in McDevitt Appendix Fig. B2;
- the Guo--McDevitt--Tang analytic synchrotron O-point estimate, used only to
  cap that energy scale when radiation limits the accessible energy;
- a momentum-dependent pitch profile projected onto a finite Legendre basis.

The regression target is McDevitt Appendix Fig. B3(a) at `E/Ec=3.2321`,
`alpha=0.5`, `Zeff=2`, `ln Lambda=20`.  The deliberately short CPU test uses
only `6 tau_c` and begins the growth fit after 5% of the history.  Independent
replicas compare the analytic warm start with the legacy uniform energetic RE
seed.  The warm start is considered useful only when it reduces transient bias
without changing the asymptotic published growth rate.

This approximation is intended for **well-above-marginal** growth-rate scans.
Near the avalanche threshold the Rosenbluth high-field form is a poor
approximation; threshold benchmarks retain long integrations and conservative
burn-in rather than relying on this shortcut.
