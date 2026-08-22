# JONTA Conventions

## 1. Purpose

This document fixes normalization, units, coordinate/sign conventions, state layout, and naming. These conventions are part of the code contract; changing them requires coordinated code, documentation, and validation updates.

## 2. Floating-point precision

- Core physics dtype: FP64.
- Particle weights: FP64.
- Integer particle slot IDs: 64-bit integer where supported.
- Boolean activity masks: boolean.
- No silent FP32 conversion is allowed in the physics path.

## 3. Kinetic normalization

The initial 0-D and circular 1-D implementations retain RAMc's dimensionless kinetic normalization.

### 3.1 Time

\[
\bar t=t/\tau_{c0},
\]

with

\[
\tau_{c0}
=\frac{4\pi\epsilon_0^2m_e^2c^3}
{n_{e0}e^4\ln\Lambda_0}.
\]

Particle and large-angle timesteps are therefore normally specified in units of \(\tau_{c0}\).

### 3.2 Position

Poloidal-plane coordinates are normalized to minor radius:

\[
x=X/a,\qquad y=Y/a,\qquad r=\sqrt{x^2+y^2}.
\]

For the circular RAMc geometry,

\[
x=r\cos\theta,\qquad y=r\sin\theta.
\]

The major-radius factor used in the analytic RAMc equations is

\[
R/R_0=1+\epsilon x,
\qquad
\epsilon=a/R_0.
\]

`phi` is the toroidal angle in radians.

### 3.3 Momentum

Momentum is normalized to \(m_ec\):

\[
p=|\mathbf p|/(m_ec),
\qquad
\gamma=\sqrt{1+p^2},
\qquad
\xi=p_\parallel/p.
\]

The persistent particle state stores `gamma` rather than `p`.

### 3.4 Electric field

\[
\bar E=E/E_{c0},
\qquad
E_{c0}=m_ec/(e\tau_{c0}).
\]

The circular field stores `e1(r)` such that

\[
E_\phi=E_1(r)/(R/R_0)
\]

in normalized units.

### 3.5 Magnetic field

\[
\bar B=B/B_0.
\]

The dimensionless cyclotron parameter stored as `a_omega_ce_over_c` is

\[
\bar\omega_{ce}=a\omega_{ce0}/c.
\]

The orbit equations frequently use its inverse \(c/(a\omega_{ce0})\).

### 3.6 Parallel transit parameter

`c_tau_over_a` denotes

\[
\frac{c\tau_{c0}}{a}.
\]

This can be large (often \(10^5\)-\(10^6\)) and sets the separation between orbit and collision timescales.

### 3.7 Synchrotron parameter

`alpha_syn` denotes the RAMc parameter

\[
\alpha=\tau_{c0}/\tau_s.
\]

The local radiation strength includes the appropriate \((B/B_0)^2\) factor.

## 4. Background-plasma units

The kinetic normalization and background profile units are intentionally explicit rather than inferred from array names.

The initial RAMc-compatible collision modules use:

- `ne_cm3`: cm\(^{-3}\);
- `te_ev`, `ti_ev`: eV;
- `zeff`: dimensionless;
- Coulomb logarithm: dimensionless.

This matches the supplied RAMc profile/collision formulas and avoids hidden conversion during validation.

New plasma modules that use SI must include unit suffixes in variable names.

## 5. Field-equation normalization

The normalized current used by the circular RAMc field solver is

\[
\bar j=a^2j/I_A,
\]

where \(I_A\) is the Alfvén current.

The normalized resistivity is

\[
\bar\eta=\frac{\eta I_A}{E_{c0}a^2}.
\]

The radial electric-field equation is written in the corresponding RAMc dimensionless variables. `deposit_ramc_parallel_current` and `ramc_spitzer_eta_bar` implement these conventions explicitly.

## 6. Atomic/source/energy units

Modules outside the normalized orbit core use explicit physical-unit suffixes.

- Tritium density: m\(^{-3}\).
- Tritium source rate: m\(^{-3}\) s\(^{-1}\).
- Photon energy: eV.
- Differential gamma flux: named with its expected m\(^{-2}\) s\(^{-1}\) eV\(^{-1}\) convention.
- Atomic ionization/recombination coefficients: m\(^3\) s\(^{-1}\).
- Thermal/radiative powers: W m\(^{-3}\).
- Thermal energy density: J m\(^{-3}\).

Do not pass a normalized quantity to an SI-named interface without an explicit conversion.

## 7. Electron charge and pitch sign

JONTA preserves the RAMc electron sign convention. The symbol `e` in derivations denotes the positive magnitude of electron charge while the particle charge is \(-e\).

For the 0-D normalized equations,

\[
\dot\gamma=-v\xi E_\parallel+\cdots.
\]

Therefore a positive `E_parallel` accelerates an electron with negative pitch `xi < 0`. Current deposition includes the negative electron charge, so such electrons contribute positive conventional current in the legacy RAMc convention.

Do not change this sign convention locally.

## 8. Particle arrays

All persistent particle fields have the same leading dimension `N`:

```text
gamma  (N,) real_dtype
xi     (N,) real_dtype
x      (N,) real_dtype
y      (N,) real_dtype
phi    (N,) real_dtype
weight (N,) real_dtype
alive  (N,) bool
pid    (N,) index_dtype
```

`real_dtype` defaults to FP64 and may be selected as FP32 at process startup
with `JONTA_PRECISION=32` or through `core.precision.configure_precision`.
FP64 remains reference precision; FP32 requires separate convergence evidence.

A particle slot is physically inactive when `alive=False` and/or `weight=0`. Code should use both consistently.

`pid` identifies a computational slot for deterministic stochastic streams; it is not a physical electron identity and may be reset after resampling.

## 9. Pitch and low-energy boundaries

Pitch is physically restricted to

\[
-1\le\xi\le1.
\]

The small-angle reference operator reflects stochastic overshoot at these boundaries.

The relativistic energy boundary is

\[
\gamma\ge1.
\]

The reference small-angle operator reflects stochastic overshoot through \(\gamma=1\). A separate absorbing low-energy boundary may be applied by a boundary model.

## 10. Toroidal angle

`phi` is represented in radians. Collision steps wrap it to `[0, 2*pi)` for numerical cleanliness. Orbit modules may permit an unwrapped angle internally if needed for a particular integrator, provided field sampling is consistent.

## 11. Radial grids

The initial circular 1-D implementation assumes an ascending radial grid, normally uniform and spanning `[0, 1]`.

Current deposition and radial field operators assume at least three points. The axis is treated by regularity/Neumann conditions rather than evaluating `1/r` literally.

Generic axisymmetric `(R,Z)` field interpolation has independent monotone axes and does not require a uniform grid.

## 12. Array axis ordering

- Particle dimension: leading/only dimension for particle fields.
- Radial profiles: `(Nr,)`.
- Axisymmetric tables: `(NR, NZ)`.
- Atomic tables: `(NT, Z+1)` for one species.
- Evolved charge-state populations: spatial dimensions followed by charge state, `(..., Z+1)`.

Do not transpose these conventions inside individual modules.

## 13. Naming

Prefer descriptive names in code while retaining the mathematical symbol in docstrings.

Examples:

- `gamma`, not `g`, except in short local derivations;
- `a_omega_ce_over_c`, not ambiguous `wce`;
- `c_tau_over_a`, not ambiguous `tauc`;
- `eta_bar` for normalized resistivity;
- unit suffixes such as `_ev`, `_cm3`, `_m3`, `_w_m3`, `_s` for dimensional quantities.

Legacy RAMc names may appear in comments/docstrings when mapping equations to source code.

## 14. Geometry ownership

Generic modules must not infer circular geometry from `x` and `y`. The circular assumptions `x=r cos(theta)` and `R/R0=1+epsilon*x` belong to `fields/circular.py`, `orbits/ramc_circular.py`, and RAMc-specific deposition/coupling modules.

## 15. Interpolation ownership

A field/profile backend owns interpolation and any derivative tables required by its representation. Integrators and collision operators consume evaluated values and must not hard-code a particular interpolation library or order.
