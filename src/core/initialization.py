"""Marker-ensemble initialization helpers."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from .constants import PI
from .precision import PrecisionValue, configure_precision, index_dtype, real_dtype
from .state import KinematicState, ParticleState


def particles_from_arrays(gamma, xi, x, y, phi, weight, alive=None, pid=None, *, precision: PrecisionValue = None):
    configure_precision(precision)
    dtype = real_dtype()
    gamma = jnp.asarray(gamma, dtype=dtype)
    n = gamma.shape[0]
    kin = KinematicState(
        gamma,
        jnp.asarray(xi, dtype=dtype),
        jnp.asarray(x, dtype=dtype),
        jnp.asarray(y, dtype=dtype),
        jnp.asarray(phi, dtype=dtype),
    )
    w = jnp.broadcast_to(jnp.asarray(weight, dtype=dtype), (n,))
    if alive is None:
        alive_arr = jnp.ones((n,), dtype=jnp.bool_)
    else:
        alive_arr = jnp.broadcast_to(jnp.asarray(alive, dtype=jnp.bool_), (n,))
    if pid is None:
        pid_arr = jnp.arange(n, dtype=index_dtype())
    else:
        pid_arr = jnp.broadcast_to(jnp.asarray(pid, dtype=index_dtype()), (n,))
    return ParticleState(kin, w, alive_arr, pid_arr)


def uniform_markers(
    key,
    n: int,
    gamma_range=(1.01, 2.0),
    xi_range=(-1.0, 1.0),
    r_range=(0.0, 0.9),
    total_weight: float = 1.0,
    *,
    precision: PrecisionValue = None,
):
    """Simple uniform marker initializer for benchmarks and examples."""

    configure_precision(precision)
    dtype = real_dtype()
    kg, kxi, kr, kth, kphi = jax.random.split(key, 5)
    gamma = jax.random.uniform(kg, (n,), dtype=dtype, minval=gamma_range[0], maxval=gamma_range[1])
    xi = jax.random.uniform(kxi, (n,), dtype=dtype, minval=xi_range[0], maxval=xi_range[1])
    # sqrt-uniform radius gives uniform area density in the poloidal plane.
    r2 = jax.random.uniform(kr, (n,), dtype=dtype, minval=r_range[0] ** 2, maxval=r_range[1] ** 2)
    r = jnp.sqrt(r2)
    theta = jax.random.uniform(kth, (n,), dtype=dtype, minval=-PI, maxval=PI)
    phi = jax.random.uniform(kphi, (n,), dtype=dtype, minval=0.0, maxval=2.0 * PI)
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return particles_from_arrays(gamma, xi, x, y, phi, total_weight / n)


def rosenbluth_growth_coefficient(zeff: float, coulog: float, toroidal_factor: float = 1.0) -> float:
    """Return the asymptotic Rosenbluth-Putvinski ``gamma0 * tau_c``.

    This is the high-field coefficient used in McDevitt et al. (2019), Eq. (2),
    specialized to a prescribed toroidicity factor.  ``toroidal_factor=1`` is
    the slab / magnetic-axis limit used by the Appendix-B avalanche tests.
    """

    zeff = float(zeff)
    coulog = float(coulog)
    toroidal_factor = float(toroidal_factor)
    if coulog <= 0.0 or zeff <= -5.0 or toroidal_factor <= 0.0:
        raise ValueError("invalid Rosenbluth-Putvinski parameters")
    return float(jnp.sqrt(PI * toroidal_factor / (3.0 * (zeff + 5.0))) / coulog)


def rosenbluth_avalanche_growth_estimate(
    e_over_ec: float,
    zeff: float,
    coulog: float,
    toroidal_factor: float = 1.0,
) -> float:
    """High-field estimate of ``gamma_av * tau_c`` used for initialization."""

    return rosenbluth_growth_coefficient(zeff, coulog, toroidal_factor) * max(
        float(e_over_ec) - 1.0, 0.0
    )


def runaway_o_point_momentum(e_over_ec: float, zeff: float, alpha_syn: float) -> float:
    """Large-p Guo-McDevitt-Tang estimate for the runaway-vortex O point.

    The expression is useful here only as a radiation-limited characteristic
    energy for initialization.  The subsequent Monte Carlo solve determines
    the actual avalanche eigenmode.
    """

    e = float(e_over_ec)
    z = float(zeff)
    alpha = float(alpha_syn)
    if e <= 1.0 or alpha <= 0.0 or z <= -1.0:
        return float("inf")
    return float(jnp.sqrt(2.0) * (e + alpha) * (e - 1.0) / ((1.0 + z) * alpha))


def runaway_pitch_channel_width(
    e_over_ec: float,
    p: float,
    alpha_syn: float,
    *,
    width_floor: float = 0.02,
    width_ceiling: float = 0.35,
) -> float:
    """Approximate field-aligned runaway-channel width in pitch.

    This is the large-p channel-width estimate of Guo et al. (2017), Eq. (16),
    clipped to a numerically useful range for a finite Legendre warm start.
    """

    e = float(e_over_ec)
    p = max(float(p), 1.0e-8)
    alpha = max(float(alpha_syn), 0.0)
    gamma = float(jnp.sqrt(1.0 + p * p))
    numerator = e - 1.0 - 1.0 / (p * p)
    denominator = 2.0 * alpha * p * gamma + e
    if denominator <= 0.0 or numerator <= 0.0:
        width = width_ceiling
    else:
        width = numerator / denominator
    return float(min(width_ceiling, max(width_floor, width)))


def rosenbluth_energy_scale(
    e_over_ec: float,
    zeff: float,
    coulog: float,
    toroidal_factor: float = 1.0,
    *,
    alpha_syn: float = 0.0,
    radiation_limit: bool = True,
) -> float:
    """Estimate the exponential kinetic-energy scale of the avalanche mode.

    For a relativistic runaway, ``d(gamma-1)/d(t/tau_c) ~= E/Ec - 1``.  Combining
    that acceleration rate with ``gamma_av ~= gamma0 (E/Ec - 1)`` gives the
    Rosenbluth scale ``T_R/(m_e c^2) ~= 1/(gamma0 tau_c)`` used in McDevitt
    Fig. B2(b).

    At the moderate fields used by Fig. B3, synchrotron losses can cap the
    accessible energy well below the asymptotic Rosenbluth scale.  For warm
    starts only, we therefore limit the scale by the analytically estimated
    runaway-vortex O-point kinetic energy when ``alpha_syn > 0``.  This keeps
    the initializer physically close without changing the kinetic operator.
    """

    acceleration = max(float(e_over_ec) - 1.0, 0.0)
    growth = rosenbluth_avalanche_growth_estimate(
        e_over_ec, zeff, coulog, toroidal_factor
    )
    if acceleration <= 0.0 or growth <= 0.0:
        raise ValueError("Rosenbluth avalanche initializer requires E/Ec > 1")
    scale = acceleration / growth
    if radiation_limit and float(alpha_syn) > 0.0:
        p_o = runaway_o_point_momentum(e_over_ec, zeff, alpha_syn)
        if jnp.isfinite(p_o):
            kinetic_o = float(jnp.sqrt(1.0 + p_o * p_o) - 1.0)
            scale = min(scale, kinetic_o)
    return float(scale)


def _legendre_beam_pdf_grid(
    *,
    width: float,
    lmax: int,
    n_grid: int = 2049,
    quadrature_order: int = 160,
):
    """Project a field-aligned exponential beam onto a finite Legendre basis.

    The target marginal pitch density is ``exp[-(1+xi)/width]`` and is peaked
    at ``xi=-1`` for the sign convention used by JONTA's runaway benchmarks.
    The finite expansion is clipped only when forming a sampling CDF, which
    prevents small Gibbs undershoots from generating a nonphysical PDF.
    """

    import numpy as np
    from numpy.polynomial.legendre import leggauss, legval

    width = float(width)
    lmax = int(lmax)
    if width <= 0.0 or lmax < 0:
        raise ValueError("width must be positive and lmax non-negative")

    xq, wq = leggauss(int(quadrature_order))
    target = np.exp(-(1.0 + xq) / width)
    coeff = np.empty(lmax + 1, dtype=float)
    for ell in range(lmax + 1):
        basis_coeff = np.zeros(ell + 1)
        basis_coeff[-1] = 1.0
        pell = legval(xq, basis_coeff)
        coeff[ell] = 0.5 * (2 * ell + 1) * np.sum(wq * target * pell)

    xi_grid = np.linspace(-1.0, 1.0, int(n_grid))
    raw_pdf = legval(xi_grid, coeff)
    pdf = np.maximum(raw_pdf, 0.0)
    norm = np.trapezoid(pdf, xi_grid)
    if not np.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("Legendre pitch projection produced an invalid PDF")
    pdf /= norm
    cdf = np.zeros_like(pdf)
    dx = np.diff(xi_grid)
    cdf[1:] = np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dx)
    cdf /= cdf[-1]
    return xi_grid, pdf, cdf, coeff


def rosenbluth_legendre_markers(
    key,
    n: int,
    *,
    e_over_ec: float,
    zeff: float,
    coulog: float,
    gamma_min: float = 1.02,
    energy_extent: float = 6.0,
    pitch_width: float | None = None,
    lmax: int = 10,
    toroidal_factor: float = 1.0,
    alpha_syn: float = 0.0,
    total_weight: float = 1.0,
):
    """Approximate avalanche-eigenmode initializer for fixed-capacity calculations.

    Energy is sampled from the Rosenbluth exponentially decaying kinetic-energy
    spectrum using an analytically estimated acceleration time / avalanche
    growth rate.  Pitch is represented by a finite Legendre expansion of a
    narrow field-aligned beam and sampled from that positive reconstructed
    marginal.  The construction is intentionally approximate: it is a warm
    start whose value is judged by reduced Monte-Carlo burn-in, not an attempt
    to replace the kinetic solve.
    """

    import numpy as np

    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if gamma_min <= 1.0:
        raise ValueError("gamma_min must exceed 1")
    if energy_extent <= 0.0:
        raise ValueError("energy_extent must be positive")

    tav = rosenbluth_energy_scale(
        e_over_ec,
        zeff,
        coulog,
        toroidal_factor,
        alpha_syn=alpha_syn,
        radiation_limit=True,
    )
    kmin = float(gamma_min) - 1.0
    kmax = kmin + float(energy_extent) * tav

    # JAX keys keep replica initialization deterministic and device-independent.
    k_energy, k_pitch = jax.random.split(key, 2)
    u_energy = np.asarray(jax.random.uniform(k_energy, (n,), dtype=real_dtype()))
    u_pitch = np.asarray(jax.random.uniform(k_pitch, (n,), dtype=real_dtype()))

    # Inverse CDF of a truncated exponential in kinetic energy gamma-1.
    span = kmax - kmin
    retained = 1.0 - np.exp(-span / tav)
    kinetic = kmin - tav * np.log1p(-u_energy * retained)
    gamma = 1.0 + kinetic

    if pitch_width is None:
        # The runaway channel narrows strongly with momentum.  Represent this
        # natural p-xi correlation by allowing the Legendre coefficients a_l(p)
        # to vary through a small set of momentum-dependent pitch-width bins.
        momentum = np.sqrt(np.maximum(gamma * gamma - 1.0, 0.0))
        widths = np.asarray(
            [
                runaway_pitch_channel_width(e_over_ec, pp, alpha_syn)
                for pp in momentum
            ],
            dtype=float,
        )
        wmin = float(np.min(widths))
        wmax = float(np.max(widths))
        if np.isclose(wmin, wmax):
            centers = np.asarray([wmin])
        else:
            centers = np.geomspace(max(wmin, 1.0e-4), wmax, 24)
        # Assign each particle to the closest logarithmic width bin.
        logw = np.log(np.maximum(widths, 1.0e-12))
        logc = np.log(np.maximum(centers, 1.0e-12))
        ibin = np.argmin(np.abs(logw[:, None] - logc[None, :]), axis=1)
        xi = np.empty(n, dtype=float)
        for j, width_j in enumerate(centers):
            mask = ibin == j
            if not np.any(mask):
                continue
            xi_grid, _pdf, cdf, _coeff = _legendre_beam_pdf_grid(
                width=float(width_j),
                lmax=lmax,
            )
            xi[mask] = np.interp(u_pitch[mask], cdf, xi_grid)
    else:
        xi_grid, _pdf, cdf, _coeff = _legendre_beam_pdf_grid(
            width=float(pitch_width),
            lmax=lmax,
        )
        xi = np.interp(u_pitch, cdf, xi_grid)

    z = np.zeros(n)
    return particles_from_arrays(
        gamma,
        xi,
        z,
        z,
        z,
        np.full(n, float(total_weight) / n),
        pid=np.arange(n),
    )
