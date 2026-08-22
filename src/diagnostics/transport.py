"""Monte Carlo estimators for radial convection and diffusion.

The estimators follow the appendix of McDevitt, Guo & Tang,
Plasma Phys. Control. Fusion 61, 024004 (2019):

    V = d <Delta r> / dt
    D = 1/2 d [<Delta r^2> - <Delta r>^2] / dt.

The hot simulation kernels accumulate the moments in JAX.  Linear fits are
performed on the host because they are validation diagnostics rather than part
of the production particle push.
"""

from __future__ import annotations

import numpy as np


def fit_transport_coefficients(time, mean_dr, mean_dr2, fit_start_fraction=0.25):
    """Fit convection and diffusion from radial-displacement moments.

    Parameters
    ----------
    time : array-like, shape (nt,)
        Time normalized to tau_c.
    mean_dr, mean_dr2 : array-like, shape (nt, ngroup)
        Group-averaged first and second radial-displacement moments, with
        radius normalized to the minor radius.
    fit_start_fraction : float
        Fraction of the history discarded before the fit.  This removes the
        rapid finite-orbit-width transient before the diffusive regime.

    Returns
    -------
    dict containing ``V``, ``D``, and R^2 values for the first-moment and
    variance fits.  ``D`` is normalized as tau_c D / a^2.
    """

    time = np.asarray(time, dtype=float)
    mean_dr = np.asarray(mean_dr, dtype=float)
    mean_dr2 = np.asarray(mean_dr2, dtype=float)
    variance = mean_dr2 - mean_dr * mean_dr

    if mean_dr.ndim == 1:
        mean_dr = mean_dr[:, None]
        mean_dr2 = mean_dr2[:, None]
        variance = variance[:, None]

    start = max(1, int(round(fit_start_fraction * (time.size - 1))))
    tfit = time[start:]
    x = tfit - np.mean(tfit)
    denom = np.sum(x * x)

    def fit(y):
        yfit = y[start:]
        ybar = np.mean(yfit, axis=0)
        slope = np.sum(x[:, None] * (yfit - ybar), axis=0) / denom
        intercept = ybar - slope * np.mean(tfit)
        pred = intercept[None, :] + tfit[:, None] * slope[None, :]
        ss_res = np.sum((yfit - pred) ** 2, axis=0)
        ss_tot = np.sum((yfit - ybar[None, :]) ** 2, axis=0)
        r2 = 1.0 - ss_res / np.maximum(ss_tot, 1.0e-300)
        return slope, intercept, r2

    v, mean_intercept, r2_mean = fit(mean_dr)
    variance_slope, variance_intercept, r2_variance = fit(variance)
    return {
        "V": v,
        "D": 0.5 * variance_slope,
        "mean_intercept": mean_intercept,
        "variance_slope": variance_slope,
        "variance_intercept": variance_intercept,
        "r2_mean": r2_mean,
        "r2_variance": r2_variance,
        "variance": variance,
        "fit_start_index": start,
    }
