"""Diagnostics for runaway-electron avalanche benchmarks."""

from __future__ import annotations

import numpy as np


def replicate_mean_sem(values):
    """Return mean, sample standard deviation, and SEM across independent runs.

    Replica-to-replica statistics are the reference uncertainty measure for
    branching calculations because capacity overflow resampling correlates
    markers inside any one ensemble.
    """

    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("at least two 1-D independent replicates are required")
    std = float(np.std(x, ddof=1))
    return float(np.mean(x)), std, float(std / np.sqrt(x.size))


def monte_carlo_convergence_slope(marker_counts, stddev):
    """Fit ``sigma ~ N**slope`` for a marker-count convergence scan."""

    n = np.asarray(marker_counts, dtype=float)
    sigma = np.asarray(stddev, dtype=float)
    mask = np.isfinite(n) & np.isfinite(sigma) & (n > 0.0) & (sigma > 0.0)
    if np.count_nonzero(mask) < 2:
        raise ValueError("at least two positive convergence points are required")
    slope, intercept = np.polyfit(np.log(n[mask]), np.log(sigma[mask]), 1)
    return float(slope), float(intercept)


def fit_exponential_growth(time, total_weight, fit_start_fraction: float = 0.35):
    """Fit ``N(t) ~ exp(gamma_av t)`` and return slope, intercept, and R^2.

    The fit is performed in log space after discarding the requested initial
    fraction of samples. Non-positive/non-finite weights are excluded.
    """

    t = np.asarray(time, dtype=float)
    w = np.asarray(total_weight, dtype=float)
    if t.ndim != 1 or w.ndim != 1 or t.size != w.size:
        raise ValueError("time and total_weight must be 1-D arrays of equal length")
    start = int(np.floor(np.clip(fit_start_fraction, 0.0, 0.95) * t.size))
    mask = np.isfinite(t) & np.isfinite(w) & (w > 0.0)
    mask[:start] = False
    if np.count_nonzero(mask) < 3:
        raise ValueError("not enough positive samples for avalanche-growth fit")
    x = t[mask]
    y = np.log(w[mask])
    slope, intercept = np.polyfit(x, y, 1)
    yfit = slope * x + intercept
    ss_res = np.sum((y - yfit) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return float(slope), float(intercept), float(r2)


def fit_window_sensitivity(
    time,
    total_weight,
    fit_start_fraction: float = 0.35,
    offsets=(-0.20, -0.10, 0.0, 0.10, 0.20),
):
    """Estimate growth-rate sensitivity to the selected fit window.

    Returns the sample standard deviation of valid slopes from nearby fit
    windows. This is a systematic fit-choice component, not an independent
    replica SEM, and therefore must not be divided by ``sqrt(N)``.
    """

    fractions = sorted(
        {
            float(np.clip(fit_start_fraction + offset, 0.0, 0.9))
            for offset in offsets
        }
    )
    slopes = []
    for fraction in fractions:
        try:
            slopes.append(fit_exponential_growth(time, total_weight, fraction)[0])
        except ValueError:
            continue
    if len(slopes) < 2:
        return float("nan")
    return float(np.std(np.asarray(slopes), ddof=1))
