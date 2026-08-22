"""Atomic-data interfaces for charge-state and radiation models.

JONTA does not perform network access inside simulation kernels. OPEN-ADAS
coefficients should be downloaded/preprocessed outside the hot path and
provided as tables implementing this interface.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol

import jax.numpy as jnp


class AtomicRates(NamedTuple):
    ionization_m3_s: jnp.ndarray
    recombination_m3_s: jnp.ndarray
    line_radiation_w_m3_per_ne_ni: jnp.ndarray


class AtomicDataProvider(Protocol):
    def rates(self, te_ev):
        ...


class TabulatedAtomicData(NamedTuple):
    """Preprocessed OPEN-ADAS-compatible rate tables for one species.

    Arrays ``ionization`` and ``recombination`` have shape ``(nT, Z+1)``.
    The last ionization coefficient and first recombination coefficient are
    conventionally zero. ``line_radiation`` uses the same layout.
    """

    temperature_ev: jnp.ndarray
    ionization_m3_s: jnp.ndarray
    recombination_m3_s: jnp.ndarray
    line_radiation_w_m3_per_ne_ni: jnp.ndarray


def interpolate_atomic_rates(table: TabulatedAtomicData, te_ev) -> AtomicRates:
    """Log-temperature interpolation of preprocessed atomic coefficients."""

    logt = jnp.log(jnp.maximum(table.temperature_ev, 1.0e-12))
    query = jnp.log(jnp.maximum(te_ev, 1.0e-12))

    def interp_col(values):
        return jnp.interp(query, logt, values)

    ion = jnp.stack([interp_col(table.ionization_m3_s[:, z]) for z in range(table.ionization_m3_s.shape[1])], axis=-1)
    rec = jnp.stack([interp_col(table.recombination_m3_s[:, z]) for z in range(table.recombination_m3_s.shape[1])], axis=-1)
    rad = jnp.stack([
        interp_col(table.line_radiation_w_m3_per_ne_ni[:, z])
        for z in range(table.line_radiation_w_m3_per_ne_ni.shape[1])
    ], axis=-1)
    return AtomicRates(ion, rec, rad)
