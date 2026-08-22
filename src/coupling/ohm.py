"""Reduced Ohm-law closures used by 0-D and coupling tests."""

from __future__ import annotations


def ohmic_current(e_parallel, eta):
    return e_parallel / eta


def electric_field_from_total_current(j_total, j_kinetic, eta):
    """Algebraic Ohm law E = eta (j_total - j_kinetic)."""

    return eta * (j_total - j_kinetic)
