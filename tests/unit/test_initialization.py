import jax
import numpy as np

from core.initialization import (
    rosenbluth_avalanche_growth_estimate,
    rosenbluth_energy_scale,
    rosenbluth_growth_coefficient,
    rosenbluth_legendre_markers,
)

jax.config.update("jax_enable_x64", True)


def test_rosenbluth_energy_scale_matches_high_field_formula():
    gamma0 = rosenbluth_growth_coefficient(zeff=1.0, coulog=10.0)
    expected = np.sqrt(np.pi / 18.0) / 10.0
    assert np.isclose(gamma0, expected, rtol=1e-13)

    growth = rosenbluth_avalanche_growth_estimate(70.0, 1.0, 10.0)
    assert np.isclose(growth, gamma0 * 69.0, rtol=1e-13)

    # The acceleration/growth estimate collapses to 1/gamma0 in the
    # high-field Rosenbluth-Putvinski model used by McDevitt Fig. B2.
    scale = rosenbluth_energy_scale(70.0, 1.0, 10.0)
    assert np.isclose(scale, 1.0 / gamma0, rtol=1e-13)

    capped = rosenbluth_energy_scale(3.2321, 2.0, 20.0, alpha_syn=0.5)
    assert capped < rosenbluth_energy_scale(3.2321, 2.0, 20.0)
    assert 5.0 < capped < 9.0


def test_rosenbluth_legendre_markers_are_energetic_and_field_aligned():
    p = rosenbluth_legendre_markers(
        jax.random.key(7),
        4096,
        e_over_ec=3.2321,
        zeff=2.0,
        coulog=20.0,
        pitch_width=None,
        lmax=16,
        alpha_syn=0.5,
    )
    gamma = np.asarray(p.kin.gamma)
    xi = np.asarray(p.kin.xi)
    weight = np.asarray(p.weight)

    assert np.all(gamma >= 1.02)
    assert np.all((-1.0 <= xi) & (xi <= 1.0))
    assert np.mean(xi) < -0.80
    assert np.quantile(gamma - 1.0, 0.5) > 2.0
    assert np.isclose(np.sum(weight), 1.0, rtol=1e-13)
