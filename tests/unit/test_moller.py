import jax
import jax.numpy as jnp

from collisions.moller import (
    gain_loss_candidates,
    moller_dsigma_dgamma,
    moller_outgoing_pair,
    moller_tail_cross_section,
    source_only_candidates,
)
from core.config import MollerConfig
from core.initialization import particles_from_arrays
from core.state import BackgroundProfiles


def _background():
    r = jnp.linspace(0.0, 1.0, 8)
    return BackgroundProfiles(r, jnp.full_like(r, 1e14), jnp.full_like(r, 10.0), jnp.full_like(r, 10.0), jnp.ones_like(r), jnp.ones_like(r))


def test_moller_cross_section_positive():
    assert float(moller_tail_cross_section(jnp.array(5.0), jnp.array(1.1))) > 0.0


def test_outgoing_pair_conserves_kinetic_energy_and_parallel_momentum():
    g0 = jnp.array([5.0])
    xi0 = jnp.array([-0.8])
    gp, xip, gs, xis = moller_outgoing_pair(g0, xi0, jnp.array([1.5]), jnp.array([0.37]))
    assert jnp.allclose((gp - 1.0) + (gs - 1.0), g0 - 1.0, rtol=1e-13)
    p0 = jnp.sqrt(g0*g0 - 1.0)
    pp = jnp.sqrt(gp*gp - 1.0)
    ps = jnp.sqrt(gs*gs - 1.0)
    assert jnp.allclose(pp*xip + ps*xis, p0*xi0, rtol=1e-12, atol=1e-12)


def test_gain_loss_candidates_conserve_energy_and_have_expected_weight_gain():
    p = particles_from_arrays([5.0, 8.0], [-0.8, -0.9], [0.1, 0.2], [0.0, 0.0], [0.0, 0.0], [2.0, 3.0])
    cfg = MollerConfig(1e14, 15.0, 1.1, max_collision_fraction=0.25)
    cand, q_raw = gain_loss_candidates(p, _background(), 0.01, jax.random.key(2), 0, cfg)
    q = jnp.clip(q_raw, 0.0, cfg.max_collision_fraction)
    expected_weight = jnp.sum(p.weight) + jnp.sum(p.weight * q)
    assert jnp.allclose(jnp.sum(cand.weight), expected_weight, rtol=1e-13)
    e0 = jnp.sum(p.weight * (p.kin.gamma - 1.0))
    ec = jnp.sum(cand.weight * (cand.kin.gamma - 1.0))
    assert jnp.allclose(ec, e0, rtol=1e-12, atol=1e-12)



def test_tail_cross_section_matches_numerical_integral():
    """The analytic collision probability integrates the differential kernel."""

    for gamma0 in (3.0, 5.0, 10.0, 20.0):
        for gamma_min in (1.02, 1.1, 1.5):
            if gamma0 <= 2.0 * gamma_min - 1.0:
                continue
            grid_t = jnp.geomspace(gamma_min - 1.0, 0.5 * (gamma0 - 1.0), 30001)
            grid = 1.0 + grid_t
            numeric = jnp.trapezoid(moller_dsigma_dgamma(gamma0, grid), grid)
            analytic = moller_tail_cross_section(gamma0, gamma_min)
            assert jnp.allclose(numeric, analytic, rtol=2e-7, atol=1e-10)


def test_outgoing_pair_closes_full_three_momentum_kinematics():
    """Cold-target Moller kinematics closes the full 3-D momentum triangle."""

    g0 = jnp.array([3.0, 5.0, 12.0, 20.0])
    xi0 = jnp.array([-0.95, -0.8, -0.4, 0.2])
    gs = jnp.array([1.1, 1.4, 2.5, 4.0])
    u = jnp.array([0.0, 0.17, 0.43, 0.91])
    gp, xip, gs, xis = moller_outgoing_pair(g0, xi0, gs, u)

    p0 = jnp.sqrt(g0 * g0 - 1.0)
    ps = jnp.sqrt(gs * gs - 1.0)
    pp = jnp.sqrt(gp * gp - 1.0)
    cos_theta = jnp.sqrt((g0 + 1.0) * (gs - 1.0) / ((g0 - 1.0) * (gs + 1.0)))
    # |p0 - ps|^2 = pp^2 is the transverse+parallel vector closure.
    pp_from_triangle = jnp.sqrt(p0 * p0 + ps * ps - 2.0 * p0 * ps * cos_theta)
    assert jnp.allclose(pp_from_triangle, pp, rtol=2e-13, atol=2e-13)
    assert jnp.allclose(pp * xip + ps * xis, p0 * xi0, rtol=2e-12, atol=2e-12)


def test_source_only_candidates_leave_primary_and_add_expected_secondary_weight():
    p = particles_from_arrays(
        [5.0, 8.0], [-0.8, -0.9], [0.1, 0.2], [0.0, 0.0], [0.0, 0.0], [2.0, 3.0]
    )
    cfg = MollerConfig(1e14, 15.0, 1.1, max_collision_fraction=0.25)
    cand, q_raw = source_only_candidates(p, _background(), 0.01, jax.random.key(9), 0, cfg)
    q = jnp.clip(q_raw, 0.0, cfg.max_collision_fraction)
    n = p.weight.size
    assert jnp.allclose(cand.kin.gamma[:n], p.kin.gamma)
    assert jnp.allclose(cand.kin.xi[:n], p.kin.xi)
    assert jnp.allclose(cand.weight[:n], p.weight)
    assert jnp.allclose(jnp.sum(cand.weight[n:]), jnp.sum(p.weight * q), rtol=1e-13)
    assert jnp.all(cand.kin.gamma[n:] >= cfg.gamma_min)
    assert jnp.all(cand.kin.gamma[n:] <= 0.5 * (p.kin.gamma + 1.0) + 1e-13)
