import jax.numpy as jnp

from core.state import KinematicState
from fields.uniform import UniformField
from orbits.zero_d import zero_d_rhs


def _state(gamma=2.0, xi=-0.8):
    a = jnp.array([gamma], dtype=jnp.float64)
    z = jnp.zeros_like(a)
    return KinematicState(a, jnp.array([xi]), z, z, z)


def test_zero_d_no_force_is_stationary():
    rhs = zero_d_rhs(_state(), 0.0, UniformField(0.0), alpha_syn=0.0)
    assert float(rhs.gamma[0]) == 0.0
    assert float(rhs.xi[0]) == 0.0


def test_positive_ramc_field_accelerates_negative_pitch_electron():
    rhs = zero_d_rhs(_state(), 0.0, UniformField(2.0), alpha_syn=0.0)
    assert float(rhs.gamma[0]) > 0.0
