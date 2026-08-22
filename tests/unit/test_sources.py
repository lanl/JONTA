import jax.numpy as jnp

from sources.tritium import tritium_source_density_p, tritium_total_rate


def test_tritium_source_positive_below_endpoint():
    p = jnp.array([0.1])
    assert float(tritium_source_density_p(p, 1e20)[0]) > 0.0
    assert float(tritium_total_rate(1e20)) > 0.0
