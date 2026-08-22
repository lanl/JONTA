import jax
import jax.numpy as jnp

from core.initialization import particles_from_arrays
from resampling.multinomial import effective_sample_size, multinomial_resample, stratified_resample


def test_multinomial_resampling_preserves_total_weight():
    p = particles_from_arrays([2.0, 3.0, 4.0], [-1.0, -0.5, 0.2], [0.0]*3, [0.0]*3, [0.0]*3, [1.0, 2.0, 5.0])
    out = multinomial_resample(p, jax.random.key(0), 3)
    assert jnp.allclose(jnp.sum(out.weight), jnp.sum(p.weight))
    assert out.weight.shape == (3,)


def test_effective_sample_size_bounds():
    ess = effective_sample_size(jnp.array([1.0, 1.0, 1.0]))
    assert jnp.allclose(ess, 3.0)


def test_stratified_resampling_preserves_total_weight():
    p = particles_from_arrays([2.0, 3.0, 4.0], [-1.0, -0.5, 0.2], [0.0]*3, [0.0]*3, [0.0]*3, [1.0, 2.0, 5.0])
    out = stratified_resample(p, jax.random.key(4), 5)
    assert jnp.allclose(jnp.sum(out.weight), jnp.sum(p.weight))
    assert out.weight.shape == (5,)
    assert jnp.all(out.alive)


def test_stratified_resampling_is_unbiased_for_weighted_moment():
    """Independent thinning replicas recover the candidate weighted mean."""

    p = particles_from_arrays(
        [1.5, 2.5, 4.0, 7.0],
        [-1.0, -0.3, 0.4, 0.9],
        [0.0] * 4,
        [0.0] * 4,
        [0.0] * 4,
        [1.0, 2.0, 4.0, 8.0],
    )
    exact = jnp.sum(p.weight * p.kin.gamma) / jnp.sum(p.weight)

    def sample(key):
        out = stratified_resample(p, key, 8)
        return jnp.sum(out.weight * out.kin.gamma) / jnp.sum(out.weight)

    keys = jnp.stack([jax.random.key(i) for i in range(2048)])
    estimates = jax.vmap(sample)(keys)
    mean = jnp.mean(estimates)
    sem = jnp.std(estimates, ddof=1) / jnp.sqrt(estimates.size)
    assert jnp.abs(mean - exact) < 5.0 * sem + 1.0e-12
