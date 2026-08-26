"""High-accuracy reference trajectories from the analytic circular-field RHS.

This is not an RK4 reference.  DOP853 integrates the analytic RAMC
characteristics with tight tolerances and adaptive internal steps.  The
result is used only as a numerical reference for timestep convergence.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp

if __package__:  # Package invocation: ``python -m benchmarks.one_d.tokamak_orbits.analytic_reference``.
    from .run import _git_metadata, _load_config, make_case
else:  # Direct script invocation used by benchmark documentation.
    from run import _git_metadata, _load_config, make_case

from core.state import KinematicState
from diagnostics.invariants import (
    magnetic_moment_circular,
    toroidal_canonical_momentum_circular,
)
from orbits.ramc_circular import ramc_circular_rhs

jax.config.update("jax_enable_x64", True)


def _reference_marker(y0, t_eval, profiles, norm):
    def rhs_jax(y):
        state = KinematicState(*y)
        out = ramc_circular_rhs(state, 0.0, profiles, norm)
        return jnp.stack((out.gamma, out.xi, out.x, out.y, out.phi))

    rhs_jit = jax.jit(rhs_jax)

    def rhs(_time, y):
        return np.asarray(rhs_jit(jnp.asarray(y)), dtype=float)

    solution = solve_ivp(
        rhs,
        (float(t_eval[0]), float(t_eval[-1])),
        np.asarray(y0, dtype=float),
        method="DOP853",
        t_eval=t_eval,
        rtol=2.0e-13,
        atol=2.0e-14,
        max_step=np.inf,
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    return solution.y.T


def run_reference(config, dt: float, final_time: float, output_dir: Path):
    kin, labels, profiles, norm = make_case(2, config)
    times = np.arange(int(round(final_time / dt)) + 1, dtype=float) * dt
    started = time.perf_counter()
    trajectories = []
    for i in range(len(labels)):
        y0 = jnp.stack((kin.gamma[i], kin.xi[i], kin.x[i], kin.y[i], kin.phi[i]))
        trajectories.append(_reference_marker(y0, times, profiles, norm))
    elapsed = time.perf_counter() - started

    history = np.empty((times.size, 6, len(labels)), dtype=float)
    for i, trajectory in enumerate(trajectories):
        gamma, xi, x, y, phi = [jnp.asarray(trajectory[:, k]) for k in range(5)]
        state = type(kin)(gamma, xi, x, y, phi)
        radius = jnp.sqrt(x * x + y * y)
        theta = jnp.arctan2(y, x)
        pphi = toroidal_canonical_momentum_circular(state, 0.0, profiles, norm)
        mu = magnetic_moment_circular(state, profiles, norm)
        history[:, :, i] = np.asarray(
            jnp.stack((radius, theta, xi, gamma, pphi, mu), axis=1)
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "analytic_reference_trajectories.npz",
        time=times,
        labels=labels,
        history=history,
    )
    commit, dirty = _git_metadata()
    manifest = {
        "benchmark": "one_d.tokamak_orbits",
        "mode": "analytic_equation_reference",
        "solver": "scipy.solve_ivp:DOP853",
        "rtol": 2.0e-13,
        "atol": 2.0e-14,
        "dt_output": dt,
        "final_time": final_time,
        "n_samples": int(times.size),
        "runtime_seconds": elapsed,
        "git_commit": commit,
        "git_dirty": dirty,
        "outputs": ["analytic_reference_trajectories.npz", "manifest.json"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument("--dt", type=float)
    parser.add_argument("--final-time", type=float)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config = _load_config(args.config)
    case = config["case"]
    dt = args.dt if args.dt is not None else float(case["dt"])
    final_time = args.final_time if args.final_time is not None else float(case["final_time"])
    print(json.dumps(run_reference(config, dt, final_time, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
