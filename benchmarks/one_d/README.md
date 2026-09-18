# One-dimensional circular benchmarks

These cases use JONTA's circular, radial 1-D guiding-center geometry. They
validate invariants, trapped/passing tokamak orbits, radial transport, and radial avalanche behavior before
plasma coupling is enabled.

Drivers use JAX backend selection. Invariant-conservation driver supports local
GPU scaling and distributed multi-process scaling. Other benchmark GPU paths
remain separate migration work.
