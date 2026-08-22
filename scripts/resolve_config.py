#!/usr/bin/env python3
"""Validate a JONTA YAML configuration and optionally write its resolved form."""

from __future__ import annotations

import argparse
from pathlib import Path

from core.configuration import dump_resolved_config, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="YAML configuration to validate")
    parser.add_argument(
        "--output",
        type=Path,
        help="write deterministic resolved YAML to this path",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output is not None:
        dump_resolved_config(config, args.output)
        print(f"validated; wrote {args.output}")
    else:
        print(f"validated: {config.benchmark.name}")
        print(f"geometry={config.model.geometry} precision={config.model.precision}")
        print(f"markers={config.particles.markers} output={config.output.directory}")


if __name__ == "__main__":
    main()
