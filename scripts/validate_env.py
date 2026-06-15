#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from importlib import metadata


def _version(dist_name: str) -> str:
    try:
        return metadata.version(dist_name)
    except metadata.PackageNotFoundError:
        return "<not installed>"


def _require_import(module: str) -> None:
    __import__(module)


def validate(track: str) -> int:
    common = [
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("pandas", "pandas"),
    ]
    analysis = [
        ("anndata", "anndata"),
        ("spatialdata", "spatialdata"),
        ("spatialdata-io", "spatialdata_io"),
        ("scanpy", "scanpy"),
        ("squidpy", "squidpy"),
    ]
    image = [
        ("python-javabridge", "javabridge"),
        ("python-bioformats", "bioformats"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("cellpose", "cellpose"),
        ("scikit-image", "skimage"),
        ("tifffile", "tifffile"),
        ("pims", "pims"),
    ]

    if track not in {"analysis", "image"}:
        print("track must be 'analysis' or 'image'", file=sys.stderr)
        return 2

    reqs = common + (analysis if track == "analysis" else image)

    failed: list[str] = []
    print(f"[spatial-tk] validating {track} environment\n")

    for dist, module in reqs:
        v = _version(dist)
        print(f"{dist:18s} {v}")
        try:
            _require_import(module)
        except Exception as e:
            failed.append(f"{dist} (import {module}): {e}")

    if failed:
        print("\nMissing/broken imports:", file=sys.stderr)
        for line in failed:
            print(f"- {line}", file=sys.stderr)
        return 1

    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Validate spatial-tk dependency stacks")
    p.add_argument("track", choices=["analysis", "image"])
    args = p.parse_args()
    raise SystemExit(validate(args.track))


if __name__ == "__main__":
    main()

