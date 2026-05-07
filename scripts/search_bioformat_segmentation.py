#!/usr/bin/env python3
"""Grid-search Cellpose parameters through `spatial-tk image import-bioformat`."""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _parse_csv_values(value: str, cast):
    return [cast(part.strip()) for part in value.split(",") if part.strip()]


def _image_env(image_python: Path) -> dict[str, str]:
    env = dict(os.environ)
    prefix = image_python.parent.parent
    java_home = os.getenv("JAVA_HOME") or str(prefix / "lib" / "jvm")
    env["JAVA_HOME"] = java_home
    env["PATH"] = f"{Path(java_home) / 'bin'}:{env.get('PATH', '')}"
    return env


def _object_count(objects_csv: Path) -> int | None:
    if not objects_csv.exists():
        return None
    with objects_csv.open("r", encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


def _run(cmd: list[str], *, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)


def main(argv: list[str] | None = None) -> int:
    repo = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(repo / "tests" / "test_data" / "test.oir"),
        help="Input OIR/OME-TIFF file.",
    )
    parser.add_argument(
        "--image-python",
        default=str(repo / "venv_image" / "bin" / "python"),
        help="Python executable for the image environment.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(repo / "tmp" / "csv2zarr_param_search"),
        help="Directory where parameter-search artifacts are written.",
    )
    parser.add_argument("--channels", default="0,1,2,3", help="Comma-separated channel indices.")
    parser.add_argument(
        "--diameters",
        default="8,12,16,20,24,32,40",
        help="Comma-separated Cellpose diameters in pixels.",
    )
    parser.add_argument(
        "--models",
        default="nuclei",
        help="Comma-separated Cellpose model types: nuclei,cyto,cyto2.",
    )
    parser.add_argument(
        "--z-projections",
        default="max",
        help="Comma-separated z projections to test: max,middle.",
    )
    parser.add_argument("--target-min", type=int, default=10)
    parser.add_argument("--target-max", type=int, default=50)
    parser.add_argument("--gpu", action="store_true", help="Pass --segment-gpu to Cellpose.")
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not clear the output directory before running.",
    )
    args = parser.parse_args(argv)

    inp = Path(args.input).expanduser().resolve()
    image_python = Path(args.image_python).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()

    if not inp.exists():
        parser.error(f"input file not found: {inp}")
    if not image_python.exists():
        parser.error(f"image python not found: {image_python}")

    if out_dir.exists() and not args.keep_existing:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    channels = _parse_csv_values(args.channels, int)
    diameters = _parse_csv_values(args.diameters, float)
    models = _parse_csv_values(args.models, str)
    z_projections = _parse_csv_values(args.z_projections, str)
    env = _image_env(image_python)

    rows: list[dict[str, object]] = []
    for model in models:
        for z_projection in z_projections:
            for channel in channels:
                for diameter in diameters:
                    run_name = (
                        f"model-{model}_z-{z_projection}_ch-{channel}_diam-{diameter:g}"
                    )
                    run_dir = out_dir / run_name
                    bundle_dir = run_dir / "bundle"
                    preview_png = run_dir / "preview.png"
                    run_dir.mkdir(parents=True, exist_ok=True)

                    cmd = [
                        str(image_python),
                        "-m",
                        "spatial_tk.cli",
                        "image",
                        "import-bioformat",
                        "--input",
                        str(inp),
                        "--segment",
                        "--channels",
                        str(channel),
                        "--segment-model",
                        model,
                        "--segment-diameter",
                        str(diameter),
                        "--z-projection",
                        z_projection,
                        "--preview-png",
                        str(preview_png),
                        "--export-dir",
                        str(bundle_dir),
                        "--labels-key",
                        "segmentation_labels",
                        "--shapes-key",
                        "segmentation_polygons",
                    ]
                    if args.gpu:
                        cmd.append("--segment-gpu")

                    print(f"[search] {run_name}", flush=True)
                    result = _run(cmd, env=env, cwd=repo)
                    (run_dir / "command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
                    (run_dir / "stdout.log").write_text(result.stdout, encoding="utf-8")
                    (run_dir / "stderr.log").write_text(result.stderr, encoding="utf-8")

                    n_objects = _object_count(bundle_dir / "objects.csv")
                    in_target = (
                        n_objects is not None
                        and args.target_min <= n_objects <= args.target_max
                    )
                    rows.append(
                        {
                            "run_name": run_name,
                            "model": model,
                            "z_projection": z_projection,
                            "channel": channel,
                            "diameter": diameter,
                            "returncode": result.returncode,
                            "n_objects": "" if n_objects is None else n_objects,
                            "in_target": in_target,
                            "bundle_dir": str(bundle_dir),
                            "preview_png": str(preview_png),
                        }
                    )

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "run_name",
                "model",
                "z_projection",
                "channel",
                "diameter",
                "returncode",
                "n_objects",
                "in_target",
                "bundle_dir",
                "preview_png",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    hits = [row for row in rows if row["in_target"]]
    print(f"\nWrote {summary_path}")
    if hits:
        print(f"Runs in target range [{args.target_min}, {args.target_max}]:")
        for row in hits:
            print(
                f"  {row['run_name']}: {row['n_objects']} objects -> {row['bundle_dir']}"
            )
    else:
        print(f"No runs landed in target range [{args.target_min}, {args.target_max}].")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
