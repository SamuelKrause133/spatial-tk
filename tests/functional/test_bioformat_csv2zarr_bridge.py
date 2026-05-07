"""Functional coverage for the OIR -> csv2zarr bridge."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _python_from_env(env_var: str, default_relative: str) -> Path:
    override = os.getenv(env_var)
    return Path(override).expanduser().resolve() if override else _repo_root() / default_relative


def _image_env(base_env: dict[str, str], image_python: Path) -> dict[str, str]:
    env = dict(base_env)
    prefix = image_python.parent.parent
    java_home = os.getenv("JAVA_HOME") or str(prefix / "lib" / "jvm")
    env["JAVA_HOME"] = java_home
    env["PATH"] = f"{Path(java_home) / 'bin'}:{env.get('PATH', '')}"
    return env


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, env=env)


def _object_count(objects_csv: Path) -> int:
    with objects_csv.open("r", encoding="utf-8", newline="") as fh:
        return sum(1 for _ in csv.DictReader(fh))


@pytest.mark.slow
@pytest.mark.functional_full
def test_import_bioformat_oir_bundle_loads_with_csv2zarr():
    """Read the local OIR fixture in the image env, then assemble/read SpatialData in analysis."""
    repo = _repo_root()
    oir_path = Path(os.getenv("SPATIAL_TK_TEST_OIR", repo / "tests" / "test_data" / "test.oir"))
    image_python = _python_from_env("SPATIAL_TK_IMAGE_PYTHON", "venv_image/bin/python")
    analysis_python = _python_from_env("SPATIAL_TK_ANALYSIS_PYTHON", "venv_analysis/bin/python")

    if not oir_path.exists():
        pytest.skip(f"OIR fixture not found: {oir_path}")
    if not image_python.exists():
        pytest.skip(f"image environment python not found: {image_python}")
    if not analysis_python.exists():
        pytest.skip(f"analysis environment python not found: {analysis_python}")

    artifact_dir = repo / "tests" / "functional" / "csv2zarr_artifacts"
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)

    export_dir = artifact_dir / "oir_bridge"
    output_zarr = artifact_dir / "oir_bridge.zarr"
    base_env = dict(os.environ)
    image_env = _image_env(base_env, image_python)

    import_result = _run(
        [
            str(image_python),
            "-m",
            "spatial_tk.cli",
            "image",
            "import-bioformat",
            "--input",
            str(oir_path),
            "--segment",
            "--channels",
            "2",
            "--segment-diameter",
            "20",
            "--export-dir",
            str(export_dir),
            "--image-key",
            "bioformat_image",
            "--labels-key",
            "test_labels",
            "--shapes-key",
            "test_polygons",
            "--table-key",
            "table",
        ],
        env=image_env,
    )
    assert import_result.returncode == 0, import_result.stderr
    for name in [
        "image.npy",
        "labels.npy",
        "objects.csv",
        "polygons.geojson",
        "segmentation_mask.png",
        "metadata.json",
    ]:
        assert (export_dir / name).exists(), f"missing bridge asset {name}"
    assert 10 <= _object_count(export_dir / "objects.csv") <= 50

    csv2zarr_result = _run(
        [
            str(analysis_python),
            "-m",
            "spatial_tk.cli",
            "csv2zarr",
            "--metadata-json",
            str(export_dir / "metadata.json"),
            "--output",
            str(output_zarr),
        ],
        env=base_env,
    )
    assert csv2zarr_result.returncode == 0, csv2zarr_result.stderr
    assert output_zarr.exists()

    read_result = _run(
        [
            str(analysis_python),
            "-c",
            (
                "from pathlib import Path\n"
                "import spatialdata as sd\n"
                "sdata = sd.read_zarr(Path(__import__('sys').argv[1]))\n"
                "assert 'bioformat_image' in sdata.images\n"
                "assert 'test_labels' in sdata.labels\n"
                "assert 'test_polygons' in sdata.shapes\n"
                "table = sdata.tables['table']\n"
                "assert 10 <= table.n_obs <= 50\n"
                "assert table.n_vars > 0\n"
                "assert 'spatial' in table.obsm\n"
            ),
            str(output_zarr),
        ],
        env=base_env,
    )
    assert read_result.returncode == 0, read_result.stderr
