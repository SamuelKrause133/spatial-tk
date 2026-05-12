"""Functional coverage for the OIR -> csv2zarr bridge."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
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


def _build_oir_csv2zarr_artifacts() -> tuple[Path, Path, Path, Path, dict[str, str]]:
    repo = _repo_root()
    oir_path = Path(os.getenv("SPATIAL_TK_TEST_OIR", repo / "tests" / "test_data" / "test.oir"))
    image_python = _python_from_env("SPATIAL_TK_IMAGE_PYTHON", "venv_image/bin/python")
    analysis_python = _python_from_env("SPATIAL_TK_ANALYSIS_PYTHON", "venv/bin/python")

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
            "0",
            "--segment-diameter",
            "40",
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

    return artifact_dir, export_dir, output_zarr, analysis_python, base_env


@pytest.mark.slow
@pytest.mark.functional_full
def test_import_bioformat_oir_bundle_loads_with_csv2zarr():
    """Read the local OIR fixture in the image env, then assemble/read SpatialData in analysis."""
    _, _, output_zarr, analysis_python, base_env = _build_oir_csv2zarr_artifacts()

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


@pytest.mark.slow
@pytest.mark.functional_full
def test_analysis_extracts_nuclear_chips_from_csv2zarr_artifact():
    """Use the analysis env to crop chips from the zarr assembled from import-bioformat output."""
    artifact_dir, _, output_zarr, analysis_python, base_env = _build_oir_csv2zarr_artifacts()

    montage_png = artifact_dir / "chips_montage.png"
    chips_output_zarr = artifact_dir / "oir_bridge_chips_copy.zarr"
    extract_result = _run(
        [
            str(analysis_python),
            "-m",
            "spatial_tk.cli",
            "image",
            "extract",
            "--input",
            str(output_zarr),
            "--image-key",
            "bioformat_image",
            "--labels-key",
            "test_labels",
            "--chip-size",
            "64",
            "64",
            "--include-mask-channel",
            "--montage-png",
            str(montage_png),
            "--max-chips",
            "12",
            "--output",
            str(chips_output_zarr),
        ],
        env=base_env,
    )
    assert extract_result.returncode == 0, extract_result.stderr

    chips_npz = artifact_dir / "oir_bridge.zarr_chips" / "chips.npz"
    assert chips_npz.exists()
    assert montage_png.exists()
    assert chips_output_zarr.exists()

    with np.load(chips_npz) as data:
        chips = data["chips"]
    assert chips.ndim == 4
    assert chips.shape[0] >= 10
    assert chips.shape[1:3] == (64, 64)
    assert chips.shape[-1] == 4  # 3 image channels + mask channel


@pytest.mark.slow
@pytest.mark.functional_full
def test_batch_manifest_converts_three_oirs_to_bridges_and_zarrs():
    """Run batch CSV mode across image import and analysis csv2zarr conversion."""
    repo = _repo_root()
    oir_path = Path(os.getenv("SPATIAL_TK_TEST_OIR", repo / "tests" / "test_data" / "test.oir"))
    image_python = _python_from_env("SPATIAL_TK_IMAGE_PYTHON", "venv_image/bin/python")
    analysis_python = _python_from_env("SPATIAL_TK_ANALYSIS_PYTHON", "venv/bin/python")

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

    batch_dir = artifact_dir / "batch"
    batch_dir.mkdir()
    manifest_path = artifact_dir / "batch_manifest.csv"
    rows = []
    for idx in range(3):
        rows.append(
            {
                "input_path": str(oir_path),
                "bridge_path": str(batch_dir / f"bridge_{idx}"),
                "zarr_path": str(batch_dir / f"sample_{idx}.zarr"),
            }
        )
    with manifest_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["input_path", "bridge_path", "zarr_path"])
        writer.writeheader()
        writer.writerows(rows)

    base_env = dict(os.environ)
    image_result = _run(
        [
            str(image_python),
            "-m",
            "spatial_tk.cli",
            "image",
            "import-bioformat",
            "--batch-csv",
            str(manifest_path),
            "--segment",
            "--channels",
            "0",
            "--segment-diameter",
            "40",
            "--image-key",
            "bioformat_image",
            "--labels-key",
            "test_labels",
            "--shapes-key",
            "test_polygons",
            "--table-key",
            "table",
        ],
        env=_image_env(base_env, image_python),
    )
    assert image_result.returncode == 0, image_result.stderr

    for row in rows:
        bridge_path = Path(row["bridge_path"])
        for name in [
            "image.npy",
            "labels.npy",
            "objects.csv",
            "polygons.geojson",
            "segmentation_mask.png",
            "metadata.json",
        ]:
            assert (bridge_path / name).exists(), f"missing bridge asset {bridge_path / name}"
        assert 10 <= _object_count(bridge_path / "objects.csv") <= 50

    zarr_result = _run(
        [
            str(analysis_python),
            "-m",
            "spatial_tk.cli",
            "csv2zarr",
            "--batch-csv",
            str(manifest_path),
        ],
        env=base_env,
    )
    assert zarr_result.returncode == 0, zarr_result.stderr

    for row in rows:
        zarr_path = Path(row["zarr_path"])
        assert zarr_path.exists()
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
                    "assert 'spatial' in table.obsm\n"
                ),
                str(zarr_path),
            ],
            env=base_env,
        )
        assert read_result.returncode == 0, read_result.stderr
