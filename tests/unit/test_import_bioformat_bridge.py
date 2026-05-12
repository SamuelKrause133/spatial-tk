"""Image-side unit tests: bridge bundle layout and batch manifest parsing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _write_bundle(tmp_path: Path, name: str = "bridge") -> Path:
    from spatial_tk.commands.import_bioformat import _write_export_bundle

    cyx = np.zeros((2, 8, 8), dtype=np.float32)
    cyx[0, 1:4, 1:4] = 3.0
    cyx[1, 4:7, 4:7] = 5.0
    labels = np.zeros((8, 8), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[4:7, 4:7] = 2

    export_dir = tmp_path / name
    _write_export_bundle(
        cyx=cyx,
        labels=labels,
        export_dir=export_dir,
        image_key="image",
        labels_key="labels",
        shapes_key="polygons",
        table_key="table",
        coord_system="global",
        source_path=tmp_path / "source.tif",
    )
    return export_dir


def test_import_bioformat_export_bundle_contract(tmp_path):
    export_dir = _write_bundle(tmp_path)

    assert (export_dir / "image.npy").exists()
    assert (export_dir / "labels.npy").exists()
    assert (export_dir / "objects.csv").exists()
    assert (export_dir / "polygons.geojson").exists()
    assert (export_dir / "segmentation_mask.png").exists()
    assert (export_dir / "metadata.json").exists()

    df = pd.read_csv(export_dir / "objects.csv")
    assert {"instance_id", "region", "centroid_x", "centroid_y", "mean_ch0", "sum_ch1"}.issubset(
        df.columns
    )
    metadata = json.loads((export_dir / "metadata.json").read_text())
    assert metadata["image"]["path"] == "image.npy"
    assert metadata["labels"]["path"] == "labels.npy"
    assert metadata["labels"]["preview_png"] == "segmentation_mask.png"
    assert metadata["shapes"]["path"] == "polygons.geojson"
    assert metadata["table"]["feature_columns"]


def test_read_batch_manifest_valid(tmp_path):
    from spatial_tk.commands.import_bioformat import _read_batch_manifest

    manifest = tmp_path / "batch.csv"
    manifest.write_text("input_path,bridge_path,zarr_path\nsample.oir,out_bridge,out.zarr\n")
    df, base = _read_batch_manifest(manifest)
    assert base == manifest.parent.resolve()
    assert len(df) == 1
    assert set(df.columns) >= {"input_path", "bridge_path", "zarr_path"}


def test_read_batch_manifest_missing_required_columns(tmp_path):
    from spatial_tk.commands.import_bioformat import _read_batch_manifest

    manifest = tmp_path / "bad.csv"
    manifest.write_text("input_path,bridge_path\na.oir,bridge1\n")
    with pytest.raises(ValueError, match="missing required columns"):
        _read_batch_manifest(manifest)


def test_read_batch_manifest_empty_rows(tmp_path):
    from spatial_tk.commands.import_bioformat import _read_batch_manifest

    manifest = tmp_path / "empty.csv"
    manifest.write_text("input_path,bridge_path,zarr_path\n")
    with pytest.raises(ValueError, match="at least one row"):
        _read_batch_manifest(manifest)
