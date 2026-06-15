"""Batch chip extraction from a manifest CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
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


def test_extract_batch_csv_writes_chips_npz_and_chip_montage(tmp_path):
    pytest.importorskip("spatialdata")
    pytest.importorskip("geopandas")

    from spatial_tk.commands.csv2zarr import main as csv2zarr_main
    from spatial_tk.commands.extract import add_arguments, main as extract_main

    export_dir = _write_bundle(tmp_path, "bridge")
    out_zarr = tmp_path / "sample.zarr"
    csv2zarr_main(
        argparse.Namespace(
            table_csv=None,
            metadata_json=str(export_dir / "metadata.json"),
            output=str(out_zarr),
            batch_csv=None,
            table_key=None,
            image_key=None,
            labels_key=None,
            shapes_key=None,
            coord_system=None,
            config=None,
        )
    )

    extract_out = tmp_path / "chip_out"
    batch_csv = tmp_path / "manifest.csv"
    batch_csv.write_text(
        "zarr_path,extract_path,input_path,note\n"
        f"sample.zarr,chip_out,unused.tif,batch_extract_test\n",
        encoding="utf-8",
    )

    p = argparse.ArgumentParser()
    add_arguments(p)
    args = p.parse_args(
        [
            "--batch-csv",
            str(batch_csv),
            "--labels-key",
            "labels",
            "--chip-size",
            "64",
            "64",
        ]
    )
    extract_main(args)

    assert (extract_out / "chips.npz").exists()
    assert (extract_out / "chip_montage.png").exists()
    with np.load(extract_out / "chips.npz") as data:
        chips = data["chips"]
    assert chips.ndim == 4
    assert chips.shape[0] == 2
    assert chips.shape[1:3] == (8, 8)
