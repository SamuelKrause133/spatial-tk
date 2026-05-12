import argparse
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


def test_csv2zarr_validates_missing_metadata_asset(tmp_path):
    from spatial_tk.commands.csv2zarr import main

    export_dir = _write_bundle(tmp_path)
    (export_dir / "labels.npy").unlink()

    with pytest.raises(SystemExit) as excinfo:
        main(
            argparse.Namespace(
                table_csv=None,
                metadata_json=str(export_dir / "metadata.json"),
                output=str(tmp_path / "out.zarr"),
                table_key=None,
                image_key=None,
                labels_key=None,
                shapes_key=None,
                coord_system=None,
                config=None,
            )
        )
    assert excinfo.value.code == 1


def test_csv2zarr_validates_geojson_table_id_mismatch(tmp_path):
    from spatial_tk.commands.csv2zarr import _validate_geojson_ids

    geojson = tmp_path / "polygons.geojson"
    geojson.write_text(
        json.dumps(
            {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "properties": {"instance_id": 1},
                        "geometry": {
                            "type": "Polygon",
                            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
                        },
                    }
                ],
            }
        )
    )

    with pytest.raises(ValueError, match="absent from GeoJSON"):
        _validate_geojson_ids(geojson, {1, 2}, "instance_id")


def test_csv2zarr_writes_spatialdata_zarr(tmp_path):
    pytest.importorskip("spatialdata")
    pytest.importorskip("geopandas")

    import spatialdata as sd
    from spatial_tk.commands.csv2zarr import main

    export_dir = _write_bundle(tmp_path)
    output = tmp_path / "out.zarr"

    main(
        argparse.Namespace(
            table_csv=None,
            metadata_json=str(export_dir / "metadata.json"),
            output=str(output),
            table_key=None,
            image_key=None,
            labels_key=None,
            shapes_key=None,
            coord_system=None,
            config=None,
        )
    )

    sdata = sd.read_zarr(output)
    assert "image" in sdata.images
    assert "labels" in sdata.labels
    assert "polygons" in sdata.shapes
    assert "table" in sdata.tables
    table = sdata.tables["table"]
    assert table.n_obs == 2
    assert table.n_vars > 0
    assert "spatial" in table.obsm


def test_csv2zarr_batch_validates_required_columns(tmp_path):
    from spatial_tk.commands.csv2zarr import main

    batch_csv = tmp_path / "batch.csv"
    pd.DataFrame(
        [
            {
                "input_path": tmp_path / "source.tif",
                "bridge_path": tmp_path / "bridge",
            }
        ]
    ).to_csv(batch_csv, index=False)

    with pytest.raises(SystemExit) as excinfo:
        main(
            argparse.Namespace(
                table_csv=None,
                metadata_json=None,
                output=None,
                batch_csv=str(batch_csv),
                table_key=None,
                image_key=None,
                labels_key=None,
                shapes_key=None,
                coord_system=None,
                config=None,
            )
        )
    assert excinfo.value.code == 1


def test_csv2zarr_batch_writes_multiple_spatialdata_zarrs(tmp_path):
    pytest.importorskip("spatialdata")
    pytest.importorskip("geopandas")

    import spatialdata as sd
    from spatial_tk.commands.csv2zarr import main

    rows = []
    for idx in range(2):
        bridge = _write_bundle(tmp_path, f"bridge_{idx}")
        rows.append(
            {
                "input_path": tmp_path / f"source_{idx}.tif",
                "bridge_path": bridge,
                "zarr_path": tmp_path / f"out_{idx}.zarr",
            }
        )
    batch_csv = tmp_path / "batch.csv"
    pd.DataFrame(rows).to_csv(batch_csv, index=False)

    main(
        argparse.Namespace(
            table_csv=None,
            metadata_json=None,
            output=None,
            batch_csv=str(batch_csv),
            table_key=None,
            image_key=None,
            labels_key=None,
            shapes_key=None,
            coord_system=None,
            config=None,
        )
    )

    for row in rows:
        output = Path(row["zarr_path"])
        assert output.exists()
        sdata = sd.read_zarr(output)
        assert "image" in sdata.images
        assert "labels" in sdata.labels
        assert "polygons" in sdata.shapes
        assert "table" in sdata.tables

