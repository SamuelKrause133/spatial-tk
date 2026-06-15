#!/usr/bin/env python3
"""Build a SpatialData zarr from an image-side CSV/metadata export bundle."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from spatial_tk.utils.batch_csv import read_batch_manifest, resolve_manifest_path
from spatial_tk.utils.config import load_config, merge_config_with_args
from spatial_tk.utils.helpers import prepare_spatial_data_for_save, setup_logging

CLI_HELP = "Convert image-side CSV export bundle to SpatialData zarr"
CLI_DESCRIPTION = (
    "Consume objects.csv + metadata.json + referenced image/labels/polygons assets "
    "and write an analysis-compatible SpatialData .zarr."
)

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--table-csv", required=False, help="Object table CSV from image export")
    parser.add_argument("--metadata-json", required=False, help="metadata.json from image export")
    parser.add_argument("--output", required=False, help="Output SpatialData .zarr directory")
    parser.add_argument(
        "--batch-csv",
        required=False,
        help="CSV with bridge_path and zarr_path columns (other columns allowed) for batch conversion.",
    )
    parser.add_argument("--table-key", default=None, help="Override table key")
    parser.add_argument("--image-key", default=None, help="Override image key")
    parser.add_argument("--labels-key", default=None, help="Override labels key")
    parser.add_argument("--shapes-key", default=None, help="Override shapes key")
    parser.add_argument("--coord-system", default=None, help="Override coordinate system")
    parser.add_argument("--config", help="TOML configuration (optional)")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("metadata JSON must contain an object")
    return data


def _resolve(base: Path, value: str | None, label: str) -> Path:
    if not value:
        raise ValueError(f"metadata missing required path: {label}")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def _meta_section(meta: dict[str, Any], name: str) -> dict[str, Any]:
    section = meta.get(name)
    if not isinstance(section, dict):
        raise ValueError(f"metadata missing required object: {name}")
    return section


def _require_columns(df: pd.DataFrame, required: set[str]) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"table CSV missing required columns: {sorted(missing)}")


def _feature_columns(df: pd.DataFrame, meta: dict[str, Any]) -> list[str]:
    table_meta = _meta_section(meta, "table")
    configured = table_meta.get("feature_columns")
    if configured:
        cols = [str(c) for c in configured]
    else:
        reserved = {"label_id", "instance_id", "region", "centroid_x", "centroid_y", "area_px"}
        cols = [
            c
            for c in df.columns
            if c not in reserved and pd.api.types.is_numeric_dtype(df[c])
        ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"feature columns missing from table CSV: {missing}")
    if not cols:
        raise ValueError("no feature columns available for AnnData.X")
    return cols


def _validate_table(df: pd.DataFrame, meta: dict[str, Any], labels_key: str, shapes_key: str) -> None:
    _require_columns(df, {"instance_id", "region", "centroid_x", "centroid_y"})
    if df["instance_id"].duplicated().any():
        dup = df.loc[df["instance_id"].duplicated(), "instance_id"].head().tolist()
        raise ValueError(f"instance_id values must be unique; duplicates include {dup}")
    for col in ["centroid_x", "centroid_y"]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"{col} must be numeric")
    allowed_regions = {labels_key, shapes_key}
    actual_regions = set(df["region"].astype(str))
    if not actual_regions <= allowed_regions:
        raise ValueError(
            f"region values must match labels_key/shapes_key {sorted(allowed_regions)}; got {sorted(actual_regions)}"
        )


def _validate_geojson_ids(path: Path, table_ids: set[int], id_property: str) -> None:
    data = _read_json(path)
    features = data.get("features", [])
    if not isinstance(features, list):
        raise ValueError("GeoJSON must contain a features list")
    ids: set[int] = set()
    for feature in features:
        props = feature.get("properties", {}) if isinstance(feature, dict) else {}
        if id_property not in props:
            raise ValueError(f"GeoJSON feature missing property {id_property!r}")
        ids.add(int(props[id_property]))
    extra_geojson_ids = ids - table_ids
    if extra_geojson_ids:
        raise ValueError(f"GeoJSON contains IDs absent from table: {sorted(extra_geojson_ids)[:10]}")
    missing_geojson_ids = table_ids - ids
    if missing_geojson_ids:
        raise ValueError(f"table contains IDs absent from GeoJSON: {sorted(missing_geojson_ids)[:10]}")


def _build_anndata(df: pd.DataFrame, feature_cols: list[str]) -> Any:
    import anndata as ad

    obs = df.drop(columns=feature_cols).copy()
    X = df[feature_cols].to_numpy(dtype=np.float32)
    adata = ad.AnnData(X=X, obs=obs)
    adata.var_names = feature_cols
    adata.obsm["spatial"] = df[["centroid_x", "centroid_y"]].to_numpy(dtype=np.float32)
    prepare_spatial_data_for_save(adata)
    return adata


def _build_spatialdata(
    *,
    image: np.ndarray,
    labels: np.ndarray,
    geojson_path: Path,
    adata: Any,
    image_key: str,
    labels_key: str,
    shapes_key: str,
    table_key: str,
    coord_system: str,
) -> Any:
    import geopandas as gpd
    import spatialdata as sd
    from spatialdata.models import Image2DModel, Labels2DModel, ShapesModel
    from spatialdata.transformations import Identity

    tr = {coord_system: Identity()}
    image_chunks = (
        int(image.shape[0]),
        min(1024, int(image.shape[1])),
        min(1024, int(image.shape[2])),
    )
    label_chunks = (
        min(1024, int(labels.shape[0])),
        min(1024, int(labels.shape[1])),
    )
    images = {
        image_key: Image2DModel.parse(
            image.astype(np.float32, copy=False),
            dims=("c", "y", "x"),
            transformations=tr,
            scale_factors=[],
            chunks=image_chunks,
        )
    }
    label_elements = {
        labels_key: Labels2DModel.parse(
            labels.astype(np.int32, copy=False),
            dims=("y", "x"),
            transformations=tr,
            scale_factors=[],
            chunks=label_chunks,
        )
    }
    gdf = gpd.read_file(geojson_path)
    shapes = {shapes_key: ShapesModel.parse(gdf, transformations=tr)}
    return sd.SpatialData(images=images, labels=label_elements, shapes=shapes, tables={table_key: adata})


def _convert_one(
    args: argparse.Namespace,
    metadata_json: Path,
    output: Path,
    table_csv: Path | None = None,
) -> None:
    meta_path = Path(metadata_json).expanduser().resolve()
    meta = _read_json(meta_path)
    base = meta_path.parent

    table_meta = _meta_section(meta, "table")
    image_meta = _meta_section(meta, "image")
    labels_meta = _meta_section(meta, "labels")
    shapes_meta = _meta_section(meta, "shapes")

    table_path = (
        Path(table_csv).expanduser().resolve()
        if table_csv
        else _resolve(base, table_meta.get("path"), "table.path")
    )
    image_path = _resolve(base, image_meta.get("path"), "image.path")
    labels_path = _resolve(base, labels_meta.get("path"), "labels.path")
    shapes_path = _resolve(base, shapes_meta.get("path"), "shapes.path")

    table_key = args.table_key or str(table_meta.get("key", "table"))
    image_key = args.image_key or str(image_meta.get("key", "bioformat_image"))
    labels_key = args.labels_key or str(labels_meta.get("key", "nuclei_labels"))
    shapes_key = args.shapes_key or str(shapes_meta.get("key", "nuclei_polygons"))
    coord_system = args.coord_system or str(meta.get("coordinate_system", "global"))
    id_property = str(shapes_meta.get("id_property", "instance_id"))

    df = pd.read_csv(table_path)
    feature_cols = _feature_columns(df, meta)
    _validate_table(df, meta, labels_key, shapes_key)
    _validate_geojson_ids(shapes_path, {int(x) for x in df["instance_id"]}, id_property)

    image = np.load(image_path)
    labels = np.load(labels_path)
    if image.ndim != 3:
        raise ValueError(f"image asset must be CYX, got shape {image.shape}")
    if labels.ndim != 2:
        raise ValueError(f"labels asset must be YX, got shape {labels.shape}")

    adata = _build_anndata(df, feature_cols)
    sdata = _build_spatialdata(
        image=image,
        labels=labels,
        geojson_path=shapes_path,
        adata=adata,
        image_key=image_key,
        labels_key=labels_key,
        shapes_key=shapes_key,
        table_key=table_key,
        coord_system=coord_system,
    )

    outp = Path(output).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)
    from spatial_tk.core.data_io import save_spatial_data

    save_spatial_data(sdata, outp, overwrite=outp.exists())
    logger.info("Wrote SpatialData zarr to %s", outp)


def main(args: argparse.Namespace) -> None:
    setup_logging()
    if args.config:
        tmp = argparse.ArgumentParser()
        add_arguments(tmp)
        try:
            cfg = load_config(args.config)
            args = merge_config_with_args("csv2zarr", cfg, args, tmp)
        except Exception as e:
            logging.error("Config error: %s", e)
            sys.exit(1)

    try:
        batch_csv = getattr(args, "batch_csv", None)
        if batch_csv:
            rows, base = read_batch_manifest(Path(batch_csv), required={"bridge_path", "zarr_path"})
            for idx, row in enumerate(rows):
                bridge_path = resolve_manifest_path(base, row["bridge_path"])
                zarr_path = resolve_manifest_path(base, row["zarr_path"])
                metadata_json = bridge_path / "metadata.json"
                logger.info("Processing batch row %s: %s -> %s", idx, bridge_path, zarr_path)
                _convert_one(args, metadata_json, zarr_path)
            return

        if not args.metadata_json or not args.output:
            logging.error("--metadata-json and --output are required")
            sys.exit(1)

        table_csv = Path(args.table_csv).expanduser().resolve() if args.table_csv else None
        _convert_one(
            args,
            Path(args.metadata_json).expanduser().resolve(),
            Path(args.output).expanduser().resolve(),
            table_csv=table_csv,
        )
    except Exception as e:
        logging.error("csv2zarr failed: %s", e)
        sys.exit(1)

