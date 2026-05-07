#!/usr/bin/env python3
"""
Import microscopy files and optionally export an image-side bridge bundle.

The bridge bundle is intentionally flat (NumPy arrays + CSV/GeoJSON/JSON) so
the image environment does not need to assemble AnnData/SpatialData directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, List

import numpy as np

from spatial_tk.utils.config import load_config, merge_config_with_args
from spatial_tk.utils.helpers import setup_logging

CLI_HELP = "Import microscopy files using Bio-Formats"
CLI_DESCRIPTION = (
    "Read OIR/OME-TIFF and write a flat csv2zarr bridge bundle with image, "
    "labels, intensities, polygons, and metadata."
)

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=False, help="Input microscopy file (.oir, .ome.tif, …)")
    parser.add_argument(
        "--export-dir",
        required=False,
        help="Directory for csv2zarr bridge bundle (image.npy, labels.npy, objects.csv, polygons.geojson, metadata.json).",
    )
    parser.add_argument(
        "--dtype",
        default="float32",
        choices=["float32", "float64"],
        help="Array dtype for stored image (default float32)",
    )
    parser.add_argument(
        "--image-key",
        default="bioformat_image",
        help="Image key to record in metadata",
    )
    parser.add_argument(
        "--labels-key",
        default="nuclei_labels",
        help="Labels key to record in metadata",
    )
    parser.add_argument(
        "--shapes-key",
        default="nuclei_polygons",
        help="Shapes key to record in metadata",
    )
    parser.add_argument(
        "--table-key",
        default="table",
        help="Table key to record in metadata",
    )
    parser.add_argument(
        "--coord-system",
        default="global",
        help="Coordinate system name recorded in metadata (default: global)",
    )
    parser.add_argument(
        "--z-projection",
        default="max",
        choices=["max", "middle"],
        help="How to collapse Z when the reader yields a 3D stack per channel",
    )
    parser.add_argument(
        "--preview-png",
        default=None,
        help="Optional path to write a quick PNG (first channel, after projection)",
    )
    parser.add_argument(
        "--segment",
        action="store_true",
        help="Run Cellpose segmentation during import before writing the export bundle.",
    )
    parser.add_argument(
        "--labels-npy",
        default=None,
        help="Optional existing YX label mask (.npy) to use instead of running Cellpose.",
    )
    parser.add_argument(
        "--channels",
        default="0",
        help="Comma-separated channel index/indices for Cellpose (first index used; default 0).",
    )
    parser.add_argument(
        "--segment-model",
        default="nuclei",
        choices=["nuclei", "cyto", "cyto2"],
        help="Cellpose model_type when --segment is used.",
    )
    parser.add_argument(
        "--segment-diameter",
        type=float,
        default=None,
        help="Expected object diameter in pixels for Cellpose. Defaults to Cellpose auto-estimation.",
    )
    parser.add_argument(
        "--segment-gpu",
        action="store_true",
        help="Use GPU acceleration for Cellpose segmentation when available.",
    )
    parser.add_argument("--config", help="TOML configuration (optional)")


def _dtype(name: str) -> type:
    return np.float32 if name == "float32" else np.float64


def _read_tiff(path: Path, dt: type) -> np.ndarray:
    import tifffile

    arr = tifffile.imread(path)
    arr = np.asarray(arr, dtype=dt)
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        # CYX or ZYX — if first dim small, treat as C
        if arr.shape[0] <= 32 and arr.shape[0] < min(arr.shape[1], arr.shape[2]):
            return arr
        return np.max(arr, axis=0, keepdims=False)[np.newaxis, ...]
    raise ValueError(f"Unsupported TIFF shape {arr.shape}")


def _read_bioformats_javabridge(path: Path, dt: type, z_mode: str) -> np.ndarray:
    """Java/Bio-Formats via python-bioformats + javabridge (no PIMS / JPype)."""
    import javabridge
    import bioformats
    from bioformats import ImageReader

    javabridge.start_vm(class_path=bioformats.JARS, run_headless=True, max_heap_size="4096m")
    try:
        z_use_max = z_mode == "max"
        with ImageReader(path=str(path)) as reader:
            reader.rdr.setSeries(0)
            nz = reader.rdr.getSizeZ()
            nc = reader.rdr.getSizeC()

            def read_plane(z: int, c: int) -> np.ndarray:
                im = reader.read(c=c, z=z, t=0, rescale=False)
                if im.ndim == 3:
                    if im.shape[2] <= 8:
                        return im[..., 0].astype(np.float32)
                    raise ValueError(f"Unsupported multichannel plane shape {im.shape}")
                return im.astype(np.float32)

            z_planes: List[np.ndarray] = []
            for zi in range(nz):
                chans = [read_plane(zi, ci) for ci in range(nc)]
                z_planes.append(np.stack(chans, axis=0))

            if nz == 0:
                raise ValueError("Bio-Formats reported zero Z planes")

            cyx_vol: np.ndarray
            if nz == 1:
                cyx_vol = z_planes[0]
            elif z_use_max:
                vol_czyx = np.stack(z_planes, axis=1)
                cyx_vol = np.max(vol_czyx, axis=1)
            else:
                vol_czyx = np.stack(z_planes, axis=1)
                cyx_vol = vol_czyx[:, nz // 2, ...]

        return cyx_vol.astype(dt, copy=False)
    finally:
        javabridge.kill_vm()


def read_to_cyx(path: Path, dtype: type, z_projection: str) -> np.ndarray:
    suf = path.suffix.lower()
    if suf in (".tif", ".tiff"):
        return _read_tiff(path, dtype)
    return _read_bioformats_javabridge(path, dtype, z_projection)


def _segment_cellpose(
    cyx: np.ndarray,
    channels: str,
    model_type: str,
    diameter: float | None = None,
    gpu: bool = False,
) -> np.ndarray:
    from cellpose import models

    ch_parts = [int(x.strip()) for x in str(channels).split(",") if x.strip()]
    if not ch_parts:
        raise ValueError("Invalid --channels; expected at least one channel index")
    idx = ch_parts[0]
    if idx < 0 or idx >= cyx.shape[0]:
        raise ValueError(f"Channel index {idx} out of bounds for image with {cyx.shape[0]} channels")

    plane = np.asarray(cyx[idx])
    model = models.Cellpose(model_type=model_type, gpu=gpu)
    masks, *_ = model.eval(plane, channels=[0, 0], diameter=diameter)
    return masks.astype(np.int32, copy=False)


def _load_or_segment_labels(args: argparse.Namespace, cyx: np.ndarray) -> np.ndarray | None:
    if args.labels_npy:
        labels = np.load(Path(args.labels_npy).expanduser().resolve())
        if labels.ndim != 2:
            raise ValueError(f"--labels-npy must contain a 2D YX mask, got shape {labels.shape}")
        return labels.astype(np.int32, copy=False)
    if args.segment:
        return _segment_cellpose(
            cyx,
            args.channels,
            args.segment_model,
            diameter=args.segment_diameter,
            gpu=args.segment_gpu,
        )
    return None


def _quantify_labels(cyx: np.ndarray, labels: np.ndarray, region: str) -> tuple[Any, list[str]]:
    import pandas as pd
    from skimage.measure import regionprops

    rows = []
    for r in regionprops(labels.astype(np.int32)):
        lid = int(r.label)
        mask = labels == lid
        row = {
            "label_id": lid,
            "instance_id": lid,
            "region": region,
            "centroid_y": float(r.centroid[0]),
            "centroid_x": float(r.centroid[1]),
            "area_px": int(r.area),
        }
        for c in range(cyx.shape[0]):
            vals = cyx[c][mask]
            row[f"mean_ch{c}"] = float(np.mean(vals))
            row[f"sum_ch{c}"] = float(np.sum(vals))
            row[f"max_ch{c}"] = float(np.max(vals))
        rows.append(row)

    df = pd.DataFrame(rows)
    feature_cols = [c for c in df.columns if c.startswith(("mean_ch", "sum_ch", "max_ch"))]
    return df, feature_cols


def _polygon_for_label(labels: np.ndarray, label_id: int) -> list[list[float]] | None:
    from skimage.measure import find_contours

    contours = find_contours((labels == label_id).astype(np.uint8), level=0.5)
    if not contours:
        return None
    contour = max(contours, key=len)
    if len(contour) < 3:
        return None
    coords = [[float(x), float(y)] for y, x in contour]
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def _write_polygons_geojson(labels: np.ndarray, path: Path) -> None:
    features = []
    for lid in [int(x) for x in np.unique(labels) if x > 0]:
        coords = _polygon_for_label(labels, lid)
        if not coords:
            continue
        features.append(
            {
                "type": "Feature",
                "properties": {"label_id": lid, "instance_id": lid},
                "geometry": {"type": "Polygon", "coordinates": [coords]},
            }
        )
    payload = {"type": "FeatureCollection", "features": features}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_segmentation_mask_png(labels: np.ndarray, path: Path) -> None:
    import matplotlib.pyplot as plt

    # Use a qualitative colormap so neighboring labels are distinguishable by eye.
    plt.imsave(path, labels.astype(np.int32, copy=False), cmap="nipy_spectral")


def _write_export_bundle(
    *,
    cyx: np.ndarray,
    labels: np.ndarray,
    export_dir: Path,
    image_key: str,
    labels_key: str,
    shapes_key: str,
    table_key: str,
    coord_system: str,
    source_path: Path,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)

    image_path = export_dir / "image.npy"
    labels_path = export_dir / "labels.npy"
    table_path = export_dir / "objects.csv"
    shapes_path = export_dir / "polygons.geojson"
    mask_png_path = export_dir / "segmentation_mask.png"
    metadata_path = export_dir / "metadata.json"

    np.save(image_path, cyx)
    np.save(labels_path, labels.astype(np.int32, copy=False))

    table, feature_cols = _quantify_labels(cyx, labels, labels_key)
    table.to_csv(table_path, index=False)
    _write_polygons_geojson(labels, shapes_path)
    _write_segmentation_mask_png(labels, mask_png_path)

    metadata = {
        "version": 1,
        "source": str(source_path),
        "coordinate_system": coord_system,
        "table": {
            "path": table_path.name,
            "key": table_key,
            "feature_columns": feature_cols,
        },
        "image": {
            "path": image_path.name,
            "key": image_key,
            "dims": ["c", "y", "x"],
            "dtype": str(cyx.dtype),
        },
        "labels": {
            "path": labels_path.name,
            "preview_png": mask_png_path.name,
            "key": labels_key,
            "dims": ["y", "x"],
            "dtype": "int32",
        },
        "shapes": {
            "path": shapes_path.name,
            "key": shapes_key,
            "id_property": "instance_id",
            "format": "geojson",
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Wrote csv2zarr export bundle to %s", export_dir)


def main(args: argparse.Namespace) -> None:
    setup_logging()
    if args.config:
        temp = argparse.ArgumentParser()
        add_arguments(temp)
        try:
            cfg = load_config(args.config)
            args = merge_config_with_args("import_bioformat", cfg, args, temp)
        except Exception as e:
            logging.error("Config error: %s", e)
            sys.exit(1)

    if not args.input:
        logging.error("--input is required")
        sys.exit(1)
    if not args.export_dir:
        logging.error("--export-dir is required")
        sys.exit(1)

    inp = Path(args.input).expanduser().resolve()
    dt = _dtype(args.dtype)

    logger.info("Reading %s", inp)
    try:
        cyx = read_to_cyx(inp, dt, args.z_projection)
    except ImportError as e:
        logger.error(
            "Missing optional image dependency (javabridge/bioformats/pims). Use venv_image. %s", e
        )
        sys.exit(1)

    logger.info("CYX array shape=%s dtype=%s", cyx.shape, cyx.dtype)
    if args.preview_png:
        import matplotlib.pyplot as plt

        ch0 = np.asarray(cyx[0, ...])
        ch0n = (ch0 - ch0.min()) / (np.ptp(ch0) + 1e-9)
        plt.imsave(args.preview_png, ch0n, cmap="gray")
        logger.info("Wrote preview %s", args.preview_png)

    labels = None
    try:
        labels = _load_or_segment_labels(args, cyx)
    except ImportError as e:
        logger.error("Missing optional segmentation dependency (cellpose). Use venv_image. %s", e)
        sys.exit(1)

    if labels is None:
        logging.error("--export-dir requires --segment or --labels-npy so labels can be exported")
        sys.exit(1)
    _write_export_bundle(
        cyx=cyx,
        labels=labels,
        export_dir=Path(args.export_dir).expanduser().resolve(),
        image_key=args.image_key,
        labels_key=args.labels_key,
        shapes_key=args.shapes_key,
        table_key=args.table_key,
        coord_system=args.coord_system,
        source_path=inp,
    )