#!/usr/bin/env python3
"""Segment cells/nuclei (Cellpose) on an imported SpatialData zarr."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from spatial_tk.core.data_io import save_spatial_data
from spatial_tk.core.imaging_common import (
    first_image_key,
    image_to_numpy_cyx,
    read_sdata,
    ensure_obs_from_labels,
)
from spatial_tk.utils.config import load_config, merge_config_with_args
from spatial_tk.utils.helpers import setup_logging

CLI_HELP = "Segment objects with Cellpose"
CLI_DESCRIPTION = (
    "Deprecated compatibility command. Prefer `spatial-tk image import-bioformat "
    "--segment --export-dir ...` for the split image/analysis workflow."
)

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=False, help="SpatialData .zarr from import-bioformat")
    parser.add_argument("--inplace", action="store_true", help="Overwrite the input zarr store")
    parser.add_argument("--output", required=False, help="Output .zarr (if not inplace)")
    parser.add_argument("--image-key", default=None, help="SpatialData.images key (default: first image)")
    parser.add_argument("--mode", default="cellpose", choices=["cellpose"])
    parser.add_argument(
        "--channels",
        default="0",
        help="Comma-separated image channel index/indices for Cellpose (e.g. 0 or 1,2)",
    )
    parser.add_argument(
        "--segment-model",
        default="nuclei",
        choices=["nuclei", "cyto", "cyto2"],
        help="Cellpose model_type",
    )
    parser.add_argument(
        "--labels-key",
        default="nuclei_labels",
        help="SpatialData.labels key for segmentation output",
    )
    parser.add_argument("--config", help="TOML configuration (optional)")


def main(args: argparse.Namespace) -> None:
    setup_logging()
    if args.config:
        tmp = argparse.ArgumentParser()
        add_arguments(tmp)
        try:
            cfg = load_config(args.config)
            args = merge_config_with_args("image_segment", cfg, args, tmp)
        except Exception as e:
            logging.error("Config error: %s", e)
            sys.exit(1)

    if not args.input:
        logging.error("--input is required")
        sys.exit(1)

    inp = Path(args.input).expanduser().resolve()
    sdata = read_sdata(inp)
    im_key = args.image_key or first_image_key(sdata)
    cyx = image_to_numpy_cyx(sdata.images[im_key])
    logger.warning(
        "`spatial-tk image segment` is deprecated for the split-env workflow; "
        "prefer `spatial-tk image import-bioformat --segment --export-dir ...`."
    )

    ch_parts = [int(x.strip()) for x in str(args.channels).split(",") if x.strip()]
    if not ch_parts:
        logging.error("Invalid --channels")
        sys.exit(1)

    if args.mode != "cellpose":
        raise NotImplementedError(args.mode)

    from spatial_tk.commands.import_bioformat import _segment_cellpose

    masks = _segment_cellpose(cyx, args.channels, args.segment_model)

    from spatialdata.models import Labels2DModel
    from spatialdata.transformations import Identity

    labels_el = Labels2DModel.parse(
        masks.astype(np.int32),
        dims=("y", "x"),
        transformations={"global": Identity()},
    )
    # Merge into labels dict (SpatialData.labels is mapping-like)
    prev = getattr(sdata, "labels", None)
    merged = dict(prev) if prev is not None else {}
    merged[args.labels_key] = labels_el
    sdata.labels = merged

    ensure_obs_from_labels(sdata, args.labels_key, masks.astype(np.int32))

    out_path = inp if args.inplace else Path(args.output or inp)
    if not args.inplace:
        if not args.output:
            logging.error("Provide --output or use --inplace")
            sys.exit(1)
        out_path = Path(args.output).expanduser().resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_spatial_data(sdata, out_path, overwrite=out_path.exists())
    logger.info("Saved segmentation to %s (labels[%s])", out_path, args.labels_key)
