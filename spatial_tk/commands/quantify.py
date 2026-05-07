#!/usr/bin/env python3
"""Per-object intensity summaries for segmented SpatialData images."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from spatial_tk.core.data_io import save_spatial_data
from spatial_tk.core.imaging_common import (
    first_image_key,
    image_to_numpy_cyx,
    labels_to_numpy,
    read_sdata,
)
from spatial_tk.utils.config import load_config, merge_config_with_args
from spatial_tk.utils.helpers import setup_logging

CLI_HELP = "Quantify channel intensities per segmented object"
CLI_DESCRIPTION = "Region summary stats for each label id and store in AnnData.obs / obsm."

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=False, help="SpatialData .zarr")
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--output", required=False)
    parser.add_argument("--image-key", default=None)
    parser.add_argument("--labels-key", default="nuclei_labels")
    parser.add_argument(
        "--table-key",
        default="table",
        help="Table name for quantification results (default: table)",
    )
    parser.add_argument("--config", help="TOML configuration (optional)")


def _quantify(
    cyx: np.ndarray,
    labels: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    from skimage.measure import regionprops

    rows = []
    for r in regionprops(labels.astype(np.int32), intensity_image=None):
        lid = int(r.label)
        row = {
            "label_id": lid,
            "centroid_y": float(r.centroid[0]),
            "centroid_x": float(r.centroid[1]),
            "area_px": int(r.area),
        }
        mask = labels == lid
        for c in range(cyx.shape[0]):
            plane = cyx[c]
            vals = plane[mask]
            row[f"mean_ch{c}"] = float(np.mean(vals))
            row[f"sum_ch{c}"] = float(np.sum(vals))
            row[f"max_ch{c}"] = float(np.max(vals))
        rows.append(row)

    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c.startswith(("mean_", "sum_", "max_"))]
    X = df[feat_cols].values.astype(np.float32)
    return df, X


def main(args: argparse.Namespace) -> None:
    setup_logging()
    if args.config:
        tmp = argparse.ArgumentParser()
        add_arguments(tmp)
        try:
            cfg = load_config(args.config)
            args = merge_config_with_args("image_quantify", cfg, args, tmp)
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

    if not getattr(sdata, "labels", None) or args.labels_key not in sdata.labels:
        logging.error("Labels key %s not found on SpatialData", args.labels_key)
        sys.exit(1)
    lab_arr = labels_to_numpy(sdata.labels[args.labels_key])

    df, X = _quantify(cyx, lab_arr)

    import anndata as ad

    adata = ad.AnnData(X=X, obs=df.set_index("label_id"))
    adata.obs["region"] = args.labels_key
    adata.obs["instance_id"] = np.arange(adata.n_obs)
    adata.var_names = [f"feat_{i}" for i in range(adata.n_vars)]

    table_key = args.table_key
    if hasattr(sdata, "tables"):
        sdata.tables[table_key] = adata
    else:
        sdata.table = adata

    if args.inplace:
        out_path = inp
    else:
        if not args.output:
            logging.error("Provide --output or --inplace")
            sys.exit(1)
        out_path = Path(args.output).expanduser().resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_spatial_data(sdata, out_path, overwrite=out_path.exists())
    logger.info("Saved quantified table to %s [%s]", out_path, table_key)
