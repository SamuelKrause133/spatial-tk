#!/usr/bin/env python3
"""Extract fixed-size 2D chips around segmented objects."""

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
    labels_to_numpy,
    read_sdata,
    write_montage_png,
)
from spatial_tk.utils.config import load_config, merge_config_with_args
from spatial_tk.utils.helpers import setup_logging

CLI_HELP = "Extract per-object image chips"
CLI_DESCRIPTION = "Crop fixed (H,W) windows around centroids; optional montage PNG."

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=False)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--output", required=False)
    parser.add_argument("--image-key", default=None)
    parser.add_argument("--labels-key", default="nuclei_labels")
    parser.add_argument(
        "--projection-mode",
        default="equatorial",
        choices=["equatorial"],
        help="reserved for future Z handling (2D only for now)",
    )
    parser.add_argument(
        "--chip-size",
        nargs=2,
        type=int,
        default=(64, 64),
        metavar=("H", "W"),
        help="Chip height and width in pixels (default 64 64)",
    )
    parser.add_argument(
        "--include-mask-channel",
        action="store_true",
        help="Append a binary mask channel to each chip",
    )
    parser.add_argument(
        "--montage-png",
        default=None,
        help="Optional path to save a tiled PNG of the first 12 chips",
    )
    parser.add_argument("--max-chips", type=int, default=12, help="Montage chip count (default 12)")
    parser.add_argument("--config", help="TOML configuration (optional)")


def _extract_chips(
    cyx: np.ndarray,
    labels: np.ndarray,
    chip_hw: tuple[int, int],
    include_mask: bool,
    max_labels: int = 256,
) -> np.ndarray:
    h, w = chip_hw
    ids = np.unique(labels)
    ids = ids[ids > 0][:max_labels]
    chips = []
    for lid in ids:
        m = labels == lid
        yy, xx = np.nonzero(m)
        cy, cx = float(np.mean(yy)), float(np.mean(xx))
        y0 = int(round(cy)) - h // 2
        x0 = int(round(cx)) - w // 2
        y0 = max(0, min(y0, labels.shape[0] - h))
        x0 = max(0, min(x0, labels.shape[1] - w))

        planes = []
        for c in range(cyx.shape[0]):
            planes.append(cyx[c, y0 : y0 + h, x0 : x0 + w])
        vol = np.stack(planes, axis=-1)  # H W C

        if include_mask:
            m_crop = m[y0 : y0 + h, x0 : x0 + w].astype(np.float32)[:, :, np.newaxis]
            vol = np.concatenate([vol, m_crop], axis=-1)
        chips.append(vol)
    return np.stack(chips, axis=0)


def main(args: argparse.Namespace) -> None:
    setup_logging()
    if args.config:
        tmp = argparse.ArgumentParser()
        add_arguments(tmp)
        try:
            cfg = load_config(args.config)
            args = merge_config_with_args("image_extract", cfg, args, tmp)
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
    lab = labels_to_numpy(sdata.labels[args.labels_key])

    chip_hw = (int(args.chip_size[0]), int(args.chip_size[1]))
    chips = _extract_chips(
        cyx,
        lab,
        chip_hw,
        include_mask=args.include_mask_channel,
        max_labels=max(256, args.max_chips),
    )

    if args.montage_png:
        n = min(args.max_chips, chips.shape[0])
        # normalize first channel for display
        disp = chips[:n].astype(np.float32)
        if disp.shape[-1] >= 1:
            d0 = disp[..., 0]
            d0 = (d0 - d0.min()) / (np.ptp(d0) + 1e-9)
            disp = d0[..., np.newaxis]
        write_montage_png(disp, Path(args.montage_png), n=n)
        logger.info("Montage written to %s", args.montage_png)

    out_dir = inp.parent / (inp.name + "_chips")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "chips.npz", chips=chips)
    logger.info("Saved chip array to %s", out_dir / "chips.npz")

    if args.inplace:
        out_path = inp
    else:
        if not args.output:
            logging.error("Provide --output or --inplace")
            sys.exit(1)
        out_path = Path(args.output).expanduser().resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_spatial_data(sdata, out_path, overwrite=out_path.exists())
    logger.info("SpatialData written to %s", out_path)
