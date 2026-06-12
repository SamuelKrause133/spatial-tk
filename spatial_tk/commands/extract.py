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
from spatial_tk.utils.batch_csv import read_batch_manifest, resolve_manifest_path
from spatial_tk.utils.config import load_config, merge_config_with_args
from spatial_tk.utils.helpers import setup_logging

CLI_HELP = "Extract per-object image chips"
CLI_DESCRIPTION = "Crop fixed (H,W) windows around centroids; optional montage PNG."

logger = logging.getLogger(__name__)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=False)
    parser.add_argument("--inplace", action="store_true")
    parser.add_argument("--output", required=False)
    parser.add_argument(
        "--batch-csv",
        required=False,
        help="CSV with zarr_path and extract_path columns (other columns allowed); writes chips.npz and chip_montage.png per row.",
    )
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


def _prepare_montage_display(
    chips: np.ndarray, max_chips: int, include_mask_channel: bool
) -> np.ndarray:
    n = min(max_chips, chips.shape[0])
    disp = chips[:n].astype(np.float32, copy=True)
    if include_mask_channel:
        mask = disp[..., -1] > 0
        disp = disp[..., :-1]
        disp[~mask] = 0
    if disp.shape[-1] >= 3:
        disp = disp[..., :3]
        lo = disp.min(axis=(0, 1, 2), keepdims=True)
        hi = disp.max(axis=(0, 1, 2), keepdims=True)
        disp = (disp - lo) / (hi - lo + 1e-9)
    elif disp.shape[-1] >= 1:
        d0 = disp[..., 0]
        d0 = (d0 - d0.min()) / (np.ptp(d0) + 1e-9)
        disp = d0[..., np.newaxis]
    return disp


def _write_chips_and_montage(
    chips: np.ndarray,
    *,
    chips_npz: Path,
    montage_png: Path | None,
    max_chips: int,
    include_mask_channel: bool,
) -> None:
    chips_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(chips_npz, chips=chips)
    logger.info("Saved chip array to %s", chips_npz)
    if montage_png is not None:
        disp = _prepare_montage_display(chips, max_chips, include_mask_channel)
        n = min(max_chips, chips.shape[0])
        write_montage_png(disp, montage_png, n=n)
        logger.info("Montage written to %s", montage_png)


def _run_chip_pipeline(
    args: argparse.Namespace,
    sdata_path: Path,
    chips_npz: Path,
    montage_png: Path | None,
) -> None:
    sdata = read_sdata(sdata_path)
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
    _write_chips_and_montage(
        chips,
        chips_npz=chips_npz,
        montage_png=montage_png,
        max_chips=args.max_chips,
        include_mask_channel=args.include_mask_channel,
    )


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

    batch_csv = getattr(args, "batch_csv", None)
    try:
        if batch_csv:
            rows, base = read_batch_manifest(
                Path(batch_csv), required={"zarr_path", "extract_path"}
            )
            for idx, row in enumerate(rows):
                zarr_path = resolve_manifest_path(base, row["zarr_path"])
                extract_path = resolve_manifest_path(base, row["extract_path"])
                extract_path.mkdir(parents=True, exist_ok=True)
                chips_npz = extract_path / "chips.npz"
                montage_png = extract_path / "chip_montage.png"
                logger.info(
                    "Processing batch row %s: %s -> %s",
                    idx,
                    zarr_path,
                    extract_path,
                )
                _run_chip_pipeline(
                    args,
                    zarr_path,
                    chips_npz,
                    montage_png,
                )
            return

        if not args.input:
            logging.error("--input is required unless --batch-csv is set")
            sys.exit(1)

        inp = Path(args.input).expanduser().resolve()
        montage_path = Path(args.montage_png).expanduser().resolve() if args.montage_png else None
        chips_dir = inp.parent / (inp.name + "_chips")
        _run_chip_pipeline(
            args,
            inp,
            chips_dir / "chips.npz",
            montage_path,
        )

        if args.inplace:
            out_path = inp
        else:
            if not args.output:
                logging.error("Provide --output or --inplace")
                sys.exit(1)
            out_path = Path(args.output).expanduser().resolve()

        sdata = read_sdata(inp)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        save_spatial_data(sdata, out_path, overwrite=out_path.exists())
        logger.info("SpatialData written to %s", out_path)
    except Exception as e:
        logging.error("extract failed: %s", e)
        sys.exit(1)
