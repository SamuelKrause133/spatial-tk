#!/usr/bin/env python3
"""
visualize command: render full-slide or ROI spatial plots.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from spatial_tk.core.data_io import load_existing_spatial_data, load_xenium_dataset, setup_squidpy_structure
from spatial_tk.core.visualization import (
    compile_style_arrays,
    extract_image_overlay,
    load_visualization_spec,
    render_roi_plot,
    resolve_rois,
    write_resolved_settings,
    write_roi_metadata,
)
from spatial_tk.utils.config import load_config, merge_config_with_args
from spatial_tk.utils.helpers import get_table


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=False, help="Path to input .zarr file")
    parser.add_argument("--output", required=False, help="Output image path or directory")
    parser.add_argument("--table-key", default=None, help="Optional table key in SpatialData.tables")
    parser.add_argument("--spatial-key", default="spatial", help="obsm key for spatial coordinates")
    parser.add_argument("--view", choices=["full", "roi"], default="full", help="Render full slide or ROI(s)")
    parser.add_argument("--roi", action="append", default=None, help="ROI bbox xmin,ymin,xmax,ymax (repeatable)")
    parser.add_argument("--roi-file", default=None, help="CSV with ROI bbox columns xmin,ymin,xmax,ymax and optional name")
    parser.add_argument("--random-rois", type=int, default=0, help="Number of random ROIs to generate")
    parser.add_argument("--roi-width", type=float, default=None, help="Random ROI width in coordinate units")
    parser.add_argument("--roi-height", type=float, default=None, help="Random ROI height in coordinate units")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for random ROI generation")
    parser.add_argument("--max-points", type=int, default=None, help="Optional max points to render (uniform random sample)")
    parser.add_argument("--spec", default=None, help="Supplemental TOML visualization specification")
    parser.add_argument("--figsize", default=None, help="Figure size width,height (overrides spec)")
    parser.add_argument("--dpi", type=int, default=None, help="Output DPI (overrides spec)")
    parser.add_argument("--title", default=None, help="Optional plot title override")
    parser.add_argument("--overlay-image", action="store_true", help="Overlay image in background if available")
    parser.add_argument("--image-layer", default=None, help="Image layer key from SpatialData.images")
    parser.add_argument("--image-scale", type=int, default=None, help="Multiscale image pyramid level to render")
    parser.add_argument("--image-channel", default=None, help="Image channel name or index to render")
    parser.add_argument("--image-alpha", type=float, default=None, help="Background image alpha")
    parser.add_argument("--overwrite", action="store_true", help="Allow overwriting output files")
    parser.add_argument("--config", help="Path to TOML configuration file (optional)")


def _parse_figsize(figsize: Optional[str], default: list[float]) -> list[float]:
    if not figsize:
        return default
    parts = [p.strip() for p in figsize.split(",")]
    if len(parts) != 2:
        raise ValueError("--figsize must be width,height")
    return [float(parts[0]), float(parts[1])]


def _resolve_output(args: argparse.Namespace, n_rois: int) -> Dict[str, Path]:
    if not args.output:
        raise ValueError("--output is required")
    output = Path(args.output)
    if args.view == "full" and n_rois == 1 and output.suffix.lower() == ".png":
        return {"mode": "single_file", "path": output}
    if args.view == "roi" and n_rois == 1 and output.suffix.lower() == ".png":
        return {"mode": "single_file", "path": output}
    return {"mode": "directory", "path": output}


def main(args: argparse.Namespace) -> None:
    if args.config:
        try:
            config_dict = load_config(args.config)
            temp_parser = argparse.ArgumentParser()
            add_arguments(temp_parser)
            args = merge_config_with_args("visualize", config_dict, args, temp_parser)
        except Exception as exc:
            logging.error(f"Error loading config file: {exc}")
            sys.exit(1)

    if not args.input:
        logging.error("--input is required (provide via CLI or config file)")
        sys.exit(1)
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)

    try:
        spec = load_visualization_spec(args.spec)
        if input_path.suffix == ".zarr":
            sdata = load_existing_spatial_data(input_path, load_images=args.overlay_image)
        else:
            # Allow direct rendering from raw Xenium output directory.
            sdata = load_xenium_dataset(input_path, sample_name=input_path.name)
            setup_squidpy_structure(sdata, library_id=input_path.name)
        adata = get_table(sdata, table_key=args.table_key)
        if adata is None:
            raise ValueError("No expression table found in spatial data")
        if args.spatial_key not in adata.obsm:
            raise ValueError(f"Missing coordinates in adata.obsm['{args.spatial_key}']")

        coords = adata.obsm[args.spatial_key]
        if coords.shape[1] < 2:
            raise ValueError("Spatial coordinate array requires at least two columns")
        obs = adata.obs.copy()

        if args.max_points and args.max_points > 0 and adata.n_obs > args.max_points:
            sample_idx = np.random.default_rng(args.random_state).choice(adata.n_obs, size=args.max_points, replace=False)
            coords = coords[sample_idx]
            obs = obs.iloc[sample_idx].copy()

        rois = resolve_rois(
            coords=coords,
            view=args.view,
            roi_strings=args.roi,
            roi_file=args.roi_file,
            random_rois=args.random_rois,
            roi_width=args.roi_width,
            roi_height=args.roi_height,
            random_state=args.random_state,
        )
        output_info = _resolve_output(args, n_rois=len(rois))

        plot_spec = spec.get("plot", {})
        figsize = _parse_figsize(args.figsize, default=plot_spec.get("figsize", [8, 8]))
        dpi = int(args.dpi if args.dpi is not None else plot_spec.get("dpi", 300))
        image_alpha = float(args.image_alpha if args.image_alpha is not None else plot_spec.get("image_alpha", 0.5))

        background_image = None
        if args.overlay_image or plot_spec.get("background", False):
            if hasattr(sdata, "images") and sdata.images:
                image_layer = args.image_layer or plot_spec.get("image_layer")
                if not image_layer:
                    image_layer = list(sdata.images.keys())[0]
                try:
                    image_scale = args.image_scale
                    if image_scale is None and "image_scale" in plot_spec:
                        image_scale = int(plot_spec["image_scale"])
                    image_channel = args.image_channel or plot_spec.get("image_channel")
                    background_image = extract_image_overlay(
                        sdata.images[image_layer],
                        image_scale=image_scale,
                        image_channel=image_channel,
                    )
                except Exception as exc:
                    logging.warning("Could not parse image layer '%s' for overlay: %s", image_layer, exc)
                    background_image = None
            else:
                logging.warning("Overlay image requested but no SpatialData images found")

        style_arrays = compile_style_arrays(obs=obs, spec=spec)

        if output_info["mode"] == "single_file":
            output_path = output_info["path"]
            if output_path.exists() and not args.overwrite:
                raise ValueError(f"Output file exists: {output_path}. Use --overwrite.")
            render_roi_plot(
                coords=coords,
                obs=obs,
                roi=rois[0],
                style_arrays=style_arrays,
                output_path=output_path,
                title=args.title or plot_spec.get("title"),
                figsize=figsize,
                dpi=dpi,
                background_image=background_image,
                image_alpha=image_alpha,
            )
        else:
            out_dir = output_info["path"]
            if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
                raise ValueError(f"Output directory not empty: {out_dir}. Use --overwrite.")
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, roi in enumerate(rois, start=1):
                filename = f"roi_{i:03d}.png"
                render_roi_plot(
                    coords=coords,
                    obs=obs,
                    roi=roi,
                    style_arrays=style_arrays,
                    output_path=out_dir / filename,
                    title=args.title or plot_spec.get("title") or roi.name,
                    figsize=figsize,
                    dpi=dpi,
                    background_image=background_image,
                    image_alpha=image_alpha,
                )
            write_roi_metadata(rois=rois, output_path=out_dir / "rois.csv", random_state=args.random_state)
            resolved = {
                "input": args.input,
                "output": str(out_dir),
                "view": args.view,
                "spatial_key": args.spatial_key,
                "table_key": args.table_key,
                "figsize": figsize,
                "dpi": dpi,
                "overlay_image": bool(args.overlay_image or plot_spec.get("background", False)),
                "image_layer": args.image_layer or plot_spec.get("image_layer"),
                "image_alpha": image_alpha,
                "random_state": args.random_state,
                "n_rois": len(rois),
                "spec_file": args.spec,
            }
            write_resolved_settings(resolved, out_dir / "visualize.resolved.json")

    except Exception as exc:
        logging.error(f"Visualize failed: {exc}", exc_info=True)
        sys.exit(1)
