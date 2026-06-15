#!/usr/bin/env python3
"""
spatial_neighbors command: Build a spatial neighbor graph with Squidpy.
"""

import argparse
import logging
import sys
from pathlib import Path

from spatial_tk.core import spatial_neighbors as spatial_neighbors_core
from spatial_tk.core.data_io import load_existing_spatial_data, save_spatial_data
from spatial_tk.utils.helpers import get_output_path, get_table, set_table, prepare_spatial_data_for_save
from spatial_tk.utils.config import load_config, merge_config_with_args


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the spatial_neighbors command."""
    parser.add_argument(
        "--input",
        required=False,
        help="Path to input .zarr file",
    )
    parser.add_argument(
        "--output",
        help="Path to output .zarr file (required unless --inplace is used)",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Modify the input file in place instead of creating a new file",
    )
    parser.add_argument(
        "--table-key",
        default=None,
        help="Optional table key in SpatialData.tables to use",
    )
    parser.add_argument(
        "--spatial-key",
        default="spatial",
        help="AnnData obsm key for spatial coordinates (default: spatial)",
    )
    parser.add_argument(
        "--library-key",
        default=None,
        help="AnnData obs column storing per-cell library ids",
    )
    parser.add_argument(
        "--library-id",
        default=None,
        help=(
            "Convenience single-library id. If set without --library-key, "
            "a temporary obs library column is created with this constant value."
        ),
    )
    parser.add_argument(
        "--coord-type",
        choices=["grid", "generic"],
        default=None,
        help="Coordinate type for Squidpy spatial graph",
    )
    parser.add_argument(
        "--n-neighs",
        type=int,
        default=6,
        help="Number of nearest neighbors for generic coordinates (default: 6)",
    )
    parser.add_argument(
        "--radius",
        default=None,
        help="Radius as float (e.g., 100) or interval min,max (e.g., 50,200)",
    )
    parser.add_argument(
        "--transform",
        choices=["spectral", "cosine", "none"],
        default="none",
        help="Optional adjacency transform (default: none)",
    )
    parser.add_argument(
        "--key-added",
        default="spatial",
        help="Prefix for output keys in obsp/uns (default: spatial)",
    )
    parser.add_argument(
        "--config",
        help="Path to TOML configuration file (optional)",
    )


def main(args: argparse.Namespace) -> None:
    """Execute the spatial_neighbors command."""
    if args.config:
        try:
            config_dict = load_config(args.config)
            temp_parser = argparse.ArgumentParser()
            add_arguments(temp_parser)
            args = merge_config_with_args("spatial_neighbors", config_dict, args, temp_parser)
        except Exception as exc:
            logging.error(f"Error loading config file: {exc}")
            sys.exit(1)

    if not args.input:
        logging.error("--input is required (provide via CLI or config file)")
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)

    if args.n_neighs <= 0:
        logging.error("--n-neighs must be > 0")
        sys.exit(1)

    try:
        output_path = get_output_path(args.input, args.output, args.inplace)
        parsed_radius = spatial_neighbors_core.parse_radius(args.radius)
        normalized_transform = spatial_neighbors_core.normalize_transform(args.transform)
    except ValueError as exc:
        logging.error(str(exc))
        sys.exit(1)

    try:
        sdata = load_existing_spatial_data(input_path)
        adata = get_table(sdata, table_key=args.table_key)
        if adata is None:
            if args.table_key:
                raise ValueError(f"No table found for --table-key={args.table_key}")
            raise ValueError("No expression table found in spatial data")

        library_key = args.library_key
        if args.library_id and not library_key:
            library_key = "__spatial_tk_library_id"
            adata.obs[library_key] = args.library_id
            logging.info(
                "Using --library-id via temporary obs column '%s' with value '%s'",
                library_key,
                args.library_id,
            )
        elif args.library_id and library_key:
            logging.info(
                "Both --library-key and --library-id supplied; using --library-key='%s'",
                library_key,
            )

        spatial_neighbors_core.compute_spatial_neighbors(
            adata=adata,
            spatial_key=args.spatial_key,
            library_key=library_key,
            coord_type=args.coord_type,
            n_neighs=args.n_neighs,
            radius=parsed_radius,
            transform=normalized_transform,
            key_added=args.key_added,
        )

        prepare_spatial_data_for_save(adata)
        set_table(sdata, adata, table_key=args.table_key)

        if not args.inplace:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        save_spatial_data(sdata, output_path, overwrite=args.inplace)
        logging.info("Spatial neighbors complete: %s", output_path)
    except Exception as exc:
        logging.error(f"Spatial neighbors failed: {exc}", exc_info=True)
        sys.exit(1)
