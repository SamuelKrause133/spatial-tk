#!/usr/bin/env python3
"""
Quantitate command: Run MLM/ULM enrichment scoring across the dataset.

Supports a custom marker gene list (CSV), decoupler built-in resources
(panglao, hallmark, collectri, dorothea, progeny), or both simultaneously.
An optional cell filter restricts scoring to a subset of cells; scores are
written back into the full dataset with NaN for excluded cells.
"""

import argparse
import logging
import sys
from pathlib import Path

from spatial_tk.core.cli_constants import PRESET_RESOURCE_NAMES
from spatial_tk.core.data_io import load_existing_spatial_data
from spatial_tk.core import annotation
from spatial_tk.utils.helpers import (
    get_table,
    get_output_path,
    save_command_output,
)
from spatial_tk.utils.config import load_config, merge_config_with_args

VALID_METHODS = ("mlm", "ulm")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the quantitate command."""
    parser.add_argument(
        "--input",
        required=False,
        help="Path to input clustered .zarr file",
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
        "--markers",
        default=None,
        help=(
            "Path to CSV file with marker genes (columns: cell_type, gene). "
            "At least one of --markers or --preset-resources is required."
        ),
    )
    parser.add_argument(
        "--score-key",
        default="custom",
        help=(
            "Key suffix for custom marker scores stored in obsm. "
            "Result is stored at obsm['score_<method>_<score-key>']. "
            "Default: 'custom'"
        ),
    )
    parser.add_argument(
        "--method",
        default="mlm",
        choices=list(VALID_METHODS),
        help="Decoupler scoring method: 'mlm' (default) or 'ulm'",
    )
    parser.add_argument(
        "--tmin",
        type=int,
        default=2,
        help="Minimum number of targets per source for decoupler (default: 2)",
    )
    parser.add_argument(
        "--preset-resources",
        default=None,
        help=(
            "Comma-separated list of built-in decoupler resources to score against. "
            f"Valid names: {', '.join(PRESET_RESOURCE_NAMES)}. "
            "Each resource is stored at obsm['score_<method>_<resource>']."
        ),
    )
    parser.add_argument(
        "--panglao-min-sensitivity",
        type=float,
        default=0.5,
        help="Minimum sensitivity for PanglaoDB markers (default: 0.5)",
    )
    parser.add_argument(
        "--panglao-canonical-only",
        action="store_true",
        default=True,
        help="Only use canonical PanglaoDB markers (default: True)",
    )
    parser.add_argument(
        "--filter-obs",
        default=None,
        help=(
            "Filter expression 'column==value' to subset cells before scoring "
            "(e.g. 'cell_type==Fibroblast'). Scores for excluded cells are NaN."
        ),
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate and save enrichment heatmap plots",
    )
    parser.add_argument(
        "--config",
        help="Path to TOML configuration file (optional)",
    )


def main(args: argparse.Namespace) -> None:
    """Execute the quantitate command."""
    # Load and merge config if provided
    if args.config:
        try:
            config_dict = load_config(args.config)
            temp_parser = argparse.ArgumentParser()
            add_arguments(temp_parser)
            args = merge_config_with_args("quantitate", config_dict, args, temp_parser)
        except Exception as exc:
            logging.error(f"Error loading config file: {exc}")
            sys.exit(1)

    # Validate required arguments
    if not args.input:
        logging.error("--input is required (provide via CLI or config file)")
        sys.exit(1)

    if not args.markers and not args.preset_resources:
        logging.error(
            "At least one of --markers or --preset-resources must be specified."
        )
        sys.exit(1)

    logging.info("=" * 60)
    logging.info("Xenium Process: Enrichment Scoring (quantitate)")
    logging.info("=" * 60)

    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)

    try:
        output_path = get_output_path(args.input, args.output, args.inplace)
    except ValueError as exc:
        logging.error(str(exc))
        sys.exit(1)

    try:
        sdata = load_existing_spatial_data(input_path)
        adata = get_table(sdata)

        if adata is None:
            raise ValueError("No expression table found in spatial data")

        logging.info(f"Loaded dataset: {adata.n_obs} cells × {adata.n_vars} genes")

        # ------------------------------------------------------------------ #
        # Optional cell filter
        # ------------------------------------------------------------------ #
        mask = None
        if args.filter_obs:
            try:
                mask, _ = annotation.filter_cells_by_obs(adata, args.filter_obs)
            except (ValueError, KeyError) as exc:
                logging.error(f"Invalid --filter-obs expression: {exc}")
                sys.exit(1)

        # ------------------------------------------------------------------ #
        # Custom marker gene scoring
        # ------------------------------------------------------------------ #
        if args.markers:
            markers_path = Path(args.markers)
            if not markers_path.exists():
                logging.error(f"Markers file not found: {markers_path}")
                sys.exit(1)

            markers = annotation.load_marker_genes(str(markers_path))
            net_df = annotation.markers_dict_to_dataframe(markers)

            all_marker_genes = set(net_df["target"])
            missing = all_marker_genes - set(adata.var_names)
            if missing:
                logging.info(f"Note: {len(missing)} marker genes not found in dataset")

            adata = annotation.run_enrichment_scoring(
                adata,
                net_df=net_df,
                score_key=args.score_key,
                method=args.method,
                tmin=args.tmin,
                mask=mask,
            )

        # ------------------------------------------------------------------ #
        # Preset resource scoring
        # ------------------------------------------------------------------ #
        if args.preset_resources:
            preset_names = [n.strip() for n in args.preset_resources.split(",") if n.strip()]
            for name in preset_names:
                try:
                    net_df = annotation.load_preset_resource(
                        name,
                        panglao_min_sensitivity=args.panglao_min_sensitivity,
                        panglao_canonical_only=args.panglao_canonical_only,
                    )
                except ValueError as exc:
                    logging.error(str(exc))
                    sys.exit(1)

                adata = annotation.run_enrichment_scoring(
                    adata,
                    net_df=net_df,
                    score_key=name,
                    method=args.method,
                    tmin=args.tmin,
                    mask=mask,
                )

        # ------------------------------------------------------------------ #
        # Save
        # ------------------------------------------------------------------ #
        save_command_output(adata, input_path, output_path, inplace=args.inplace)
        logging.info(f"Saved results to: {output_path}")

        # ------------------------------------------------------------------ #
        # Optional plots
        # ------------------------------------------------------------------ #
        if args.save_plots:
            from spatial_tk.core import plotting
            plots_dir = output_path.parent / "plots"
            plots_dir.mkdir(exist_ok=True)

            scored_keys = []
            if args.markers:
                scored_keys.append(f"score_{args.method}_{args.score_key}")
            if args.preset_resources:
                for name in [n.strip() for n in args.preset_resources.split(",") if n.strip()]:
                    scored_keys.append(f"score_{args.method}_{name}")

            for key in scored_keys:
                if key in adata.obsm:
                    try:
                        plotting.create_enrichment_heatmap(
                            adata, plots_dir, key, resolution=None
                        )
                    except Exception as exc:
                        logging.warning(f"  Could not save enrichment heatmap for {key}: {exc}")

        logging.info("=" * 60)
        logging.info("Quantitate complete.")
        logging.info("=" * 60)

    except Exception as exc:
        logging.error(f"Quantitate failed: {exc}", exc_info=True)
        sys.exit(1)
