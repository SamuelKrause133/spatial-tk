#!/usr/bin/env python3
"""
Assign command: Assign cell type labels to clusters from enrichment scores.

Reads a score matrix stored in obsm (produced by the quantitate command),
applies a configurable assignment strategy to label each cluster, and
optionally runs per-cluster differential expression.
"""

import argparse
import logging
import sys
from pathlib import Path

from spatial_tk.core.cli_constants import ASSIGNMENT_STRATEGY_CHOICES
from spatial_tk.core.data_io import load_existing_spatial_data
from spatial_tk.core import annotation
from spatial_tk.core import differential
from spatial_tk.core import plotting
from spatial_tk.utils.helpers import (
    get_output_path,
    get_table,
    save_command_output,
)
from spatial_tk.utils.config import load_config, merge_config_with_args


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the assign command."""
    parser.add_argument(
        "--input",
        required=False,
        help="Path to input scored .zarr file (produced by quantitate)",
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
        "--score-key",
        required=False,
        default=None,
        help=(
            "Full obsm key name holding the enrichment scores to use for assignment "
            "(e.g. 'score_mlm_custom', 'score_mlm_PanglaoDB'). "
            "Must match the key produced by the quantitate command."
        ),
    )
    parser.add_argument(
        "--cluster-key",
        default=None,
        help=(
            "Cluster column key in obs to annotate (e.g. leiden_res0p5). "
            "If not specified, all leiden_res* columns are used."
        ),
    )
    parser.add_argument(
        "--annotation-key",
        default=None,
        help=(
            "obs column name to write cell type labels into. "
            "Defaults to 'cell_type_res{resolution}' derived from --cluster-key."
        ),
    )
    parser.add_argument(
        "--strategy",
        default="top_positive",
        choices=list(ASSIGNMENT_STRATEGY_CHOICES),
        help=(
            "Assignment strategy. Default: 'top_positive' (highest positive "
            "enrichment stat per cluster; 'Unknown' if none)."
        ),
    )
    parser.add_argument(
        "--run-de",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Run differential expression (rank_genes_groups) per cluster key. "
            "Use --no-run-de to skip. Default: True"
        ),
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Generate and save UMAP, dotplot, DE, and enrichment heatmap plots",
    )
    parser.add_argument(
        "--config",
        help="Path to TOML configuration file (optional)",
    )


def main(args: argparse.Namespace) -> None:
    """Execute the assign command."""
    # Load and merge config if provided
    if args.config:
        try:
            config_dict = load_config(args.config)
            temp_parser = argparse.ArgumentParser()
            add_arguments(temp_parser)
            args = merge_config_with_args("assign", config_dict, args, temp_parser)
        except Exception as exc:
            logging.error(f"Error loading config file: {exc}")
            sys.exit(1)

    # Validate required arguments
    if not args.input:
        logging.error("--input is required (provide via CLI or config file)")
        sys.exit(1)

    if not args.score_key:
        logging.error("--score-key is required (provide via CLI or config file)")
        sys.exit(1)

    logging.info("=" * 60)
    logging.info("Xenium Process: Cluster Label Assignment (assign)")
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

        # Validate score key presence
        if args.score_key not in adata.obsm:
            logging.error(
                f"Score key '{args.score_key}' not found in adata.obsm. "
                f"Available keys: {list(adata.obsm.keys())}. "
                "Run the quantitate command first."
            )
            sys.exit(1)

        # Determine cluster keys
        if args.cluster_key:
            cluster_keys = [args.cluster_key]
        else:
            cluster_keys = [col for col in adata.obs.columns if col.startswith("leiden_res")]
            if not cluster_keys:
                logging.warning(
                    "No leiden_res* columns found. Run the cluster command first."
                )

        # ------------------------------------------------------------------ #
        # Assign cell type labels per clustering resolution
        # ------------------------------------------------------------------ #
        for cluster_key in cluster_keys:
            res_str = cluster_key.replace("leiden_res", "")

            # Determine annotation key
            if args.annotation_key:
                annotation_key = args.annotation_key
            else:
                annotation_key = f"cell_type_res{res_str}"

            adata = annotation.assign_clusters(
                adata,
                score_key=args.score_key,
                cluster_key=cluster_key,
                annotation_key=annotation_key,
                strategy=args.strategy,
            )

        # ------------------------------------------------------------------ #
        # Differential expression
        # ------------------------------------------------------------------ #
        if args.run_de:
            for cluster_key in cluster_keys:
                adata, _ = differential.run_gene_expression_de(adata, cluster_key)

        # ------------------------------------------------------------------ #
        # Save
        # ------------------------------------------------------------------ #
        save_command_output(adata, input_path, output_path, inplace=args.inplace)
        logging.info(f"Saved results to: {output_path}")

        # ------------------------------------------------------------------ #
        # Optional plots
        # ------------------------------------------------------------------ #
        if args.save_plots:
            plots_dir = output_path.parent / "plots"
            plots_dir.mkdir(exist_ok=True)

            for cluster_key in cluster_keys:
                res_str = cluster_key.replace("leiden_res", "")
                try:
                    resolution = float(res_str.replace("p", "."))
                except ValueError:
                    resolution = None

                if args.annotation_key:
                    annotation_key = args.annotation_key
                else:
                    annotation_key = f"cell_type_res{res_str}"

                if annotation_key in adata.obs.columns:
                    plotting.save_umap_plots(
                        adata, plots_dir, cluster_key, annotation_key, resolution
                    )
                    plotting.create_enrichment_heatmap(
                        adata, plots_dir, cluster_key, resolution
                    )

                if args.run_de:
                    plotting.save_de_plots(adata, plots_dir, cluster_key, resolution)

        logging.info("=" * 60)
        logging.info("Assign complete.")
        logging.info("=" * 60)

    except Exception as exc:
        logging.error(f"Assign failed: {exc}", exc_info=True)
        sys.exit(1)
