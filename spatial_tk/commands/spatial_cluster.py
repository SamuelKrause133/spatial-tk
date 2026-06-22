#!/usr/bin/env python3
"""
spatial_cluster command: Cluster neighborhood composition profiles.
"""

import argparse
import logging
import sys
from pathlib import Path

from spatial_tk.core.data_io import load_existing_spatial_data
from spatial_tk.core import spatial_clustering
from spatial_tk.core import spatial_neighbors as spatial_neighbors_core
from spatial_tk.utils.helpers import (
    get_output_path,
    get_table,
    save_command_output,
)
from spatial_tk.utils.config import load_config, merge_config_with_args


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input", required=False, help="Path to input .zarr file")
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
        "--cell-type-key",
        default=None,
        help="Required adata.obs column containing cell-type labels",
    )
    parser.add_argument(
        "--connectivities-key",
        default="spatial_connectivities",
        help="adata.obsp key for neighborhood connectivity matrix",
    )
    parser.add_argument(
        "--neighbor-k",
        type=int,
        default=None,
        help="If connectivities are missing, compute neighbors on demand with this k",
    )
    parser.add_argument(
        "--spatial-key",
        default="spatial",
        help="obsm key for coordinates when computing neighbors on demand",
    )
    parser.add_argument(
        "--library-key",
        default=None,
        help="obs column for library ids when computing neighbors on demand",
    )
    parser.add_argument(
        "--output-key",
        default="spatial_cluster",
        help="obs column for best selected spatial cluster labels",
    )
    parser.add_argument(
        "--results-key",
        default="spatial_cluster",
        help="uns key for detailed spatial clustering outputs",
    )
    parser.add_argument(
        "--mode",
        choices=["kmeans", "hdbscan"],
        default="kmeans",
        help="Clustering mode: kmeans (default) or hdbscan",
    )
    parser.add_argument(
        "--min-clusters",
        type=int,
        default=2,
        help="Minimum k-means cluster count to test (default: 2)",
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=20,
        help="Maximum k-means cluster count to test (default: 20)",
    )
    parser.add_argument(
        "--force-n-clusters",
        type=int,
        default=None,
        help="Force final selected k-means cluster count while still storing full sweep",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for k-means reproducibility (default: 0)",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=5,
        help="Minimum cluster size for HDBSCAN mode (default: 5)",
    )
    parser.add_argument(
        "--hdbscan-min-samples",
        type=int,
        default=None,
        help="min_samples for HDBSCAN mode (default: None)",
    )
    parser.add_argument(
        "--hdbscan-cluster-selection-epsilon",
        type=float,
        default=0.0,
        help="cluster_selection_epsilon for HDBSCAN mode (default: 0.0)",
    )
    parser.add_argument(
        "--hdbscan-metric",
        default="euclidean",
        help="Distance metric for HDBSCAN mode (default: euclidean)",
    )
    parser.add_argument(
        "--hdbscan-allow-single-cluster",
        action="store_true",
        default=False,
        help="Allow single-cluster result in HDBSCAN mode",
    )
    parser.add_argument(
        "--include-self",
        action="store_true",
        default=True,
        help="Include the focal cell in neighborhood composition vectors (default: true)",
    )
    parser.add_argument(
        "--exclude-self",
        action="store_false",
        dest="include_self",
        help="Exclude the focal cell from neighborhood composition vectors",
    )
    parser.add_argument(
        "--normalize-composition",
        action="store_true",
        default=True,
        help="Normalize composition rows to proportions (default: true)",
    )
    parser.add_argument(
        "--raw-composition",
        action="store_false",
        dest="normalize_composition",
        help="Keep raw neighborhood cell-type counts instead of proportions",
    )
    parser.add_argument(
        "--config",
        help="Path to TOML configuration file (optional)",
    )


def main(args: argparse.Namespace) -> None:
    if args.config:
        try:
            config_dict = load_config(args.config)
            temp_parser = argparse.ArgumentParser()
            add_arguments(temp_parser)
            args = merge_config_with_args("spatial_cluster", config_dict, args, temp_parser)
        except Exception as exc:
            logging.error(f"Error loading config file: {exc}")
            sys.exit(1)

    if not args.input:
        logging.error("--input is required (provide via CLI or config file)")
        sys.exit(1)
    if not args.cell_type_key:
        logging.error("--cell-type-key is required")
        sys.exit(1)
    if args.mode != "kmeans" and args.force_n_clusters is not None:
        logging.error("--force-n-clusters is only supported when --mode kmeans")
        sys.exit(1)

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
        adata = get_table(sdata, table_key=args.table_key)
        if adata is None:
            if args.table_key:
                raise ValueError(f"No table found for --table-key={args.table_key}")
            raise ValueError("No expression table found in spatial data")

        adata = spatial_clustering.run_spatial_cluster(
            adata,
            cell_type_key=args.cell_type_key,
            connectivities_key=args.connectivities_key,
            mode=args.mode,
            output_key=args.output_key,
            results_key=args.results_key,
            include_self=args.include_self,
            normalize_composition=args.normalize_composition,
            store_composition_in_obsm=True,
            neighbor_k=args.neighbor_k,
            spatial_key=args.spatial_key,
            library_key=args.library_key,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            random_state=args.random_state,
            force_n_clusters=args.force_n_clusters,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
            hdbscan_min_samples=args.hdbscan_min_samples,
            hdbscan_cluster_selection_epsilon=args.hdbscan_cluster_selection_epsilon,
            hdbscan_metric=args.hdbscan_metric,
            hdbscan_allow_single_cluster=args.hdbscan_allow_single_cluster,
        )

        save_command_output(
            adata,
            input_path,
            output_path,
            inplace=args.inplace,
            table_key=args.table_key,
        )
        logging.info("Spatial cluster complete: %s", output_path)
    except Exception as exc:
        logging.error(f"Spatial cluster failed: {exc}", exc_info=True)
        sys.exit(1)
