#!/usr/bin/env python3
"""
Main CLI entry point for spatial-tk.

Analysis subcommands are registered with lazy imports so optional image / JVM
stacks are never loaded unless ``spatial-tk image ...`` is used.
"""

import argparse
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

_MISSING_ANALYSIS_MSG = (
    "This command requires the analysis dependencies, which are not installed in this environment.\n"
    "Install the analysis stack (recommended via Makefile):\n"
    "  make venv\n"
    "Or install extras:\n"
    "  pip install -e \".[analysis]\"\n"
)


def _missing_analysis_subcommand(name: str):
    def _run(_args):
        print(
            _MISSING_ANALYSIS_MSG
            + f"\n(Subcommand '{name}' is not available — missing modules.)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    return _run


def create_parser() -> argparse.ArgumentParser:
    """
    Create the analysis-only argument parser (concat, normalize, cluster, ...).

    Command modules are imported only while building this parser — not at
    ``import spatial_tk.cli`` time.

    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        prog="spatial-tk",
        description="Xenium Spatial Transcriptomics Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Concatenate multiple samples
  spatial-tk concat --input samples.csv --output merged.zarr
  
  # Normalize (in place to save space)
  spatial-tk normalize --input merged.zarr --inplace --save-plots
  
  # Cluster with multiple resolutions
  spatial-tk cluster --input merged.zarr --inplace --leiden-resolution 0.2,0.5,1.0
  
  # Score enrichment with a custom marker list (all cells)
  spatial-tk quantitate --input clustered.zarr --inplace --markers markers.csv

  # Score only fibroblasts against a custom list, plus built-in PanglaoDB
  spatial-tk quantitate --input clustered.zarr --inplace \\
      --markers markers.csv --filter-obs "cell_type==Fibroblast" \\
      --preset-resources panglao

  # Assign cell type labels to clusters from the computed scores
  spatial-tk assign --input clustered.zarr --inplace \\
      --score-key score_mlm_custom

  # Build a Squidpy spatial neighbors graph
  spatial-tk spatial_neighbors --input clustered.zarr --inplace \\
      --spatial-key spatial --n-neighs 8 --transform cosine

  # Cluster neighborhood compositions from spatial neighbor graph
  spatial-tk spatial_cluster --input clustered.zarr --inplace \\
      --cell-type-key cell_type_res0p5 --max-clusters 20

  # Differential analysis between groups
  spatial-tk differential --input annotated.zarr --output-dir results/ \\
      --groupby status --compare-groups HIV,NEG
  
  # Full pipeline (separate files)
  spatial-tk concat --input samples.csv --output step1_concat.zarr
  spatial-tk normalize --input step1_concat.zarr --output step2_normalized.zarr
  spatial-tk cluster --input step2_normalized.zarr --output step3_clustered.zarr
  spatial-tk quantitate --input step3_clustered.zarr --output step4_scored.zarr \\
      --markers markers.csv
  spatial-tk assign --input step4_scored.zarr --output step5_annotated.zarr \\
      --score-key score_mlm_custom
  spatial-tk differential --input step5_annotated.zarr --output-dir results/

Image / microscopy (optional deps — separate environment):
  spatial-tk image --help
        """,
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        required=True,
    )

    _register_concat(subparsers)
    _register_normalize(subparsers)
    _register_cluster(subparsers)
    _register_quantitate(subparsers)
    _register_spatial_neighbors(subparsers)
    _register_spatial_cluster(subparsers)
    _register_csv2zarr(subparsers)
    _register_assign(subparsers)
    _register_differential(subparsers)

    return parser


def _register_concat(subparsers: argparse._SubParsersAction) -> None:
    try:
        from spatial_tk.commands import concat
    except ImportError:
        p = subparsers.add_parser("concat", help="Concatenate multiple Xenium .zarr files")
        p.set_defaults(func=_missing_analysis_subcommand("concat"))
        return

    concat_parser = subparsers.add_parser(
        "concat",
        help="Concatenate multiple Xenium .zarr files",
        description="Join multiple Xenium spatial datasets into a single .zarr file",
    )
    concat.add_arguments(concat_parser)
    concat_parser.set_defaults(func=concat.main)


def _register_normalize(subparsers: argparse._SubParsersAction) -> None:
    try:
        from spatial_tk.commands import normalize
    except ImportError:
        p = subparsers.add_parser("normalize", help="Normalize and preprocess data")
        p.set_defaults(func=_missing_analysis_subcommand("normalize"))
        return

    normalize_parser = subparsers.add_parser(
        "normalize",
        help="Normalize and preprocess data",
        description="Perform QC, filtering, normalization, and feature selection",
    )
    normalize.add_arguments(normalize_parser)
    normalize_parser.set_defaults(func=normalize.main)


def _register_cluster(subparsers: argparse._SubParsersAction) -> None:
    try:
        from spatial_tk.commands import cluster
    except ImportError:
        p = subparsers.add_parser("cluster", help="Perform clustering analysis")
        p.set_defaults(func=_missing_analysis_subcommand("cluster"))
        return

    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Perform clustering analysis",
        description="Run PCA, compute neighbors, UMAP, and Leiden clustering",
    )
    cluster.add_arguments(cluster_parser)
    cluster_parser.set_defaults(func=cluster.main)


def _register_quantitate(subparsers: argparse._SubParsersAction) -> None:
    try:
        from spatial_tk.commands import quantitate
    except ImportError:
        p = subparsers.add_parser("quantitate", help="Run enrichment scoring (MLM/ULM) on a gene list or built-in resources")
        p.set_defaults(func=_missing_analysis_subcommand("quantitate"))
        return

    quantitate_parser = subparsers.add_parser(
        "quantitate",
        help="Run enrichment scoring (MLM/ULM) on a gene list or built-in resources",
        description=(
            "Run MLM or ULM enrichment scoring using a custom marker gene list, "
            "decoupler built-in resources (panglao, hallmark, collectri, dorothea, progeny), "
            "or both. Supports optional cell filtering via --filter-obs."
        ),
    )
    quantitate.add_arguments(quantitate_parser)
    quantitate_parser.set_defaults(func=quantitate.main)


def _register_spatial_neighbors(subparsers: argparse._SubParsersAction) -> None:
    try:
        from spatial_tk.commands import spatial_neighbors
    except ImportError:
        p = subparsers.add_parser("spatial_neighbors", help="Compute spatial neighbor graph with Squidpy")
        p.set_defaults(func=_missing_analysis_subcommand("spatial_neighbors"))
        return

    spatial_neighbors_parser = subparsers.add_parser(
        "spatial_neighbors",
        help="Compute spatial neighbor graph with Squidpy",
        description=(
            "Build spatial connectivities/distances with squidpy.gr.spatial_neighbors "
            "using configurable spatial key, neighbor definition, and transform."
        ),
    )
    spatial_neighbors.add_arguments(spatial_neighbors_parser)
    spatial_neighbors_parser.set_defaults(func=spatial_neighbors.main)


def _register_spatial_cluster(subparsers: argparse._SubParsersAction) -> None:
    try:
        from spatial_tk.commands import spatial_cluster
    except ImportError:
        p = subparsers.add_parser("spatial_cluster", help="Cluster spatial neighborhood composition profiles")
        p.set_defaults(func=_missing_analysis_subcommand("spatial_cluster"))
        return

    spatial_cluster_parser = subparsers.add_parser(
        "spatial_cluster",
        help="Cluster spatial neighborhood composition profiles",
        description=(
            "Build neighborhood composition vectors from spatial graph connectivity and "
            "cell-type labels, then run k-means over a cluster-count sweep with silhouette scoring."
        ),
    )
    spatial_cluster.add_arguments(spatial_cluster_parser)
    spatial_cluster_parser.set_defaults(func=spatial_cluster.main)


def _register_csv2zarr(subparsers: argparse._SubParsersAction) -> None:
    try:
        from spatial_tk.commands import csv2zarr
    except ImportError:
        p = subparsers.add_parser(
            "csv2zarr",
            help="Convert image-side CSV export bundle to SpatialData zarr",
        )
        p.set_defaults(func=_missing_analysis_subcommand("csv2zarr"))
        return

    csv2zarr_parser = subparsers.add_parser(
        "csv2zarr",
        help=getattr(csv2zarr, "CLI_HELP", "Convert CSV bundle to SpatialData zarr"),
        description=getattr(csv2zarr, "CLI_DESCRIPTION", "Build a SpatialData zarr from flat files."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    csv2zarr.add_arguments(csv2zarr_parser)
    csv2zarr_parser.set_defaults(func=csv2zarr.main)


def _register_assign(subparsers: argparse._SubParsersAction) -> None:
    try:
        from spatial_tk.commands import assign
    except ImportError:
        p = subparsers.add_parser("assign", help="Assign cell type labels to clusters from enrichment scores")
        p.set_defaults(func=_missing_analysis_subcommand("assign"))
        return

    assign_parser = subparsers.add_parser(
        "assign",
        help="Assign cell type labels to clusters from enrichment scores",
        description=(
            "Read an enrichment score matrix from obsm (produced by quantitate) "
            "and assign a cell type label to each cluster using a configurable strategy. "
            "Optionally runs per-cluster differential expression."
        ),
    )
    assign.add_arguments(assign_parser)
    assign_parser.set_defaults(func=assign.main)


def _register_differential(subparsers: argparse._SubParsersAction) -> None:
    try:
        from spatial_tk.commands import differential
    except ImportError:
        p = subparsers.add_parser("differential", help="Differential expression analysis")
        p.set_defaults(func=_missing_analysis_subcommand("differential"))
        return

    differential_parser = subparsers.add_parser(
        "differential",
        help="Differential expression analysis",
        description="Perform differential analysis between groups or find cluster markers",
    )
    differential.add_arguments(differential_parser)
    differential_parser.set_defaults(func=differential.main)


def image_main() -> None:
    """Handle ``spatial-tk image ...`` after argv has been rewritten."""
    from spatial_tk.utils.helpers import setup_logging
    from spatial_tk.commands import image_group

    setup_logging()
    try:
        image_group.image_main()
    except SystemExit:
        raise
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    """Main entry point for the CLI."""
    argv = sys.argv[1:]
    if argv and argv[0] == "image":
        # Strip the namespace so image argparse sees subcommand first
        sys.argv = [sys.argv[0]] + argv[1:]
        image_main()
        return

    from spatial_tk.utils.helpers import setup_logging

    setup_logging()

    parser = create_parser()
    args = parser.parse_args()

    try:
        args.func(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
