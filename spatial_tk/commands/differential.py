#!/usr/bin/env python3
"""
Differential command: Perform differential analysis using obs variables and obsm embeddings.

Supports two modes:
- Mode A: Group comparisons (e.g., HIV vs NEG status)
- Mode B: Marker genes per cluster

This command is a thin I/O wrapper over :mod:`spatial_tk.core.differential`.
"""

import argparse
import logging
import sys
from pathlib import Path

from spatial_tk.core import differential as differential_core
from spatial_tk.core.data_io import load_existing_spatial_data
from spatial_tk.utils.helpers import get_table
from spatial_tk.utils.config import load_config, merge_config_with_args


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for the differential command.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        '--input',
        required=False,
        help='Path to input .zarr file with annotations'
    )
    parser.add_argument(
        '--output-dir',
        required=False,
        help='Directory to save differential analysis results'
    )
    parser.add_argument(
        '--groupby',
        required=False,
        help='Column in obs to group by for differential analysis (e.g., "status", "cell_type", or "leiden_res0p5")'
    )
    parser.add_argument(
        '--compare-groups',
        help='Comma-separated list of exactly 2 groups to compare (Mode A). E.g., "HIV,NEG". If not provided, finds markers for all groups (Mode B)'
    )
    parser.add_argument(
        '--within',
        help='Optional obs column whose categories stratify the analysis (e.g., "cell_type"). The differential analysis is run separately within each category.'
    )
    parser.add_argument(
        '--within-subset',
        help='Comma-separated subset of --within categories to restrict the analysis to (e.g., "T cells,B cells"). Requires --within.'
    )
    parser.add_argument(
        '--on',
        help='Data source: "gene_expression" (default), a layer name, or an obsm key (e.g., "score_mlm_PanglaoDB").'
    )
    parser.add_argument(
        '--obsm-layer',
        help='Deprecated: alias for --on pointing at an obsm key.'
    )
    parser.add_argument(
        '--method',
        default=None,
        choices=['wilcoxon', 't-test', 'logreg', 'ttest', 'means', 'rankby'],
        help='Statistical engine. Gene expression: wilcoxon (default), t-test, logreg. obsm: ttest, means, rankby.'
    )
    parser.add_argument(
        '--layer',
        default=None,
        help='Layer to use for gene expression (default: None uses .X)'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Generate and save differential analysis plots'
    )
    parser.add_argument(
        '--n-genes',
        type=int,
        default=100,
        help='Number of top genes to save per group (default: 100)'
    )
    parser.add_argument(
        '--config',
        help='Path to TOML configuration file (optional)'
    )


def main(args: argparse.Namespace) -> None:
    """
    Execute the differential command.
    
    Args:
        args: Parsed command-line arguments
    """
    # Load and merge config if provided
    if args.config:
        try:
            config_dict = load_config(args.config)
            # Create a temporary parser to get defaults
            temp_parser = argparse.ArgumentParser()
            add_arguments(temp_parser)
            args = merge_config_with_args('differential', config_dict, args, temp_parser)
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Validate required arguments (after config merge)
    if not args.input:
        logging.error("--input is required (provide via CLI or config file)")
        sys.exit(1)
    if not args.output_dir:
        logging.error("--output-dir is required (provide via CLI or config file)")
        sys.exit(1)
    if not args.groupby:
        logging.error("--groupby is required (provide via CLI or config file)")
        sys.exit(1)
    
    logging.info("="*60)
    logging.info("Xenium Process: Differential Analysis")
    logging.info("="*60)
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse compare groups if provided
    compare_groups = None
    if args.compare_groups:
        compare_groups = [g.strip() for g in args.compare_groups.split(',')]
        if len(compare_groups) != 2:
            logging.error("--compare-groups must specify exactly 2 groups")
            sys.exit(1)

    # Parse within_subset if provided
    within_subset = None
    if args.within_subset:
        within_subset = [s.strip() for s in args.within_subset.split(',')]
    if within_subset and not args.within:
        logging.error("--within-subset requires --within")
        sys.exit(1)

    # Resolve the data source. Precedence: --on, then --obsm-layer (deprecated
    # alias), then --layer (gene expression on a specific layer), else .X.
    if args.on:
        on = args.on
        if args.obsm_layer:
            logging.warning("--obsm-layer ignored because --on was provided")
    elif args.obsm_layer:
        logging.warning("--obsm-layer is deprecated; use --on instead")
        on = args.obsm_layer
    elif args.layer:
        on = args.layer
    else:
        on = "gene_expression"
    
    try:
        import scanpy as sc
        from spatial_tk.core import plotting

        # Load spatial data
        sdata = load_existing_spatial_data(input_path)
        adata = get_table(sdata)
        
        if adata is None:
            raise ValueError("No expression table found in spatial data")
        
        logging.info(f"Starting differential analysis: {adata.n_obs} cells × {adata.n_vars} genes")
        
        # Validate groupby column
        if args.groupby not in adata.obs.columns:
            logging.error(f"Column '{args.groupby}' not found in obs")
            logging.info(f"Available columns: {', '.join(adata.obs.columns)}")
            sys.exit(1)

        # Validate within column if provided
        if args.within and args.within not in adata.obs.columns:
            logging.error(f"Column '{args.within}' not found in obs")
            logging.info(f"Available columns: {', '.join(adata.obs.columns)}")
            sys.exit(1)

        # Validate the data source resolves to a layer or obsm key
        is_obsm = on in adata.obsm
        if not (on in ("gene_expression", "X") or on in adata.layers or is_obsm):
            logging.error(
                f"--on '{on}' is not 'gene_expression'/'X', a layer, or an obsm key"
            )
            logging.info(f"Available layers: {list(adata.layers)}")
            logging.info(f"Available obsm keys: {list(adata.obsm.keys())}")
            sys.exit(1)

        # Validate method against the resolved source
        if args.method:
            ge_methods = {'wilcoxon', 't-test', 'logreg'}
            obsm_methods = {'ttest', 'means', 'rankby'}
            allowed = obsm_methods if is_obsm else ge_methods
            if args.method not in allowed:
                logging.error(
                    f"--method '{args.method}' is not valid for this source; "
                    f"allowed: {sorted(allowed)}"
                )
                sys.exit(1)

        # Validate compare groups if provided
        if compare_groups:
            unique_groups = adata.obs[args.groupby].unique()
            for group in compare_groups:
                if group not in unique_groups:
                    logging.error(f"Group '{group}' not found in column '{args.groupby}'")
                    logging.info(f"Available groups: {', '.join(map(str, unique_groups))}")
                    sys.exit(1)

        # Run differential analysis via the unified core API
        results = differential_core.run_differential(
            adata,
            args.groupby,
            on=on,
            compare_groups=compare_groups,
            within=args.within,
            within_subset=within_subset,
            method=args.method,
        )

        # Save results
        differential_core.save_differential_results(
            results,
            output_dir,
            groupby=args.groupby,
            compare_groups=compare_groups,
            within=args.within,
            n_top=args.n_genes,
        )

        # adata holding the rank results (subset copy in Mode A, full adata otherwise)
        de_adata = results.adata

        # Generate plots if requested
        if args.save_plots:
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            # If we have leiden clustering results, create standard DE plots
            if args.groupby.startswith('leiden_res'):
                res_str = args.groupby.replace("leiden_res", "")
                try:
                    resolution = float(res_str.replace("p", "."))
                except ValueError:
                    resolution = None
                
                plotting.save_de_plots(de_adata, plots_dir, args.groupby, resolution)
            
            # UMAP colored by groupby variable
            if "X_umap" in adata.obsm:
                try:
                    sc.pl.umap(adata, color=args.groupby, show=False)
                    import matplotlib.pyplot as plt
                    plt.savefig(plots_dir / f"umap_{args.groupby}.png", bbox_inches="tight", dpi=150)
                    plt.close()
                    logging.info(f"  Saved UMAP plot colored by {args.groupby}")
                except Exception as e:
                    logging.warning(f"  Could not generate UMAP plot: {e}")
        
        logging.info("="*60)
        logging.info(f"Differential analysis complete")
        logging.info(f"Results saved to: {output_dir}")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"Differential analysis failed: {e}", exc_info=True)
        sys.exit(1)
