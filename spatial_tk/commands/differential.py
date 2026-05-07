#!/usr/bin/env python3
"""
Differential command: Perform differential analysis using obs variables and obsm embeddings.

Supports two modes:
- Mode A: Group comparisons (e.g., HIV vs NEG status)
- Mode B: Marker genes per cluster
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, List

import pandas as pd

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
        '--obsm-layer',
        help='Optional obsm layer to use for enrichment-based differential analysis (e.g., "score_mlm_PanglaoDB")'
    )
    parser.add_argument(
        '--method',
        default='wilcoxon',
        choices=['wilcoxon', 't-test', 'logreg'],
        help='Statistical test method for gene expression DE (default: wilcoxon)'
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


def perform_gene_expression_de(
    adata,
    groupby: str,
    compare_groups: Optional[List[str]],
    method: str,
    layer: Optional[str],
    output_dir: Path,
    n_genes: int
) -> None:
    """
    Perform differential expression analysis on gene expression data.
    
    Args:
        adata: AnnData object
        groupby: Column to group by
        compare_groups: Optional list of 2 groups to compare
        method: Statistical test method
        layer: Layer to use for expression data
        output_dir: Output directory
        n_genes: Number of top genes to save
    """
    import scanpy as sc

    logging.info("Performing gene expression differential analysis")
    
    if compare_groups and len(compare_groups) == 2:
        # Mode A: Compare two specific groups
        logging.info(f"  Comparing {compare_groups[0]} vs {compare_groups[1]}")
        
        # Filter to only these groups
        mask = adata.obs[groupby].isin(compare_groups)
        adata_subset = adata[mask].copy()
        
        # Run differential expression
        sc.tl.rank_genes_groups(
            adata_subset,
            groupby=groupby,
            groups=[compare_groups[0]],
            reference=compare_groups[1],
            method=method,
            layer=layer
        )
        
        # Save results
        result_df = sc.get.rank_genes_groups_df(adata_subset, group=compare_groups[0])
        
        # Add comparison info
        result_df['group1'] = compare_groups[0]
        result_df['group2'] = compare_groups[1]
        
        output_file = output_dir / f"de_genes_{compare_groups[0]}_vs_{compare_groups[1]}.csv"
        result_df.to_csv(output_file, index=False)
        logging.info(f"  Saved gene DE results to {output_file}")
        
        # Save top N genes
        top_genes = result_df.head(n_genes)
        top_file = output_dir / f"de_genes_top{n_genes}_{compare_groups[0]}_vs_{compare_groups[1]}.csv"
        top_genes.to_csv(top_file, index=False)
        
    else:
        # Mode B: Find markers for all groups
        logging.info(f"  Finding marker genes for all groups in {groupby}")
        
        sc.tl.rank_genes_groups(
            adata,
            groupby=groupby,
            method=method,
            layer=layer
        )
        
        # Save results for all groups
        result_df = sc.get.rank_genes_groups_df(adata, group=None)
        
        output_file = output_dir / f"de_genes_all_groups_{groupby}.csv"
        result_df.to_csv(output_file, index=False)
        logging.info(f"  Saved gene DE results to {output_file}")
        
        # Save top N per group
        top_df = result_df.groupby('group').head(n_genes)
        top_file = output_dir / f"de_genes_top{n_genes}_per_group_{groupby}.csv"
        top_df.to_csv(top_file, index=False)


def perform_obsm_de(
    adata,
    groupby: str,
    obsm_layer: str,
    compare_groups: Optional[List[str]],
    output_dir: Path,
    n_top: int = 50
) -> None:
    """
    Perform differential analysis on obsm embeddings (e.g., MLM scores).
    
    Args:
        adata: AnnData object
        groupby: Column to group by
        obsm_layer: Name of obsm layer (e.g., "score_mlm_PanglaoDB")
        compare_groups: Optional list of 2 groups to compare
        output_dir: Output directory
        n_top: Number of top features to report
    """
    logging.info(f"Performing obsm differential analysis on {obsm_layer}")
    
    if obsm_layer not in adata.obsm:
        logging.warning(f"  obsm layer '{obsm_layer}' not found. Skipping.")
        return
    
    # Get the embedding matrix
    embedding = adata.obsm[obsm_layer]
    
    # Create DataFrame with proper column names
    if hasattr(embedding, 'var_names'):
        # It's an AnnData-like object
        feature_names = list(embedding.var_names)
        embedding_df = pd.DataFrame(embedding.X, columns=feature_names, index=adata.obs_names)
    else:
        # It's a plain array
        feature_names = [f"feature_{i}" for i in range(embedding.shape[1])]
        embedding_df = pd.DataFrame(embedding, columns=feature_names, index=adata.obs_names)
    
    # Add grouping variable
    embedding_df[groupby] = adata.obs[groupby].values
    
    if compare_groups and len(compare_groups) == 2:
        # Mode A: Compare two specific groups
        logging.info(f"  Comparing {compare_groups[0]} vs {compare_groups[1]}")
        
        group1_data = embedding_df[embedding_df[groupby] == compare_groups[0]][feature_names]
        group2_data = embedding_df[embedding_df[groupby] == compare_groups[1]][feature_names]
        
        # Calculate mean difference and t-test
        from scipy import stats
        
        results = []
        for feature in feature_names:
            g1_vals = group1_data[feature].values
            g2_vals = group2_data[feature].values
            
            mean_diff = g1_vals.mean() - g2_vals.mean()
            statistic, pval = stats.ttest_ind(g1_vals, g2_vals)
            
            results.append({
                'feature': feature,
                'mean_group1': g1_vals.mean(),
                'mean_group2': g2_vals.mean(),
                'mean_difference': mean_diff,
                't_statistic': statistic,
                'pvalue': pval,
                'group1': compare_groups[0],
                'group2': compare_groups[1]
            })
        
        result_df = pd.DataFrame(results)
        
        # Sort by absolute mean difference
        result_df = result_df.sort_values('mean_difference', ascending=False, key=abs)
        
        output_file = output_dir / f"de_{obsm_layer}_{compare_groups[0]}_vs_{compare_groups[1]}.csv"
        result_df.to_csv(output_file, index=False)
        logging.info(f"  Saved obsm DE results to {output_file}")
        
        # Save top N
        top_file = output_dir / f"de_{obsm_layer}_top{n_top}_{compare_groups[0]}_vs_{compare_groups[1]}.csv"
        result_df.head(n_top).to_csv(top_file, index=False)
        
    else:
        # Mode B: Mean values per group
        logging.info(f"  Calculating mean {obsm_layer} values per group")
        
        group_means = embedding_df.groupby(groupby)[feature_names].mean()
        
        output_file = output_dir / f"mean_{obsm_layer}_per_group_{groupby}.csv"
        group_means.to_csv(output_file)
        logging.info(f"  Saved obsm group means to {output_file}")


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
        
        # Validate compare groups if provided
        if compare_groups:
            unique_groups = adata.obs[args.groupby].unique()
            for group in compare_groups:
                if group not in unique_groups:
                    logging.error(f"Group '{group}' not found in column '{args.groupby}'")
                    logging.info(f"Available groups: {', '.join(map(str, unique_groups))}")
                    sys.exit(1)
        
        # Perform gene expression differential analysis
        perform_gene_expression_de(
            adata,
            args.groupby,
            compare_groups,
            args.method,
            args.layer,
            output_dir,
            args.n_genes
        )
        
        # Perform obsm differential analysis if requested
        if args.obsm_layer:
            perform_obsm_de(
                adata,
                args.groupby,
                args.obsm_layer,
                compare_groups,
                output_dir
            )
        
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
                
                plotting.save_de_plots(adata, plots_dir, args.groupby, resolution)
            
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

