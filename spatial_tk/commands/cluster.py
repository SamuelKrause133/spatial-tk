#!/usr/bin/env python3
"""
Cluster command: Perform PCA, neighbor graph, UMAP, and Leiden clustering.
"""

import argparse
import logging
import sys
from pathlib import Path

from spatial_tk.core.data_io import load_table_only
from spatial_tk.utils.helpers import (
    get_output_path,
    parse_resolutions,
    save_command_output,
)
from spatial_tk.utils.config import load_config, merge_config_with_args


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for the cluster command.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        '--input',
        required=False,
        help='Path to input normalized .zarr file'
    )
    parser.add_argument(
        '--output',
        help='Path to output .zarr file (required unless --inplace is used)'
    )
    parser.add_argument(
        '--inplace',
        action='store_true',
        help='Modify the input file in place instead of creating a new file'
    )
    parser.add_argument(
        '--leiden-resolution',
        type=str,
        default='0.5',
        help='Leiden clustering resolution(s), comma-separated for multiple (default: 0.5)'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Generate and save UMAP plots'
    )
    parser.add_argument(
        '--config',
        help='Path to TOML configuration file (optional)'
    )


def main(args: argparse.Namespace) -> None:
    """
    Execute the cluster command.
    
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
            args = merge_config_with_args('cluster', config_dict, args, temp_parser)
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Validate required arguments (after config merge)
    if not args.input:
        logging.error("--input is required (provide via CLI or config file)")
        sys.exit(1)
    
    logging.info("="*60)
    logging.info("Xenium Process: Clustering Analysis")
    logging.info("="*60)
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    try:
        output_path = get_output_path(args.input, args.output, args.inplace)
        resolutions = parse_resolutions(args.leiden_resolution)
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)
    
    try:
        from spatial_tk.core import clustering, plotting

        adata = load_table_only(input_path)
        
        if adata is None:
            raise ValueError("No expression table found in spatial data")
        
        logging.info(f"Starting clustering: {adata.n_obs} cells × {adata.n_vars} genes")
        
        # Dimensionality reduction
        adata = clustering.run_pca(adata)
        adata = clustering.compute_neighbors_and_umap(adata)
        
        # Clustering at multiple resolutions
        for resolution in resolutions:
            res_str = str(resolution).replace(".", "p")
            cluster_key = f"leiden_res{res_str}"
            adata = clustering.cluster_leiden(adata, resolution, key_added=cluster_key)
        
        # Save results
        logging.info(f"Saving results to: {output_path}")
        save_command_output(adata, input_path, output_path, inplace=args.inplace)
        
        # Generate plots if requested
        if args.save_plots:
            plots_dir = output_path.parent / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            for resolution in resolutions:
                res_str = str(resolution).replace(".", "p")
                cluster_key = f"leiden_res{res_str}"
                plotting.save_umap_plots(adata, plots_dir, cluster_key, None, resolution)
        
        logging.info("="*60)
        logging.info(f"Clustering complete: {output_path}")
        logging.info(f"Resolutions: {resolutions}")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"Clustering failed: {e}", exc_info=True)
        sys.exit(1)

