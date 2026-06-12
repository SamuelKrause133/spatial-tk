#!/usr/bin/env python3
"""
Normalize command: Perform QC, filtering, normalization, and feature selection.
"""

import argparse
import logging
import sys
from pathlib import Path

from spatial_tk.core.data_io import (
    copy_spatial_store,
    load_existing_spatial_data,
    load_table_only,
    save_spatial_data,
    save_table_only,
)
from spatial_tk.utils.helpers import (
    get_output_path,
    get_table,
    prepare_spatial_data_for_save,
    set_table,
)
from spatial_tk.utils.config import load_config, merge_config_with_args


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for the normalize command.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        '--input',
        required=False,
        help='Path to input .zarr file'
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
        '--min-genes',
        type=int,
        default=100,
        help='Minimum number of genes expressed per cell (default: 100)'
    )
    parser.add_argument(
        '--min-cells',
        type=int,
        default=3,
        help='Minimum number of cells expressing a gene (default: 3)'
    )
    parser.add_argument(
        '--n-top-genes',
        type=int,
        default=2000,
        help='Number of highly variable genes to select (default: 2000)'
    )
    parser.add_argument(
        '--save-plots',
        action='store_true',
        help='Generate and save QC plots'
    )
    parser.add_argument(
        '--config',
        help='Path to TOML configuration file (optional)'
    )


def main(args: argparse.Namespace) -> None:
    """
    Execute the normalize command.
    
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
            args = merge_config_with_args('normalize', config_dict, args, temp_parser)
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Validate required arguments (after config merge)
    if not args.input:
        logging.error("--input is required (provide via CLI or config file)")
        sys.exit(1)
    
    logging.info("="*60)
    logging.info("Xenium Process: Normalize and Preprocess")
    logging.info("="*60)
    
    # Validate inputs
    input_path = Path(args.input)
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    try:
        output_path = get_output_path(args.input, args.output, args.inplace)
    except ValueError as e:
        logging.error(str(e))
        sys.exit(1)
    
    try:
        from spatial_tk.core import plotting, preprocessing

        adata = load_table_only(input_path)
        
        if adata is None:
            raise ValueError("No expression table found in spatial data")
        
        logging.info(f"Starting normalization: {adata.n_obs} cells × {adata.n_vars} genes")
        
        # Make names unique
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        
        # QC and filtering
        adata = preprocessing.calculate_qc_metrics(adata)
        adata = preprocessing.filter_cells_and_genes(adata, args.min_genes, args.min_cells)
        
        # Normalization and feature selection
        adata = preprocessing.normalize_and_log(adata)
        adata = preprocessing.select_variable_genes(adata, args.n_top_genes)
        
        # Prepare for saving
        prepare_spatial_data_for_save(adata)
        
        # Save results
        if args.inplace:
            logging.info(f"Saving results in place: {output_path}")
            save_table_only(adata, output_path, overwrite=True)
        else:
            copy_spatial_store(input_path, output_path, overwrite=False)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Saving results to: {output_path}")
            save_table_only(adata, output_path, overwrite=True)
        
        # Generate plots if requested
        if args.save_plots:
            plots_dir = output_path.parent / "plots"
            plots_dir.mkdir(exist_ok=True)
            plotting.save_qc_plots(adata, plots_dir)
        
        logging.info("="*60)
        logging.info(f"Normalization complete: {output_path}")
        logging.info(f"Final dataset: {adata.n_obs} cells × {adata.n_vars} genes")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"Normalization failed: {e}", exc_info=True)
        sys.exit(1)

