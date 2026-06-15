#!/usr/bin/env python3
"""
Concat command: Concatenate multiple Xenium .zarr files into one.
"""

import argparse
import logging
import sys
from pathlib import Path

from spatial_tk.core.data_io import (
    load_sample_metadata,
    load_spatial_datasets,
    concatenate_spatial_data,
    save_spatial_data
)
from spatial_tk.core.downsample import downsample_cells
from spatial_tk.utils.helpers import get_table
from spatial_tk.utils.config import load_config, merge_config_with_args


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Add arguments for the concat command.
    
    Args:
        parser: ArgumentParser to add arguments to
    """
    parser.add_argument(
        '--input',
        required=False,
        help='Path to CSV file listing samples (columns: sample, path, [metadata...])'
    )
    parser.add_argument(
        '--output',
        required=False,
        help='Path to output concatenated .zarr file'
    )
    parser.add_argument(
        '--downsample',
        type=float,
        default=1.0,
        help='Fraction of cells to keep (0-1, default: 1.0 = no downsampling)'
    )
    parser.add_argument(
        '--config',
        help='Path to TOML configuration file (optional)'
    )


def main(args: argparse.Namespace) -> None:
    """
    Execute the concat command.
    
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
            args = merge_config_with_args('concat', config_dict, args, temp_parser)
        except Exception as e:
            logging.error(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Validate required arguments (after config merge)
    if not args.input:
        logging.error("--input is required (provide via CLI or config file)")
        sys.exit(1)
    if not args.output:
        logging.error("--output is required (provide via CLI or config file)")
        sys.exit(1)
    
    logging.info("="*60)
    logging.info("Xenium Process: Concatenate Datasets")
    logging.info("="*60)
    
    # Validate inputs
    input_csv = Path(args.input)
    if not input_csv.exists():
        logging.error(f"Input CSV file not found: {input_csv}")
        sys.exit(1)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load sample metadata from CSV
        sample_df = load_sample_metadata(str(input_csv))
        
        # Load spatial datasets
        spatial_data_list = load_spatial_datasets(sample_df)
        
        # Concatenate into single SpatialData object
        sdata = concatenate_spatial_data(spatial_data_list, sample_df)
        
        # Extract AnnData table for downsampling if requested
        adata = get_table(sdata)
        
        if adata is None:
            raise ValueError("No expression table found in spatial data")
        
        logging.info(f"Concatenated dataset: {adata.n_obs} cells × {adata.n_vars} genes")
        
        # Downsample if requested
        if args.downsample < 1.0:
            adata = downsample_cells(adata, args.downsample)
            # Update table in sdata
            from spatial_tk.utils.helpers import set_table
            set_table(sdata, adata)
        
        # Save results
        save_spatial_data(sdata, output_path)
        
        logging.info("="*60)
        logging.info(f"Concatenation complete: {output_path}")
        logging.info(f"Final dataset: {adata.n_obs} cells × {adata.n_vars} genes")
        logging.info("="*60)
        
    except Exception as e:
        logging.error(f"Concatenation failed: {e}", exc_info=True)
        sys.exit(1)

