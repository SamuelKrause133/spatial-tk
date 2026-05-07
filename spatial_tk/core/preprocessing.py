#!/usr/bin/env python3
"""
Preprocessing functions for spatial transcriptomics data.

This module handles quality control, filtering, normalization,
and feature selection for Xenium spatial datasets.
"""

import logging
from typing import Optional

import anndata as ad
import numpy as np
import scanpy as sc


def calculate_qc_metrics(adata: ad.AnnData, resume: bool = False) -> ad.AnnData:
    """
    Calculate quality control metrics including mitochondrial, ribosomal,
    and hemoglobin gene percentages.
    
    Args:
        adata: AnnData object
        resume: If True, skip if QC metrics already exist
        
    Returns:
        AnnData object with QC metrics added
    """
    if resume and "pct_counts_mt" in adata.obs.columns:
        logging.info("QC metrics already calculated (resuming)")
        return adata
    
    logging.info("Calculating QC metrics")
    
    # Identify mitochondrial genes (MT- for human, Mt- for mouse)
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    
    # Identify ribosomal genes
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    
    # Identify hemoglobin genes
    adata.var["hb"] = adata.var_names.str.contains("^HB[^(P)]")
    
    # Calculate QC metrics
    # Limit percent_top to available gene count to avoid IndexError
    n_genes = adata.n_vars
    percent_top = [x for x in [20, 50, 100] if x < n_genes]
    if not percent_top:
        percent_top = [min(20, n_genes)]
    
    sc.pp.calculate_qc_metrics(
        adata,
        qc_vars=["mt", "ribo", "hb"],
        percent_top=percent_top,
        inplace=True,
        log1p=True
    )
    
    logging.info(f"QC metrics calculated - Median genes/cell: {np.median(adata.obs['n_genes_by_counts']):.0f}")
    logging.info(f"Median UMI/cell: {np.median(adata.obs['total_counts']):.0f}")
    logging.info(f"Median MT%: {np.median(adata.obs['pct_counts_mt']):.2f}%")
    
    return adata


def filter_cells_and_genes(
    adata: ad.AnnData,
    min_genes: int = 100,
    min_cells: int = 3
) -> ad.AnnData:
    """
    Filter cells and genes based on minimum thresholds.
    
    Args:
        adata: AnnData object
        min_genes: Minimum number of genes expressed per cell
        min_cells: Minimum number of cells expressing a gene
        
    Returns:
        Filtered AnnData object
    """
    logging.info(f"Filtering cells (min_genes={min_genes}) and genes (min_cells={min_cells})")
    n_cells_before = adata.n_obs
    n_genes_before = adata.n_vars
    
    # Filter cells with too few genes
    sc.pp.filter_cells(adata, min_genes=min_genes)
    
    # Filter genes expressed in too few cells
    sc.pp.filter_genes(adata, min_cells=min_cells)
    
    logging.info(f"Filtered: {n_cells_before - adata.n_obs} cells, {n_genes_before - adata.n_vars} genes")
    logging.info(f"Remaining: {adata.n_obs} cells × {adata.n_vars} genes")
    
    return adata


def normalize_and_log(adata: ad.AnnData, resume: bool = False) -> ad.AnnData:
    """
    Normalize to median total counts and apply log transformation.
    
    Args:
        adata: AnnData object
        resume: If True, skip if normalization already done
        
    Returns:
        Normalized AnnData object
    """
    if resume and "counts" in adata.layers:
        logging.info("Normalization already done (resuming)")
        return adata
    
    logging.info("Normalizing and log-transforming data")
    
    # Save raw counts in layers
    adata.layers["counts"] = adata.X.copy()
    
    # Normalize to median total counts
    sc.pp.normalize_total(adata)
    
    # Log transform
    sc.pp.log1p(adata)
    
    logging.info("Normalization complete")
    return adata


def select_variable_genes(adata: ad.AnnData, n_top_genes: int = 2000, resume: bool = False) -> ad.AnnData:
    """
    Select highly variable genes for downstream analysis.
    
    Args:
        adata: AnnData object
        n_top_genes: Number of highly variable genes to select
        resume: If True, skip if highly variable genes already computed
        
    Returns:
        AnnData object with highly variable genes annotated
    """
    if resume and "highly_variable" in adata.var.columns:
        logging.info(f"Highly variable genes already selected (resuming)")
        return adata
    
    logging.info(f"Selecting {n_top_genes} highly variable genes")
    
    # Use batch_key if available for batch-aware feature selection
    batch_key = "sample" if "sample" in adata.obs.columns else None
    
    sc.pp.highly_variable_genes(
        adata,
        n_top_genes=n_top_genes,
        batch_key=batch_key
    )
    
    n_hvg = adata.var["highly_variable"].sum()
    logging.info(f"Selected {n_hvg} highly variable genes")
    
    return adata

