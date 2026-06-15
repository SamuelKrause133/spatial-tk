"""
Unit tests for preprocessing module.
"""

import pytest
import numpy as np
from spatial_tk.core import preprocessing
from spatial_tk.core.downsample import downsample_cells


def test_calculate_qc_metrics(mock_adata):
    """Test QC metrics calculation."""
    adata = preprocessing.calculate_qc_metrics(mock_adata)
    
    # Check that QC metrics were added
    assert 'n_genes_by_counts' in adata.obs.columns
    assert 'total_counts' in adata.obs.columns
    assert 'pct_counts_mt' in adata.obs.columns
    
    # Check that var columns were added
    assert 'mt' in adata.var.columns
    assert 'ribo' in adata.var.columns
    assert 'hb' in adata.var.columns


def test_filter_cells_and_genes(mock_adata):
    """Test cell and gene filtering."""
    n_obs_before = mock_adata.n_obs
    n_vars_before = mock_adata.n_vars
    
    adata = preprocessing.filter_cells_and_genes(mock_adata, min_genes=5, min_cells=1)
    
    # Check that filtering occurred
    assert adata.n_obs <= n_obs_before
    assert adata.n_vars <= n_vars_before


def test_normalize_and_log(mock_adata):
    """Test normalization and log transformation."""
    adata = preprocessing.normalize_and_log(mock_adata)
    
    # Check that layers were created
    assert 'counts' in adata.layers
    
    # Check that data was normalized (values should be different from raw)
    assert not np.array_equal(adata.X, adata.layers['counts'])


def test_select_variable_genes(mock_adata):
    """Test highly variable gene selection."""
    # Normalize first
    adata = preprocessing.normalize_and_log(mock_adata)
    
    n_top = 20
    adata = preprocessing.select_variable_genes(adata, n_top_genes=n_top)
    
    # Check that highly_variable column was added
    assert 'highly_variable' in adata.var.columns
    
    # Check that correct number of genes were selected
    assert adata.var['highly_variable'].sum() <= n_top


def test_downsample_cells(mock_adata):
    """Test cell downsampling."""
    n_cells_original = mock_adata.n_obs
    fraction = 0.5
    
    adata = downsample_cells(mock_adata, fraction)
    
    # Check that approximately correct number of cells remain
    expected_cells = int(n_cells_original * fraction)
    assert abs(adata.n_obs - expected_cells) <= 2  # Allow small variance
    
    
def test_downsample_invalid_fraction(mock_adata):
    """Test that invalid fractions raise errors."""
    with pytest.raises(ValueError):
        downsample_cells(mock_adata, -0.1)

    with pytest.raises(ValueError):
        downsample_cells(mock_adata, 1.5)

