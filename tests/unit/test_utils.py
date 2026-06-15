"""
Unit tests for utils module.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from spatial_tk.utils.helpers import parse_resolutions, get_output_path, prepare_spatial_data_for_save


def test_parse_resolutions():
    """Test parsing resolution strings."""
    # Single resolution
    result = parse_resolutions("0.5")
    assert result == [0.5]
    
    # Multiple resolutions
    result = parse_resolutions("0.2,0.5,1.0")
    assert result == [0.2, 0.5, 1.0]
    
    # With spaces
    result = parse_resolutions("0.2, 0.5, 1.0")
    assert result == [0.2, 0.5, 1.0]


def test_parse_resolutions_invalid():
    """Test that invalid resolutions raise an error."""
    with pytest.raises(ValueError, match="Invalid resolution"):
        parse_resolutions("0.5,invalid,1.0")


def test_get_output_path_with_output():
    """Test getting output path when explicit output is provided."""
    result = get_output_path("input.zarr", "output.zarr", False)
    assert result == Path("output.zarr")


def test_get_output_path_with_inplace():
    """Test getting output path when inplace is True."""
    result = get_output_path("input.zarr", None, True)
    assert result == Path("input.zarr")


def test_get_output_path_both_raises_error():
    """Test that providing both output and inplace raises an error."""
    with pytest.raises(ValueError, match="Cannot specify both"):
        get_output_path("input.zarr", "output.zarr", True)


def test_get_output_path_neither_raises_error():
    """Test that providing neither output nor inplace raises an error."""
    with pytest.raises(ValueError, match="Must specify either"):
        get_output_path("input.zarr", None, False)


def test_prepare_spatial_data_keeps_region_categorical(mock_adata):
    """Test that prepare_spatial_data_for_save keeps 'region' as categorical."""
    # Add region and instance_id as categorical (as they come from SpatialData)
    mock_adata.obs['region'] = pd.Categorical(['region1'] * 50 + ['region2'] * 50)
    mock_adata.obs['instance_id'] = pd.Categorical([f'cell_{i}' for i in range(100)])
    
    # Verify they start as categorical
    assert pd.api.types.is_categorical_dtype(mock_adata.obs['region'])
    assert pd.api.types.is_categorical_dtype(mock_adata.obs['instance_id'])
    
    # Run prepare function
    prepare_spatial_data_for_save(mock_adata)
    
    # Verify region and instance_id remain categorical (required by SpatialData)
    assert pd.api.types.is_categorical_dtype(mock_adata.obs['region']), \
        "region must remain categorical for SpatialData"
    assert pd.api.types.is_categorical_dtype(mock_adata.obs['instance_id']), \
        "instance_id must remain categorical for SpatialData"


def test_prepare_spatial_data_converts_other_categoricals(mock_adata):
    """Test that prepare_spatial_data_for_save converts other categorical columns to string."""
    # Add some categorical columns including custom metadata
    mock_adata.obs['sample'] = pd.Categorical(['sample1'] * 50 + ['sample2'] * 50)
    mock_adata.obs['status'] = pd.Categorical(['HIV'] * 50 + ['NEG'] * 50)
    mock_adata.obs['location'] = pd.Categorical(['site1'] * 50 + ['site2'] * 50)
    
    # Run prepare function
    prepare_spatial_data_for_save(mock_adata)
    
    # All non-region/instance_id categorical columns should be converted to string
    assert mock_adata.obs['sample'].dtype == 'object', "sample should be converted to string"
    assert mock_adata.obs['status'].dtype == 'object', "status should be converted to string"
    assert mock_adata.obs['location'].dtype == 'object', "location should be converted to string"

