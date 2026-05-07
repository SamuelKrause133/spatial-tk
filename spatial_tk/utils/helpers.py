#!/usr/bin/env python3
"""
Utility helper functions for spatial_tk.

This module contains common functionality used across multiple commands.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import anndata as ad
    import pandas as pd
    import spatialdata as sd


def setup_logging(level: int = logging.INFO):
    """Configure logging to show INFO level messages with timestamps."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_table(sdata: "sd.SpatialData", table_key: Optional[str] = None) -> Optional["ad.AnnData"]:
    """
    Get AnnData table from SpatialData object.
    
    Handles both .table and .tables API for compatibility.
    
    Args:
        sdata: SpatialData object
        
    Returns:
        AnnData table or None if not found
    """
    # Optional dependency: only required for analysis/image pipelines that load SpatialData.
    import anndata as ad  # noqa: F401

    if hasattr(sdata, 'tables') and len(sdata.tables) > 0:
        if table_key:
            return sdata.tables.get(table_key)
        return list(sdata.tables.values())[0]
    elif hasattr(sdata, 'table'):
        return sdata.table
    return None


def set_table(
    sdata: "sd.SpatialData",
    adata: "ad.AnnData",
    table_key: Optional[str] = None,
) -> None:
    """
    Set AnnData table in SpatialData object.
    
    Handles both .table and .tables API for compatibility.
    
    Args:
        sdata: SpatialData object
        adata: AnnData table to set
    """
    import anndata as ad  # noqa: F401

    if hasattr(sdata, 'tables') and len(sdata.tables) > 0:
        # Get the table name
        table_name = table_key or list(sdata.tables.keys())[0]
        if table_key and table_name not in sdata.tables:
            raise KeyError(f"Table key not found in SpatialData.tables: {table_key}")
        sdata.tables[table_name] = adata
    else:
        sdata.table = adata


def prepare_spatial_data_for_save(adata: "ad.AnnData") -> None:
    """
    Prepare AnnData object for saving in SpatialData format.
    
    SpatialData requires 'region' and 'instance_id' to remain categorical.
    Other categorical columns are converted to string to avoid issues during save/load.
    
    Args:
        adata: AnnData object to prepare
    """
    import pandas as pd

    def _coerce_scalar(value):
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return ""
            return _coerce_scalar(value[0])
        # Handles numpy arrays and pandas array scalars without importing numpy.
        if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
            converted = value.tolist()
            if isinstance(converted, list):
                if len(converted) == 0:
                    return ""
                return _coerce_scalar(converted[0])
            return converted
        return value

    # SpatialData requires these keys to be categorical and hashable scalars.
    for required_col in ["region", "instance_id"]:
        if required_col in adata.obs:
            adata.obs[required_col] = adata.obs[required_col].map(_coerce_scalar)
            adata.obs[required_col] = pd.Categorical(adata.obs[required_col])

    # SpatialData validation expects hashable iterables in uns metadata.
    spatial_attrs = adata.uns.get("spatialdata_attrs")
    if isinstance(spatial_attrs, dict) and "region" in spatial_attrs:
        region_meta = spatial_attrs["region"]
        if hasattr(region_meta, "tolist"):
            region_meta = region_meta.tolist()
        if isinstance(region_meta, (tuple, set)):
            region_meta = list(region_meta)
        if isinstance(region_meta, list):
            spatial_attrs["region"] = [_coerce_scalar(v) for v in region_meta]
        else:
            spatial_attrs["region"] = [_coerce_scalar(region_meta)]

    categorical_cols = adata.obs.select_dtypes(include=['category']).columns
    for col in categorical_cols:
        # Keep region and instance_id as categorical (required by SpatialData)
        # Convert all other categorical columns to string
        if col not in ['region', 'instance_id']:
            adata.obs[col] = adata.obs[col].astype(str)


def parse_resolutions(resolution_str: str) -> list[float]:
    """
    Parse comma-separated resolution string into list of floats.
    
    Args:
        resolution_str: Comma-separated string of resolutions (e.g., "0.2,0.5,1.0")
        
    Returns:
        List of resolution values
        
    Raises:
        ValueError: If any resolution value is invalid
    """
    resolutions = []
    for res in resolution_str.split(","):
        try:
            resolutions.append(float(res.strip()))
        except ValueError:
            raise ValueError(f"Invalid resolution value: {res}")
    return resolutions


def get_output_path(input_path: str, output_path: Optional[str], inplace: bool) -> Path:
    """
    Determine the output path based on input, output, and inplace flags.
    
    Args:
        input_path: Input file path
        output_path: Explicit output path (if provided)
        inplace: Whether to modify file in place
        
    Returns:
        Path object for output
        
    Raises:
        ValueError: If both output_path and inplace are specified
    """
    if output_path and inplace:
        raise ValueError("Cannot specify both --output and --inplace")
    
    if inplace:
        return Path(input_path)
    elif output_path:
        return Path(output_path)
    else:
        raise ValueError("Must specify either --output or --inplace")

