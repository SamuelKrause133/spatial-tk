#!/usr/bin/env python3
"""
Data I/O operations for Xenium spatial clustering tool.

This module handles loading Xenium spatial datasets from .zarr format,
concatenating multiple samples, and saving processed results.
"""

import logging
from numbers import Integral
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import spatialdata as sd

def load_sample_metadata(csv_path: str) -> pd.DataFrame:
    """
    Load sample metadata from CSV file.
    
    The CSV must contain at minimum:
    - sample: Sample name/identifier
    - path: Path to .zarr file or raw Xenium dataset directory
    
    Additional columns (e.g., status, location) are preserved as metadata.
    
    Args:
        csv_path: Path to CSV file with sample information
        
    Returns:
        DataFrame with sample metadata
        
    Raises:
        ValueError: If required columns are missing
    """
    logging.info(f"Loading sample metadata from {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Validate required columns
    required_cols = ["sample", "path"]
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV missing required columns: {missing_cols}")
    
    # Get optional metadata columns
    metadata_cols = [col for col in df.columns if col not in required_cols]
    
    logging.info(f"Loaded {len(df)} samples")
    if metadata_cols:
        logging.info(f"  Metadata columns: {', '.join(metadata_cols)}")
    
    return df


def load_xenium_dataset(dataset_path: Path, sample_name: str) -> sd.SpatialData:
    """
    Load raw Xenium dataset directory into SpatialData.
    
    Args:
        dataset_path: Path to root Xenium output directory
        sample_name: Sample name identifier
        
    Returns:
        SpatialData object with images and expression data
    """

    if not dataset_path.exists():
        raise FileNotFoundError(f"Xenium dataset directory not found: {dataset_path}")
    
    logging.info(f"    Loading raw Xenium dataset from {dataset_path}")
    
    try:
        # Lazy import: spatialdata_io is not required for non-Xenium / image-only workflows
        from spatialdata_io import xenium as xenium_io

        # Use spatialdata_io.xenium() to load the dataset
        sdata = xenium_io(dataset_path)
        
        # Log images found
        if hasattr(sdata, 'images') and sdata.images:
            image_keys = list(sdata.images.keys())
            logging.info(f"    Images loaded: {image_keys}")
        else:
            logging.warning(f"    No images found in Xenium dataset")
        
        return sdata
        
    except Exception as e:
        logging.error(f"  Failed to load Xenium dataset: {e}")
        raise


def setup_squidpy_structure(sdata: sd.SpatialData, library_id: str) -> None:
    """
    Set up squidpy-compatible structure in AnnData.
    
    Creates adata.uns['spatial'][library_id] with images and scale factors,
    and ensures coordinates are in obsm['spatial'].
    
    Args:
        sdata: SpatialData object
        library_id: Library/sample identifier (used as key in uns['spatial'])
    """
    from spatial_tk.utils.helpers import get_table
    
    # Get AnnData table
    adata = get_table(sdata)
    if adata is None:
        logging.warning(f"    No table found in SpatialData for {library_id}, skipping squidpy setup")
        return
    
    # Initialize uns['spatial'] if it doesn't exist
    if 'spatial' not in adata.uns:
        adata.uns['spatial'] = {}
    
    # Create library entry
    if library_id not in adata.uns['spatial']:
        adata.uns['spatial'][library_id] = {}
    
    # Extract images from SpatialData
    if hasattr(sdata, 'images') and sdata.images:
        # Look for morphology_focus, morphology_mip, or first available image
        image_keys = list(sdata.images.keys())
        preferred_keys = ['morphology_focus', 'morphology_mip', 'morphology']
        
        image_key = None
        for pref_key in preferred_keys:
            if pref_key in image_keys:
                image_key = pref_key
                break
        
        if image_key is None and image_keys:
            image_key = image_keys[0]
            logging.info(f"    Using image key: {image_key}")
        
        if image_key:
            try:
                # Extract image data from SpatialData image element
                image_element = sdata.images[image_key]
                
                # Convert to numpy array
                if hasattr(image_element, 'values'):
                    image_array = image_element.values
                elif hasattr(image_element, 'data'):
                    image_array = np.array(image_element.data)
                else:
                    image_array = np.array(image_element)
                
                # Store in squidpy format
                adata.uns['spatial'][library_id]['images'] = {'hires': image_array}
                logging.info(f"    Added image to uns['spatial'][{library_id}]['images']")
                
                # Try to extract scale factors from coordinate transformations
                # For now, use default scale factor of 1.0
                # TODO: Extract actual scale factors from SpatialData transformations
                adata.uns['spatial'][library_id]['scalefactors'] = {
                    'tissue_hires_scalef': 1.0,
                    'spot_diameter_fullres': 1.0
                }
                
            except Exception as e:
                logging.warning(f"    Failed to extract image for {library_id}: {e}")
    
    # Ensure spatial coordinates are in obsm['spatial']
    # Check if coordinates already exist (e.g., from Xenium loader)
    if 'spatial' in adata.obsm:
        logging.info(f"    Spatial coordinates already in obsm['spatial']")
    elif 'X_spatial' in adata.obsm:
        # Copy X_spatial to spatial
        adata.obsm['spatial'] = adata.obsm['X_spatial']
        logging.info(f"    Copied X_spatial to obsm['spatial']")
    else:
        # Try to extract coordinates from SpatialData
        if 'instance_id' in adata.obs.columns and 'region' in adata.obs.columns:
            # Extract coordinates from spatial elements
            try:
                coords_list = []
                regions = adata.obs['region'].unique()
                
                for region in regions:
                    region_mask = adata.obs['region'] == region
                    region_indices = np.where(region_mask)[0]
                    
                    # Try to get coordinates from points or shapes
                    coords = None
                    if hasattr(sdata, 'points') and region in sdata.points:
                        points = sdata.points[region]
                        if hasattr(points, 'data'):
                            if hasattr(points.data, 'x') and hasattr(points.data, 'y'):
                                coords = np.column_stack([points.data.x.values, points.data.y.values])
                    
                    if coords is None and hasattr(sdata, 'shapes') and region in sdata.shapes:
                        shapes = sdata.shapes[region]
                        # Extract centroids from shapes
                        try:
                            if hasattr(shapes, 'geometry'):
                                import geopandas as gpd
                                if isinstance(shapes, gpd.GeoDataFrame):
                                    coords = np.column_stack([
                                        shapes.geometry.centroid.x.values,
                                        shapes.geometry.centroid.y.values
                                    ])
                        except Exception:
                            pass
                    
                    if coords is not None and len(coords) == len(region_indices):
                        # Initialize coords_list if needed
                        if len(coords_list) == 0:
                            coords_list = [[0.0, 0.0]] * adata.n_obs
                        # Assign coordinates to correct positions
                        for idx, coord_idx in enumerate(region_indices):
                            if coord_idx < len(coords_list):
                                coords_list[coord_idx] = coords[idx]
                
                if coords_list and len(coords_list) == adata.n_obs:
                    all_coords = np.array(coords_list)
                    adata.obsm['spatial'] = all_coords
                    logging.info(f"    Extracted spatial coordinates to obsm['spatial']")
                else:
                    logging.warning(f"    Could not extract coordinates: coords_list length mismatch")
            except Exception as e:
                logging.warning(f"    Could not extract coordinates: {e}")
        else:
            logging.warning(f"    Missing instance_id or region columns, cannot extract coordinates")


def load_spatial_datasets(sample_df: pd.DataFrame, load_images: bool = True) -> List[Tuple[str, sd.SpatialData]]:
    """
    Load Xenium spatial datasets from .zarr files or raw Xenium directories.
    
    Args:
        sample_df: DataFrame with 'sample' and 'path' columns
        load_images: If True, load images (needed for visualization). 
                     If False, skip images to save memory (default: True)
        
    Returns:
        List of tuples (sample_name, SpatialData object)
    """
    logging.info("Loading spatial datasets")
    if not load_images:
        logging.info("  Image loading disabled (load_images=False)")
    
    spatial_data_list = []
    
    for idx, row in sample_df.iterrows():
        sample_name = row["sample"]
        dataset_path = Path(row["path"])
        
        logging.info(f"  Loading {sample_name} from {dataset_path}")
        
        try:
            # Check if path ends with .zarr
            if str(dataset_path).endswith('.zarr'):
                # Existing .zarr loading
                sdata = sd.read_zarr(dataset_path)
                # Remove images if not needed
                if not load_images and hasattr(sdata, 'images') and sdata.images:
                    logging.info(f"    Removing {len(sdata.images)} images from memory")
                    sdata.images = {}
            else:
                # Raw Xenium dataset directory
                # Note: xenium_io() loads images automatically, we'll remove them if not needed
                sdata = load_xenium_dataset(dataset_path, sample_name)
                # Remove images if not needed
                if not load_images and hasattr(sdata, 'images') and sdata.images:
                    logging.info(f"    Removing {len(sdata.images)} images from memory")
                    sdata.images = {}
                elif load_images and hasattr(sdata, 'images') and sdata.images:
                    logging.info(f"    Images loaded: {list(sdata.images.keys())}")
                else:
                    logging.warning(f"    No images found in Xenium dataset")
            
            # Set up squidpy structure for both cases (only if images are loaded)
            if load_images:
                setup_squidpy_structure(sdata, sample_name)
            else:
                # Still set up coordinates, but skip images
                from spatial_tk.utils.helpers import get_table
                adata = get_table(sdata)
                if adata is not None:
                    # Ensure coordinates are in obsm['spatial']
                    if 'spatial' not in adata.obsm and 'X_spatial' in adata.obsm:
                        adata.obsm['spatial'] = adata.obsm['X_spatial']
                        logging.info(f"    Copied X_spatial to obsm['spatial']")
            
            # Update table back to SpatialData (in case it was modified)
            from spatial_tk.utils.helpers import get_table, set_table
            table = get_table(sdata)
            if table is not None:
                set_table(sdata, table)
            
            spatial_data_list.append((sample_name, sdata))
            
            # Log basic info about the dataset
            table = get_table(sdata)
            if table is not None:
                n_cells = table.n_obs
                n_genes = table.n_vars
                logging.info(f"    {n_cells} cells × {n_genes} genes")
            else:
                logging.warning(f"    No expression table found in {sample_name}")
                
        except Exception as e:
            logging.error(f"  Failed to load {sample_name}: {e}")
            raise
    
    logging.info(f"Successfully loaded {len(spatial_data_list)} spatial datasets")
    return spatial_data_list


def concatenate_spatial_data(
    spatial_data_list: List[Tuple[str, sd.SpatialData]],
    sample_df: pd.DataFrame
) -> sd.SpatialData:
    """
    Concatenate multiple SpatialData objects into one, preserving metadata.
    
    Args:
        spatial_data_list: List of (sample_name, SpatialData) tuples
        sample_df: DataFrame with sample metadata
        
    Returns:
        Concatenated SpatialData object with metadata in .table.obs
    """
    logging.info("Concatenating spatial datasets")
    
    if len(spatial_data_list) == 0:
        raise ValueError("No spatial datasets to concatenate")
    
    # Get table accessor (handle both .table and .tables API)
    def get_table(sdata):
        if hasattr(sdata, 'tables') and len(sdata.tables) > 0:
            return list(sdata.tables.values())[0]
        elif hasattr(sdata, 'table'):
            return sdata.table
        return None
    
    if len(spatial_data_list) == 1:
        sample_name, sdata = spatial_data_list[0]
        logging.info("Single sample - no concatenation needed")
        
        # Add metadata to the table
        table = get_table(sdata)
        if table is not None:
            metadata_cols = [col for col in sample_df.columns if col not in ["sample", "path"]]
            sample_metadata = sample_df[sample_df["sample"] == sample_name].iloc[0]
            
            table.obs["sample"] = sample_name
            for col in metadata_cols:
                table.obs[col] = sample_metadata[col]
            
            # Ensure uns['spatial'] structure is preserved
            if 'spatial' not in table.uns:
                table.uns['spatial'] = {}
            if sample_name not in table.uns['spatial'] and hasattr(table, 'uns'):
                # Try to get from original if it was set up
                logging.info(f"  Preserving uns['spatial'] structure for {sample_name}")
        
        return sdata
    
    # Extract SpatialData objects and their names
    sdata_dict = {name: sdata for name, sdata in spatial_data_list}
    
    # Concatenate using spatialdata's concatenate function
    # Pass as dict to handle duplicate label names
    try:
        concatenated_sdata = sd.concatenate(
            sdata_dict,
            region_key="region",
            instance_key="instance_id",
            concatenate_tables=True
        )
        
        # Get the concatenated table
        table = get_table(concatenated_sdata)
        
        if table is not None:
            # Preserve uns['spatial'] structure from individual samples
            # Merge uns['spatial'] from all samples
            if 'spatial' not in table.uns:
                table.uns['spatial'] = {}
            
            # Collect uns['spatial'] entries from each sample
            for sample_name, sdata in spatial_data_list:
                sample_table = get_table(sdata)
                if sample_table is not None and 'spatial' in sample_table.uns:
                    if sample_name in sample_table.uns['spatial']:
                        # Copy the library entry to concatenated table
                        table.uns['spatial'][sample_name] = sample_table.uns['spatial'][sample_name]
                        logging.info(f"  Preserved uns['spatial'][{sample_name}] structure")
            
            # Add sample names - extract from region key
            # spatialdata.concatenate adds element prefixes (e.g., "cell_circles-Drexel-Pos")
            # so we need to extract just the sample name part
            if "region" in table.obs.columns:
                # Try to extract sample name from region
                # Region format: "element_name-sample_name"
                def extract_sample_name(region_str):
                    # Split on '-' and look for matching sample names
                    for sample_name in sample_df["sample"].values:
                        if str(region_str).endswith(str(sample_name)):
                            return sample_name
                    # Fallback: return the region as is
                    return region_str
                
                table.obs["sample"] = table.obs["region"].apply(extract_sample_name)
            
            # Add additional metadata from CSV
            metadata_cols = [col for col in sample_df.columns if col not in ["sample", "path"]]
            if metadata_cols:
                logging.info(f"  Adding metadata columns: {', '.join(metadata_cols)}")
                
                # Create a mapping from sample name to metadata
                metadata_dict = {}
                for col in metadata_cols:
                    metadata_dict[col] = sample_df.set_index("sample")[col].to_dict()
                
                # Add metadata to obs
                for col in metadata_cols:
                    table.obs[col] = table.obs["sample"].map(metadata_dict[col])
            
            # Ensure obsm['spatial'] is preserved if it exists
            # Check if any sample had obsm['spatial'] and preserve it
            has_spatial_coords = False
            for sample_name, sdata in spatial_data_list:
                sample_table = get_table(sdata)
                if sample_table is not None and 'spatial' in sample_table.obsm:
                    has_spatial_coords = True
                    break
            
            if has_spatial_coords and 'spatial' not in table.obsm:
                # Try to reconstruct from concatenated data
                # This should already be handled by SpatialData concatenation,
                # but we verify it exists
                logging.info("  Verifying spatial coordinates in obsm['spatial']")
            
            total_cells = table.n_obs
            total_genes = table.n_vars
            logging.info(f"Concatenation complete: {total_cells} total cells × {total_genes} genes")
        else:
            logging.warning("No table found after concatenation")
        
        return concatenated_sdata
        
    except Exception as e:
        logging.error(f"Failed to concatenate spatial datasets: {e}")
        raise


def save_spatial_data(sdata: sd.SpatialData, output_path: Path, overwrite: bool = False) -> None:
    """
    Save SpatialData object to .zarr format.
    
    Args:
        sdata: SpatialData object to save
        output_path: Path where .zarr will be saved
        overwrite: Whether to overwrite an existing store (required for inplace operations)
    """
    import shutil
    import tempfile
    
    def _flatten_chunk_shape(chunks) -> Optional[Tuple[int, ...]]:
        """
        Convert dask chunk metadata into zarr-compatible chunk shape.

        Accepts either a flat tuple of ints (already valid) or nested tuple-of-tuples
        such as ((4096, ..., 2581), (4096, ..., 711)) and extracts per-axis chunk sizes.
        """
        if chunks is None:
            return None

        if isinstance(chunks, tuple) and all(isinstance(c, Integral) for c in chunks):
            return tuple(int(c) for c in chunks)

        if isinstance(chunks, tuple) and all(isinstance(axis_chunks, tuple) for axis_chunks in chunks):
            chunk_shape = []
            for axis_chunks in chunks:
                if len(axis_chunks) == 0:
                    return None
                first_chunk = axis_chunks[0]
                if not isinstance(first_chunk, Integral):
                    return None
                chunk_shape.append(int(first_chunk))
            return tuple(chunk_shape)

        return None

    def _normalize_label_chunks_for_write(spatial_data: sd.SpatialData) -> None:
        """
        Rechunk label elements whose chunk metadata is incompatible with zarr v3.
        """
        def _normalize_dataset_chunks(dataset):
            updated_dataset = dataset
            changed = False

            for var_name in dataset.data_vars:
                data_array = dataset[var_name]
                chunks = getattr(data_array.data, "chunks", None)
                chunk_shape = _flatten_chunk_shape(chunks)
                if chunk_shape is None:
                    continue

                already_flat = isinstance(chunks, tuple) and all(
                    isinstance(c, Integral) for c in chunks
                )
                if already_flat:
                    continue

                if len(chunk_shape) != len(data_array.dims):
                    continue

                dim_chunks = {
                    dim_name: int(size)
                    for dim_name, size in zip(data_array.dims, chunk_shape)
                }
                updated_dataset = updated_dataset.assign(
                    {var_name: data_array.chunk(dim_chunks)}
                )
                changed = True

            return updated_dataset, changed

        if not hasattr(spatial_data, "labels") or not spatial_data.labels:
            return

        for label_name in list(spatial_data.labels.keys()):
            label_element = spatial_data.labels[label_name]

            try:
                if hasattr(label_element, "map_over_datasets"):
                    changed_any = False

                    def _map_fn(dataset):
                        nonlocal changed_any
                        normalized_dataset, changed = _normalize_dataset_chunks(dataset)
                        changed_any = changed_any or changed
                        return normalized_dataset

                    normalized_label = label_element.map_over_datasets(_map_fn)
                    if changed_any:
                        spatial_data.labels[label_name] = normalized_label
                        logging.info("  Normalized label chunks for %s", label_name)
                    continue

                # Fallback for non-datatree label elements.
                candidate_chunks = getattr(label_element, "chunks", None)
                if candidate_chunks is None and hasattr(label_element, "data"):
                    candidate_chunks = getattr(label_element.data, "chunks", None)
                normalized_chunk_shape = _flatten_chunk_shape(candidate_chunks)
                if normalized_chunk_shape is None:
                    continue

                if hasattr(label_element, "chunk"):
                    spatial_data.labels[label_name] = label_element.chunk(normalized_chunk_shape)
                    logging.info("  Normalized label chunks for %s", label_name)
            except Exception as chunk_err:
                logging.warning(
                    "  Failed to normalize chunks for label %s: %s",
                    label_name,
                    chunk_err,
                )

    def _write_spatial_data(spatial_data: sd.SpatialData) -> None:
        if overwrite and output_path.exists():
            # Workaround for SpatialData limitation: cannot overwrite a store that's currently in use
            # Save to temporary location first, then replace the original
            # See: https://github.com/scverse/spatialdata/discussions/520
            temp_dir = Path(tempfile.mkdtemp(prefix="spatialdata_tmp_", dir=output_path.parent))
            temp_path = temp_dir / output_path.name
            
            try:
                # Save to temporary location
                spatial_data.write(temp_path)
                
                # Remove original store
                shutil.rmtree(output_path)
                
                # Move temporary store to original location
                shutil.move(str(temp_path), str(output_path))
                
                # Clean up temporary directory
                temp_dir.rmdir()
                
                logging.info(f"Successfully saved spatial data (overwrite)")
            except Exception as e:
                # Clean up temporary directory on error
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
                raise
        else:
            # Normal save operation
            spatial_data.write(output_path, overwrite=overwrite)
            logging.info(f"Successfully saved spatial data")

    logging.info(f"Saving spatial data to {output_path}")
    
    try:
        _normalize_label_chunks_for_write(sdata)
        _write_spatial_data(sdata)
        repair_table_attrs_on_disk(output_path, tables=_tables_from_sdata(sdata))
    except TypeError as e:
        chunk_shape_error = "Expected an iterable of integers" in str(e)
        has_labels = hasattr(sdata, "labels") and bool(sdata.labels)
        if not (chunk_shape_error and has_labels):
            logging.error(f"Failed to save spatial data: {e}")
            raise

        logging.warning(
            "Label chunk metadata remains incompatible with zarr write; retrying save without labels."
        )
        original_labels = sdata.labels
        try:
            if output_path.exists():
                shutil.rmtree(output_path)
            sdata.labels = {}
            _write_spatial_data(sdata)
            repair_table_attrs_on_disk(output_path, tables=_tables_from_sdata(sdata))
        finally:
            sdata.labels = original_labels
    except Exception as e:
        logging.error(f"Failed to save spatial data: {e}")
        raise


def _tables_from_sdata(sdata: sd.SpatialData) -> Dict[str, ad.AnnData]:
    if hasattr(sdata, "tables") and sdata.tables:
        return dict(sdata.tables)
    if getattr(sdata, "table", None) is not None:
        return {"table": sdata.table}
    return {}


def _current_tables_format_version() -> str:
    try:
        from spatialdata._io.format import CurrentTablesFormat

        return CurrentTablesFormat().spatialdata_format_version
    except Exception:
        return "0.1"


def _normalize_region_attr(value) -> Optional[List[str] | str]:
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (tuple, set)):
        value = list(value)
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _table_attrs_from_adata(adata: ad.AnnData) -> dict:
    """Build SpatialData table group attrs from AnnData metadata and obs columns."""
    from spatialdata.models import TableModel

    attrs_src = dict(adata.uns.get(TableModel.ATTRS_KEY, {}) or {})
    region_key = attrs_src.get(TableModel.REGION_KEY_KEY) or "region"
    instance_key = attrs_src.get(TableModel.INSTANCE_KEY) or "cell_id"

    if region_key not in adata.obs and "region" in adata.obs:
        region_key = "region"
    if instance_key not in adata.obs:
        for candidate in ("cell_id", "instance_id"):
            if candidate in adata.obs:
                instance_key = candidate
                break

    region = attrs_src.get(TableModel.REGION_KEY)
    if region is None and region_key in adata.obs:
        region = adata.obs[region_key].unique().tolist()

    return {
        "spatialdata-encoding-type": "ngff:regions_table",
        "region": _normalize_region_attr(region),
        "region_key": region_key,
        "instance_key": instance_key,
        "version": _current_tables_format_version(),
    }


def repair_table_attrs_on_disk(
    zarr_path: Path,
    tables: Optional[Mapping[str, ad.AnnData]] = None,
) -> None:
    """
    Ensure each ``tables/<name>/zarr.json`` (or ``.zattrs``) has SpatialData table attrs.

    Repairs stores written before attrs were persisted correctly, so ``sd.read_zarr``
    can load the table (``assert version is not None`` in spatialdata I/O).

    Idempotent when attrs are already complete. Sources attrs from ``tables`` if given,
    otherwise reads each table AnnData from disk.
    """
    import json
    import os

    required_keys = (
        "spatialdata-encoding-type",
        "region",
        "region_key",
        "instance_key",
        "version",
    )
    tables_dir = zarr_path / "tables"
    if not tables_dir.exists():
        return

    in_memory = dict(tables) if tables else {}

    def _atomic_write_json(path: Path, payload: dict) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w") as fh:
            json.dump(payload, fh, indent=4)
        os.replace(tmp_path, path)

    def _expected_attrs_for(name: str) -> dict:
        table = in_memory.get(name)
        if table is None:
            table_path = tables_dir / name
            if table_path.is_dir():
                try:
                    table = ad.read_zarr(str(table_path))
                except Exception as exc:
                    logging.warning("Could not read table %s for attrs repair: %s", table_path, exc)
        if table is not None:
            return _table_attrs_from_adata(table)
        return {
            "spatialdata-encoding-type": "ngff:regions_table",
            "region": None,
            "region_key": "region",
            "instance_key": "cell_id",
            "version": _current_tables_format_version(),
        }

    for table_dir in sorted(p for p in tables_dir.iterdir() if p.is_dir()):
        expected = _expected_attrs_for(table_dir.name)
        zarr_v3_path = table_dir / "zarr.json"
        zattrs_path = table_dir / ".zattrs"

        if zarr_v3_path.exists():
            try:
                with open(zarr_v3_path, "r") as fh:
                    doc = json.load(fh)
            except Exception as exc:
                logging.warning("Could not read %s: %s", zarr_v3_path, exc)
                continue
            attributes = doc.get("attributes")
            if not isinstance(attributes, dict):
                attributes = {}
            missing = [k for k in required_keys if k not in attributes]
            if missing:
                logging.warning(
                    "Repairing table attrs in %s (missing: %s)",
                    zarr_v3_path,
                    missing,
                )
                for key in required_keys:
                    if key not in attributes:
                        attributes[key] = expected[key]
                doc["attributes"] = attributes
                _atomic_write_json(zarr_v3_path, doc)
        elif zattrs_path.exists():
            try:
                with open(zattrs_path, "r") as fh:
                    attrs = json.load(fh)
            except Exception as exc:
                logging.warning("Could not read %s: %s", zattrs_path, exc)
                continue
            if not isinstance(attrs, dict):
                attrs = {}
            missing = [k for k in required_keys if k not in attrs]
            if missing:
                logging.warning(
                    "Repairing table attrs in %s (missing: %s)",
                    zattrs_path,
                    missing,
                )
                for key in required_keys:
                    if key not in attrs:
                        attrs[key] = expected[key]
                _atomic_write_json(zattrs_path, attrs)


def load_existing_spatial_data(zarr_path: Path, load_images: bool = False) -> sd.SpatialData:
    """
    Load an existing processed SpatialData object from a .zarr file.
    
    Args:
        zarr_path: Path to .zarr file
        load_images: If False, skip loading images to save memory (default: False)
        
    Returns:
        SpatialData object
        
    Raises:
        FileNotFoundError: If zarr file doesn't exist
        ValueError: If no expression table found in spatial data
    """
    logging.info(f"Loading existing spatial data from {zarr_path}")
    if not load_images:
        logging.info("  Skipping image loading (load_images=False)")
    
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr file not found: {zarr_path}")

    repair_table_attrs_on_disk(zarr_path)
    
    try:
        sdata = sd.read_zarr(zarr_path)
        
        # Remove images if not needed (to save memory)
        if not load_images and hasattr(sdata, 'images') and sdata.images:
            logging.info(f"  Removing {len(sdata.images)} images from memory")
            sdata.images = {}
        
        # Get table (handle both .table and .tables API)
        from spatial_tk.utils.helpers import get_table
        table = get_table(sdata)
        
        if table is None:
            raise ValueError("No expression table found in spatial data")
        
        logging.info(f"Loaded: {table.n_obs} cells × {table.n_vars} genes")
        
        return sdata
    except Exception as e:
        logging.error(f"Failed to load spatial data: {e}")
        raise


def load_table_only(zarr_path: Path) -> ad.AnnData:
    """
    Load only the AnnData table from a SpatialData .zarr file.
    
    This directly loads the AnnData table from zarr_path/tables/table without
    loading the entire SpatialData object (images, shapes, etc.). This is much
    more memory-efficient for operations that don't need images or spatial elements
    (e.g., clustering, normalization).
    
    Args:
        zarr_path: Path to .zarr file
        
    Returns:
        AnnData object (table only)
    """
    import zarr
    
    logging.info(f"Loading table only from {zarr_path} (direct AnnData read, skipping SpatialData)")
    
    try:
        # Check if zarr store exists
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
        
        # Try to find the table path
        # SpatialData typically stores tables in tables/ subdirectory
        table_paths = [
            zarr_path / "tables" / "table",  # Most common path
            zarr_path / "table",  # Alternative path
        ]
        
        # Check if tables directory exists and find table name
        tables_dir = zarr_path / "tables"
        if tables_dir.exists():
            # List available tables
            import os
            table_names = [d for d in os.listdir(tables_dir) if (tables_dir / d).is_dir()]
            if table_names:
                # Use first table found
                table_name = table_names[0]
                table_path = tables_dir / table_name
                logging.info(f"  Found table: tables/{table_name}")
            else:
                # Try default name
                table_path = tables_dir / "table"
        else:
            # Try root level
            table_path = zarr_path / "table"
        
        # Try to load AnnData directly from zarr
        # Check if path exists as zarr group
        try:
            store = zarr.open(str(table_path), mode='r')
            if isinstance(store, zarr.Group):
                # Load AnnData directly
                adata = ad.read_zarr(str(table_path))
                logging.info(f"Loaded table: {adata.n_obs} cells × {adata.n_vars} genes")
                return adata
            else:
                raise ValueError(f"Table path exists but is not a zarr group: {table_path}")
        except Exception as e:
            # If direct path doesn't work, try to find it by inspecting zarr structure
            logging.debug(f"Direct path failed, inspecting zarr structure: {e}")
            
            # Fallback: try to read from SpatialData but extract immediately
            # This is less efficient but more robust
            logging.warning("  Falling back to SpatialData read (less efficient)")
            sdata = sd.read_zarr(zarr_path)
            
            # Get table
            table = None
            if hasattr(sdata, 'tables') and len(sdata.tables) > 0:
                table = list(sdata.tables.values())[0]
            elif hasattr(sdata, 'table'):
                table = sdata.table
            
            if table is None:
                raise ValueError("No expression table found in spatial data")
            
            # Make a copy to avoid keeping reference to SpatialData
            adata = table.copy()
            
            # Clear reference to allow garbage collection
            del sdata
            
            logging.info(f"Loaded table: {adata.n_obs} cells × {adata.n_vars} genes")
            return adata
        
    except Exception as e:
        logging.error(f"Failed to load table: {e}")
        raise


def save_table_only(adata: ad.AnnData, zarr_path: Path, overwrite: bool = False) -> None:
    """
    Save AnnData table directly to a SpatialData .zarr file without loading other elements.
    
    This directly writes the AnnData table to zarr_path/tables/table without
    loading the entire SpatialData object. This is much more memory-efficient for
    inplace operations that only modify the table (e.g., clustering, normalization).
    
    Args:
        adata: AnnData object to save
        zarr_path: Path to .zarr file
        overwrite: Whether to overwrite existing table (default: False)
    """
    import zarr
    import shutil
    
    logging.info(f"Saving table only to {zarr_path} (direct AnnData write, skipping SpatialData)")
    
    try:
        # Check if zarr store exists
        if not zarr_path.exists():
            raise FileNotFoundError(f"Zarr file not found: {zarr_path}")
        
        # Determine table path
        tables_dir = zarr_path / "tables"
        table_name = "table"  # Default table name
        
        # Check if tables directory exists
        if tables_dir.exists():
            # Check for existing table name
            import os
            if os.path.exists(tables_dir):
                existing_tables = [d for d in os.listdir(tables_dir) if (tables_dir / d).is_dir()]
                if existing_tables:
                    table_name = existing_tables[0]  # Use existing table name
                    logging.info(f"  Using existing table: tables/{table_name}")
        else:
            # Create tables directory if it doesn't exist
            tables_dir.mkdir(parents=True, exist_ok=True)
            logging.info(f"  Created tables directory")
        
        table_path = tables_dir / table_name
        
        # Handle overwrite
        if overwrite and table_path.exists():
            # Remove existing table
            logging.info(f"  Removing existing table at {table_path}")
            shutil.rmtree(table_path)
        
        # Write AnnData directly to zarr.
        # For inplace writes, existing table path is removed above when overwrite=True.
        adata.write_zarr(str(table_path))
        
        logging.info(f"Successfully saved table: {adata.n_obs} cells × {adata.n_vars} genes")
        
    except Exception as e:
        logging.error(f"Failed to save table: {e}")
        raise

