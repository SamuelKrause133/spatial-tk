"""
Unit tests for data_io module.
"""

import pytest
import pandas as pd
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from spatial_tk.core import data_io
from spatial_tk.core.visualization import _image_extent_for_coords


def test_load_sample_metadata(test_samples_csv):
    """Test loading sample metadata from CSV."""
    df = data_io.load_sample_metadata(str(test_samples_csv))
    
    # Check that required columns exist
    assert 'sample' in df.columns
    assert 'path' in df.columns
    
    # Check that data was loaded
    assert len(df) > 0


def test_load_sample_metadata_missing_columns():
    """Test that missing columns raise an error."""
    # Create a temporary CSV without required columns
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write("wrong_column\n")
        f.write("value\n")
        temp_path = f.name
    
    try:
        with pytest.raises(ValueError, match="missing required columns"):
            data_io.load_sample_metadata(temp_path)
    finally:
        Path(temp_path).unlink()


def test_load_existing_spatial_data(subsampled_zarr_path):
    """Test loading existing spatial data."""
    if subsampled_zarr_path is None:
        pytest.skip("No subsampled zarr file available")
    
    sdata = data_io.load_existing_spatial_data(subsampled_zarr_path)
    
    # Check that spatial data was loaded
    assert sdata is not None
    
    # Check that table exists
    from spatial_tk.utils.helpers import get_table
    table = get_table(sdata)
    assert table is not None
    assert table.n_obs > 0
    assert table.n_vars > 0


def test_load_image_source_uses_zarr_loader(tmp_path):
    zarr_path = tmp_path / "sample.zarr"
    zarr_path.mkdir()
    mock_sdata = MagicMock()
    mock_sdata.images = {"morphology_focus": MagicMock()}

    with patch("spatial_tk.core.data_io.load_existing_spatial_data") as mock_load_existing:
        mock_load_existing.return_value = mock_sdata
        loaded = data_io.load_image_source(zarr_path)

    assert loaded is mock_sdata
    mock_load_existing.assert_called_once_with(zarr_path, load_images=True)


def test_load_image_source_uses_raw_loader(tmp_path):
    raw_path = tmp_path / "xenium_raw"
    raw_path.mkdir()
    mock_sdata = MagicMock()
    mock_sdata.images = {"morphology_focus": MagicMock()}

    with patch("spatial_tk.core.data_io.load_xenium_dataset") as mock_load_raw:
        mock_load_raw.return_value = mock_sdata
        loaded = data_io.load_image_source(raw_path, sample_name="sampleA")

    assert loaded is mock_sdata
    mock_load_raw.assert_called_once_with(raw_path, sample_name="sampleA")


def test_copy_spatial_store_copies_directory_tree(tmp_path):
    src = tmp_path / "input.zarr"
    dst = tmp_path / "output.zarr"
    (src / "images").mkdir(parents=True)
    (src / "images" / "dummy.txt").write_text("ok", encoding="utf-8")

    data_io.copy_spatial_store(src, dst, overwrite=False)

    assert (dst / "images" / "dummy.txt").exists()


def test_scale_xy_image_extent_matches_spatial_coordinates():
    coords = pd.DataFrame({"x": [0.0, 10.0], "y": [0.0, 20.0]}).to_numpy()

    extent = _image_extent_for_coords(
        full_width=100,
        full_height=200,
        coords=coords,
        image_transform="scale_xy",
    )

    assert extent == (0.0, 10.0, 0.0, 20.0)

