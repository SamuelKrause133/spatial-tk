"""
Unit tests for spatial_tk.utils.helpers.

Covers save_command_output against real .zarr stores so the user-facing write
path (save_table_only / copy_spatial_store) is exercised directly rather than
only through the CLI subprocess functional tests.
"""

import shutil

import pytest

from spatial_tk.core import data_io
from spatial_tk.utils.helpers import save_command_output


def test_save_command_output_inplace(subsampled_zarr_path, tmp_path):
    """inplace=True overwrites the table in the existing store."""
    if subsampled_zarr_path is None or not subsampled_zarr_path.exists():
        pytest.skip("No ROI fixture zarr found")

    work = tmp_path / "work.zarr"
    shutil.copytree(subsampled_zarr_path, work)

    adata = data_io.load_table_only(work)
    adata.obs["api_marker"] = "inplace"

    save_command_output(adata, work, work, inplace=True)

    reloaded = data_io.load_table_only(work)
    assert "api_marker" in reloaded.obs.columns
    assert (reloaded.obs["api_marker"] == "inplace").all()


def test_save_command_output_copy(subsampled_zarr_path, tmp_path):
    """inplace=False copies the source store, then writes the table into the copy."""
    if subsampled_zarr_path is None or not subsampled_zarr_path.exists():
        pytest.skip("No ROI fixture zarr found")

    src = tmp_path / "src.zarr"
    shutil.copytree(subsampled_zarr_path, src)
    out = tmp_path / "out.zarr"

    adata = data_io.load_table_only(src)
    adata.obs["api_marker"] = "copy"

    save_command_output(adata, src, out, inplace=False)

    assert out.exists()
    reloaded = data_io.load_table_only(out)
    assert "api_marker" in reloaded.obs.columns

    # The source store must be untouched by a non-inplace write.
    src_reloaded = data_io.load_table_only(src)
    assert "api_marker" not in src_reloaded.obs.columns
