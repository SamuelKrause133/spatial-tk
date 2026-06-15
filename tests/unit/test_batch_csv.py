"""Tests for batch manifest CSV reader (stdlib csv, per-command required columns)."""

from __future__ import annotations

from pathlib import Path

import pytest

from spatial_tk.utils.batch_csv import read_batch_manifest, resolve_manifest_path


def test_read_batch_manifest_valid_and_base_dir(tmp_path):
    manifest = tmp_path / "batch.csv"
    manifest.write_text(
        "input_path,bridge_path,zarr_path,extract_path\n"
        "sample.oir,out_bridge,out.zarr,out_chips\n"
    )
    rows, base = read_batch_manifest(manifest, required={"input_path", "bridge_path"})
    assert base == manifest.parent.resolve()
    assert len(rows) == 1
    assert rows[0]["input_path"] == "sample.oir"
    assert rows[0]["bridge_path"] == "out_bridge"
    assert rows[0]["zarr_path"] == "out.zarr"
    assert rows[0]["extract_path"] == "out_chips"


def test_read_batch_manifest_import_subset_no_zarr_column(tmp_path):
    manifest = tmp_path / "batch.csv"
    manifest.write_text("input_path,bridge_path\na.oir,bridge_dir\n")
    rows, _base = read_batch_manifest(manifest, required={"input_path", "bridge_path"})
    assert len(rows) == 1
    assert "zarr_path" not in rows[0]


def test_read_batch_manifest_missing_required_column(tmp_path):
    manifest = tmp_path / "bad.csv"
    manifest.write_text("input_path\na.oir\n")
    with pytest.raises(ValueError, match="missing required columns"):
        read_batch_manifest(manifest, required={"input_path", "bridge_path"})


def test_read_batch_manifest_empty_data(tmp_path):
    manifest = tmp_path / "empty.csv"
    manifest.write_text("input_path,bridge_path,zarr_path\n")
    with pytest.raises(ValueError, match="at least one row"):
        read_batch_manifest(manifest, required={"input_path", "bridge_path"})


def test_read_batch_manifest_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_batch_manifest(tmp_path / "nope.csv", required={"a"})


def test_resolve_manifest_path_relative(tmp_path):
    base = tmp_path.resolve()
    p = resolve_manifest_path(base, "rel/sub.zarr")
    assert p == (base / "rel" / "sub.zarr").resolve()


def test_resolve_manifest_path_absolute(tmp_path):
    base = tmp_path
    abs_p = (tmp_path / "abs.zarr").resolve()
    p = resolve_manifest_path(base, str(abs_p))
    assert p == abs_p
