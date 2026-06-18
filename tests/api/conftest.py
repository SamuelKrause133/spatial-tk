"""
Staged pipeline fixtures for the API integration tests.

These fixtures build a dependency chain that mirrors the spatial-tk pipeline
(concat -> normalize -> cluster -> quantitate -> assign -> spatial neighbors)
using the core Python API directly, rather than the CLI. Each stage is
session-scoped and starts from a ``.copy()`` of the previous stage so that
individual tests remain isolated while the expensive setup runs only once.

Every fixture calls ``pytest.skip`` when the test data is unavailable, using
the same data-availability guard as the functional tests.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatial_tk.core import annotation, clustering, preprocessing, spatial_neighbors
from spatial_tk.core import data_io
from spatial_tk.utils.helpers import get_table

# Cluster key used consistently across the staged fixtures and tests.
LEIDEN_KEY = "leiden_res0p5"
ANNOTATION_KEY = "cell_type_res0p5"
SCORE_KEY = "score_mlm_custom"

_TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"


def _resolve_samples_csv() -> Path:
    """Session-scoped equivalent of the top-level ``test_samples_csv`` fixture."""
    tier = os.getenv("SPATIAL_TK_TEST_TIER", "fast").strip().lower()
    if tier == "full":
        override = os.getenv("SPATIAL_TK_FULL_SAMPLES_CSV")
        return Path(override) if override else _TEST_DATA_DIR / "test_samples_full.csv"
    override = os.getenv("SPATIAL_TK_FAST_SAMPLES_CSV")
    return Path(override) if override else _TEST_DATA_DIR / "test_samples_fast.csv"


def _resolve_markers_csv() -> Path:
    return _TEST_DATA_DIR / "test_markers.csv"


@pytest.fixture(scope="session")
def samples_csv_path() -> Path:
    """Session-scoped samples CSV path (avoids function-scope mismatch)."""
    return _resolve_samples_csv()


@pytest.fixture(scope="session")
def markers_csv_path() -> Path:
    """Session-scoped markers CSV path (avoids function-scope mismatch)."""
    return _resolve_markers_csv()


def _select_pipeline_samples(samples_csv) -> pd.DataFrame:
    """
    Pick a small, multi-status subset of samples for the staged pipeline.

    We deliberately span both ``status`` groups (and locations where possible)
    so the differential / obsm comparison tests have two groups to contrast,
    while keeping the cell count low enough for fast session setup.
    """
    df = pd.read_csv(samples_csv)
    if "status" in df.columns:
        parts = [grp.head(2) for _, grp in df.groupby("status", sort=False)]
        subset = pd.concat(parts, ignore_index=True)
    else:
        subset = df.head(2)
    return subset


@pytest.fixture(scope="session")
def pipeline_sample_df(samples_csv_path):
    """Multi-status sample subset used to build the staged pipeline."""
    if not samples_csv_path.exists():
        pytest.skip("Test samples CSV not found")
    subset = _select_pipeline_samples(samples_csv_path)
    # All referenced zarrs must exist; otherwise downstream loading fails.
    missing = [p for p in subset["path"] if not Path(p).exists()]
    if missing:
        pytest.skip(f"ROI fixture zarrs not available: {missing}")
    return subset


@pytest.fixture(scope="session")
def raw_sdata(pipeline_sample_df):
    """Concatenated SpatialData built from the sample subset (no images)."""
    sdata_list = data_io.load_spatial_datasets(pipeline_sample_df, load_images=False)
    return data_io.concatenate_spatial_data(sdata_list, pipeline_sample_df)


@pytest.fixture(scope="session")
def raw_adata(raw_sdata):
    """Raw concatenated AnnData table (mirrors the concat step)."""
    adata = get_table(raw_sdata)
    if adata is None:
        pytest.skip("No expression table found in concatenated SpatialData")
    return adata


@pytest.fixture(scope="session")
def normalized_adata(raw_adata):
    """QC + filter + normalize + HVG selection (mirrors the normalize step)."""
    adata = raw_adata.copy()
    adata = preprocessing.calculate_qc_metrics(adata)
    adata = preprocessing.filter_cells_and_genes(adata, min_genes=10, min_cells=3)
    adata = preprocessing.normalize_and_log(adata)
    adata = preprocessing.select_variable_genes(adata, n_top_genes=500)
    return adata


@pytest.fixture(scope="session")
def clustered_adata(normalized_adata):
    """PCA + neighbors/UMAP + Leiden clustering (mirrors the cluster step)."""
    adata = normalized_adata.copy()
    adata = clustering.run_pca(adata)
    adata = clustering.compute_neighbors_and_umap(adata)
    adata = clustering.cluster_leiden(adata, resolution=0.5, key_added=LEIDEN_KEY)
    return adata


@pytest.fixture(scope="session")
def marker_net_df(markers_csv_path):
    """Decoupler-format network built from the test marker CSV."""
    if not markers_csv_path.exists():
        pytest.skip("Test markers CSV not found")
    markers = annotation.load_marker_genes(str(markers_csv_path))
    return annotation.markers_dict_to_dataframe(markers)


@pytest.fixture(scope="session")
def scored_adata(clustered_adata, marker_net_df):
    """Enrichment scoring added to obsm (mirrors the quantitate step)."""
    adata = clustered_adata.copy()
    adata = annotation.run_enrichment_scoring(
        adata, marker_net_df, score_key="custom", method="mlm", tmin=1
    )
    if SCORE_KEY not in adata.obsm:
        pytest.skip(
            f"Enrichment scoring did not produce obsm['{SCORE_KEY}'] "
            "(marker genes likely absent from the panel)"
        )
    return adata


@pytest.fixture(scope="session")
def assigned_adata(scored_adata):
    """Cluster -> cell-type assignment for every leiden_res* column (assign step)."""
    adata = scored_adata.copy()
    for col in [c for c in adata.obs.columns if c.startswith("leiden_res")]:
        res_str = col.replace("leiden_res", "")
        annotation.assign_clusters(
            adata,
            score_key=SCORE_KEY,
            cluster_key=col,
            annotation_key=f"cell_type_res{res_str}",
            strategy="top_positive",
        )
    return adata


@pytest.fixture(scope="session")
def spatial_adata(assigned_adata):
    """Spatial neighbor graph added (mirrors the spatial_neighbors step)."""
    adata = assigned_adata.copy()
    adata = spatial_neighbors.compute_spatial_neighbors(
        adata, spatial_key="spatial", n_neighs=6, key_added="spatial"
    )
    return adata


@pytest.fixture
def viz_coords_obs(subsampled_zarr_path):
    """
    (coords, obs) from a single contiguous ROI fixture zarr.

    Mirrors tests/functional/test_visualize_cli.py, which renders a single ROI
    fixture. Unlike the concatenated ``spatial_adata`` (which has large empty
    regions between samples), this single ROI is spatially contiguous, so
    random-ROI windows reliably contain points.
    """
    import spatialdata as sd

    if subsampled_zarr_path is None or not Path(subsampled_zarr_path).exists():
        pytest.skip("No ROI fixture zarr found")
    sdata = sd.read_zarr(subsampled_zarr_path)
    adata = get_table(sdata)
    if adata is None:
        pytest.skip("No expression table in ROI fixture zarr")
    coords = np.asarray(adata.obsm["spatial"])
    return coords, adata.obs.copy()
