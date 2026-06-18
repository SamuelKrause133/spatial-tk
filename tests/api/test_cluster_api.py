"""
API integration tests for the clustering step.

Mirrors the cluster steps in tests/functional/test_full_pipeline.py but
exercises the core ``clustering`` API directly instead of the CLI.
"""

from unittest.mock import patch

import pytest

from spatial_tk.core import clustering

pytestmark = pytest.mark.api


def test_run_pca_adds_x_pca(normalized_adata):
    adata = normalized_adata.copy()
    adata = clustering.run_pca(adata)
    assert "X_pca" in adata.obsm


def test_compute_neighbors_and_umap(normalized_adata):
    adata = normalized_adata.copy()
    adata = clustering.run_pca(adata)
    adata = clustering.compute_neighbors_and_umap(adata)
    assert "X_umap" in adata.obsm
    assert "neighbors" in adata.uns


def test_cluster_leiden_adds_obs_column(clustered_adata):
    adata = clustered_adata
    assert "leiden_res0p5" in adata.obs.columns
    assert adata.obs["leiden_res0p5"].dtype.name == "category"
    assert adata.obs["leiden_res0p5"].nunique() >= 2


def test_cluster_multiple_resolutions(clustered_adata):
    adata = clustered_adata.copy()
    for res in [0.3, 0.9]:
        key = f"leiden_res{str(res).replace('.', 'p')}"
        adata = clustering.cluster_leiden(adata, resolution=res, key_added=key)
    assert "leiden_res0p3" in adata.obs.columns
    assert "leiden_res0p9" in adata.obs.columns


def test_cluster_resume_skips_recompute(clustered_adata):
    adata = clustered_adata.copy()
    with patch("scanpy.tl.leiden") as mock_leiden:
        clustering.cluster_leiden(
            adata, resolution=0.5, key_added="leiden_res0p5", resume=True
        )
    assert not mock_leiden.called
