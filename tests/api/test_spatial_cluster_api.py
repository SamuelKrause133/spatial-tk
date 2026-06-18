"""
API integration tests for the spatial clustering step.

Mirrors tests/functional/test_spatial_cluster_command.py but exercises the
core ``spatial_clustering.run_spatial_cluster`` API directly instead of the CLI.
"""

import pytest

from spatial_tk.core import spatial_clustering

pytestmark = pytest.mark.api


def test_run_spatial_cluster_kmeans(spatial_adata):
    adata = spatial_adata.copy()
    adata = spatial_clustering.run_spatial_cluster(
        adata,
        cell_type_key="cell_type_res0p5",
        mode="kmeans",
        max_clusters=10,
        output_key="spatial_cluster_res",
        results_key="spatial_cluster_results",
    )
    assert "spatial_cluster_res" in adata.obs.columns
    assert adata.obs["spatial_cluster_res"].dtype.name == "category"
    assert "spatial_cluster_results" in adata.uns
    result_uns = adata.uns["spatial_cluster_results"]
    assert "silhouette_scores" in result_uns
    assert "best_n_clusters" in result_uns


def test_run_spatial_cluster_hdbscan(spatial_adata):
    adata = spatial_adata.copy()
    adata = spatial_clustering.run_spatial_cluster(
        adata,
        cell_type_key="cell_type_res0p5",
        mode="hdbscan",
        hdbscan_min_cluster_size=5,
        output_key="spatial_cluster_hdbscan",
        results_key="spatial_cluster_hdbscan_results",
    )
    result_uns = adata.uns["spatial_cluster_hdbscan_results"]
    assert result_uns["mode"] == "hdbscan"
    assert "n_clusters_found" in result_uns
    assert "noise_fraction" in result_uns


def test_run_spatial_cluster_force_n_clusters(spatial_adata):
    adata = spatial_adata.copy()
    adata = spatial_clustering.run_spatial_cluster(
        adata,
        cell_type_key="cell_type_res0p5",
        mode="kmeans",
        force_n_clusters=3,
        output_key="spatial_cluster_forced",
        results_key="spatial_cluster_forced_results",
    )
    assert adata.uns["spatial_cluster_forced_results"]["best_n_clusters"] == 3


def test_run_spatial_cluster_resume(spatial_adata):
    adata = spatial_adata.copy()
    adata = spatial_clustering.run_spatial_cluster(
        adata,
        cell_type_key="cell_type_res0p5",
        mode="kmeans",
        max_clusters=10,
        output_key="spatial_cluster_resume",
        results_key="spatial_cluster_resume_results",
    )
    labels_before = adata.obs["spatial_cluster_resume"].copy()
    adata = spatial_clustering.run_spatial_cluster(
        adata,
        cell_type_key="cell_type_res0p5",
        mode="kmeans",
        max_clusters=10,
        output_key="spatial_cluster_resume",
        results_key="spatial_cluster_resume_results",
        resume=True,
    )
    assert adata.obs["spatial_cluster_resume"].equals(labels_before)


def test_run_spatial_cluster_auto_neighbors(assigned_adata):
    # Start from an adata without a spatial graph; run_spatial_cluster should
    # compute connectivities when given neighbor_k.
    adata = assigned_adata.copy()
    adata.obsp.pop("spatial_connectivities", None)
    adata = spatial_clustering.run_spatial_cluster(
        adata,
        cell_type_key="cell_type_res0p5",
        mode="kmeans",
        max_clusters=10,
        neighbor_k=6,
        output_key="spatial_cluster_auto",
        results_key="spatial_cluster_auto_results",
    )
    assert "spatial_connectivities" in adata.obsp
    assert "spatial_cluster_auto" in adata.obs.columns
