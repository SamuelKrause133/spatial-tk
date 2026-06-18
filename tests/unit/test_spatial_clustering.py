"""
Unit tests for spatial clustering core utilities.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import anndata as ad
from scipy import sparse


def _make_adata():
    X = np.ones((6, 4), dtype=float)
    obs = pd.DataFrame(
        {
            "cell_type": pd.Categorical(["A", "A", "B", "B", "C", "C"]),
        },
        index=[f"cell_{i}" for i in range(6)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(4)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    rows = np.array([0, 0, 1, 1, 2, 3, 4, 5])
    cols = np.array([1, 2, 0, 2, 0, 4, 3, 4])
    data = np.ones(len(rows), dtype=float)
    adata.obsp["spatial_connectivities"] = sparse.csr_matrix((data, (rows, cols)), shape=(6, 6))
    return adata


def test_build_neighborhood_composition_shapes_and_categories():
    from spatial_tk.core import spatial_clustering

    adata = _make_adata()
    out = spatial_clustering.build_neighborhood_composition(
        adata,
        connectivities_key="spatial_connectivities",
        cell_type_key="cell_type",
        include_self=True,
        normalize=False,
    )

    composition = out["composition"]
    assert composition.shape == (adata.n_obs, 3)
    assert out["cell_type_categories"] == ["A", "B", "C"]
    assert out["neighbor_counts"].shape == (adata.n_obs,)


def test_include_self_changes_neighbor_counts():
    from spatial_tk.core import spatial_clustering

    adata = _make_adata()
    out_include = spatial_clustering.build_neighborhood_composition(
        adata, "spatial_connectivities", "cell_type", include_self=True, normalize=False
    )
    out_exclude = spatial_clustering.build_neighborhood_composition(
        adata, "spatial_connectivities", "cell_type", include_self=False, normalize=False
    )
    assert np.all(out_include["neighbor_counts"] >= out_exclude["neighbor_counts"])
    assert np.any(out_include["neighbor_counts"] > out_exclude["neighbor_counts"])


def test_normalize_composition_row_sums_one_for_nonzero_rows():
    from spatial_tk.core import spatial_clustering

    adata = _make_adata()
    out = spatial_clustering.build_neighborhood_composition(
        adata, "spatial_connectivities", "cell_type", include_self=True, normalize=True
    )
    row_sums = out["composition"].sum(axis=1)
    assert np.allclose(row_sums, np.ones(adata.n_obs))


def test_run_spatial_kmeans_returns_full_sweep():
    from spatial_tk.core import spatial_clustering

    rng = np.random.default_rng(7)
    composition = rng.normal(size=(40, 5))
    out = spatial_clustering.run_spatial_kmeans(
        composition=composition,
        min_clusters=2,
        max_clusters=6,
        random_state=0,
    )
    assert out["n_clusters"] == [2, 3, 4, 5, 6]
    assert len(out["silhouette_scores"]) == 5
    assert len(out["labels_by_n_clusters"]["2"]) == 40
    assert out["selection_method"] == "silhouette_max"


def test_run_spatial_kmeans_force_n_clusters_overrides_selection():
    from spatial_tk.core import spatial_clustering

    rng = np.random.default_rng(11)
    composition = rng.normal(size=(40, 6))
    out = spatial_clustering.run_spatial_kmeans(
        composition=composition,
        min_clusters=2,
        max_clusters=8,
        random_state=0,
        force_n_clusters=5,
    )
    assert out["best_n_clusters"] == 5
    assert out["force_n_clusters"] == 5
    assert out["selection_method"] == "forced"
    assert out["silhouette_best_n_clusters"] in [2, 3, 4, 5, 6, 7, 8]


def test_run_spatial_hdbscan_returns_labels_and_noise_counts():
    from spatial_tk.core import spatial_clustering

    rng = np.random.default_rng(21)
    cluster1 = rng.normal(loc=0.0, scale=0.2, size=(30, 5))
    cluster2 = rng.normal(loc=3.0, scale=0.2, size=(30, 5))
    composition = np.vstack([cluster1, cluster2])

    out = spatial_clustering.run_spatial_hdbscan(
        composition=composition,
        min_cluster_size=5,
        min_samples=2,
        cluster_selection_epsilon=0.0,
        metric="euclidean",
        allow_single_cluster=False,
    )
    assert out["mode"] == "hdbscan"
    assert len(out["labels"]) == composition.shape[0]
    assert out["n_clusters_found"] >= 1
    assert out["n_noise"] >= 0
    assert 0.0 <= out["noise_fraction"] <= 1.0


def test_run_spatial_hdbscan_silhouette_can_be_none():
    from spatial_tk.core import spatial_clustering

    # Constant vectors tend to yield one cluster or all noise.
    composition = np.zeros((20, 4), dtype=float)
    out = spatial_clustering.run_spatial_hdbscan(
        composition=composition,
        min_cluster_size=5,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        metric="euclidean",
        allow_single_cluster=True,
    )
    assert "silhouette_score" in out
    assert out["silhouette_score"] is None or isinstance(out["silhouette_score"], float)


# ---------------------------------------------------------------------------
# run_spatial_cluster orchestrator
# ---------------------------------------------------------------------------
def _make_clusterable_adata(n_per_type=8):
    """Build an adata with a clear two-block neighborhood structure."""
    cell_types = (["A"] * n_per_type) + (["B"] * n_per_type)
    n = len(cell_types)
    X = np.ones((n, 3), dtype=float)
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(cell_types)},
        index=[f"cell_{i}" for i in range(n)],
    )
    var = pd.DataFrame(index=[f"g{i}" for i in range(3)])
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Connect cells within each block so compositions separate the two blocks.
    rows, cols = [], []
    for block_start in (0, n_per_type):
        members = list(range(block_start, block_start + n_per_type))
        for i in members:
            for j in members:
                if i != j:
                    rows.append(i)
                    cols.append(j)
    data = np.ones(len(rows), dtype=float)
    adata.obsp["spatial_connectivities"] = sparse.csr_matrix(
        (data, (rows, cols)), shape=(n, n)
    )
    adata.obsm["spatial"] = np.random.default_rng(0).normal(size=(n, 2))
    return adata


def test_run_spatial_cluster_kmeans():
    from spatial_tk.core import spatial_clustering

    adata = _make_clusterable_adata()
    out = spatial_clustering.run_spatial_cluster(
        adata,
        cell_type_key="cell_type",
        mode="kmeans",
        max_clusters=4,
        output_key="spatial_cluster_res",
        results_key="spatial_cluster_results",
    )

    assert "spatial_cluster_res" in out.obs.columns
    assert out.obs["spatial_cluster_res"].dtype.name == "category"
    assert "spatial_cluster_results" in out.uns
    results = out.uns["spatial_cluster_results"]
    assert "silhouette_scores" in results
    assert "best_n_clusters" in results


def test_run_spatial_cluster_hdbscan():
    from spatial_tk.core import spatial_clustering

    adata = _make_clusterable_adata()
    out = spatial_clustering.run_spatial_cluster(
        adata,
        cell_type_key="cell_type",
        mode="hdbscan",
        hdbscan_min_cluster_size=3,
        output_key="sc_hdbscan",
        results_key="sc_hdbscan_results",
    )

    results = out.uns["sc_hdbscan_results"]
    assert results["mode"] == "hdbscan"
    assert "n_clusters_found" in results
    assert "noise_fraction" in results


def test_run_spatial_cluster_force_n_clusters():
    from spatial_tk.core import spatial_clustering

    adata = _make_clusterable_adata()
    out = spatial_clustering.run_spatial_cluster(
        adata,
        cell_type_key="cell_type",
        mode="kmeans",
        max_clusters=5,
        force_n_clusters=3,
        results_key="forced_results",
        output_key="forced",
    )
    assert out.uns["forced_results"]["best_n_clusters"] == 3


def test_run_spatial_cluster_force_n_clusters_rejected_for_hdbscan():
    from spatial_tk.core import spatial_clustering

    adata = _make_clusterable_adata()
    with pytest.raises(ValueError):
        spatial_clustering.run_spatial_cluster(
            adata,
            cell_type_key="cell_type",
            mode="hdbscan",
            force_n_clusters=3,
        )


def test_run_spatial_cluster_resume_skips():
    from spatial_tk.core import spatial_clustering

    adata = _make_clusterable_adata()
    spatial_clustering.run_spatial_cluster(
        adata,
        cell_type_key="cell_type",
        mode="kmeans",
        max_clusters=4,
        output_key="spatial_cluster",
        results_key="spatial_cluster",
    )

    with patch.object(spatial_clustering, "run_spatial_kmeans") as mock_kmeans:
        spatial_clustering.run_spatial_cluster(
            adata,
            cell_type_key="cell_type",
            mode="kmeans",
            output_key="spatial_cluster",
            results_key="spatial_cluster",
            resume=True,
        )
    assert not mock_kmeans.called


def test_run_spatial_cluster_auto_computes_neighbors():
    from spatial_tk.core import spatial_clustering, spatial_neighbors

    adata = _make_clusterable_adata()
    saved = adata.obsp["spatial_connectivities"]
    del adata.obsp["spatial_connectivities"]

    def _fake_compute(adata, **kwargs):
        adata.obsp["spatial_connectivities"] = saved
        return adata

    with patch.object(
        spatial_neighbors, "compute_spatial_neighbors", side_effect=_fake_compute
    ) as mock_compute:
        out = spatial_clustering.run_spatial_cluster(
            adata,
            cell_type_key="cell_type",
            mode="kmeans",
            max_clusters=4,
            neighbor_k=6,
        )
    assert mock_compute.called
    assert "spatial_cluster" in out.obs.columns


def test_run_spatial_cluster_missing_neighbors_without_k_raises():
    from spatial_tk.core import spatial_clustering

    adata = _make_clusterable_adata()
    del adata.obsp["spatial_connectivities"]
    with pytest.raises(ValueError):
        spatial_clustering.run_spatial_cluster(
            adata,
            cell_type_key="cell_type",
            mode="kmeans",
        )

